import torch
import numpy as np
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import GradScaler, autocast
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU


class CheckpointModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return checkpoint(self.model, x, use_reentrant=False)


class Trainer:
    """Modular segmentation learner adopting a custom control loop

    with automated checkpointing, curve plotting,
    metric tracking, and visual artifact logging under logs/[model_name]/[category].
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_function: torch.nn.Module,
        test_loader: DataLoader | None = None,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        use_amp: bool = True,
        target_batch_size: int = 1,
    ):

        self.model = CheckpointModel(model).to(device)
        self.model.compile()
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = torch.device(device)
        self.lr_scheduler = lr_scheduler
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        # Gradient Accumulation Logic
        self.physical_batch_size = self.train_loader.batch_size or 1
        self.target_batch_size = max(target_batch_size, self.physical_batch_size)
        self.accum_steps = self.target_batch_size // self.physical_batch_size

        # Directory structure setup: logs/[model_name]/[category]
        self.base_log_dir = Path("logs") / self.model_name
        self.ckpt_dir = self.base_log_dir / "checkpoints"
        self.plot_dir = self.base_log_dir / "plots"
        self.vis_dir = self.base_log_dir / "visualizations"
        self.test_dir = self.base_log_dir / "test_results"

        for d in [self.ckpt_dir, self.plot_dir, self.vis_dir, self.test_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )
        self.hausdorff_metric = HausdorffDistanceMetric(
            include_background=True, percentile=95.0, reduction="mean"
        )
        self.iou_metric = MeanIoU(include_background=True, reduction="mean")

        # History tracking buffer
        self.history = {
            "epochs": [],
            "train_loss": [],
            "val_mean_dice": [],
            "val_hausdorff": [],
            "val_mean_iou": [],
        }
        self.best_metric = -1.0

    def train(self) -> float:
        """Executes a single training epoch with automatic gradient accumulation."""
        self.model.train()
        epoch_loss = 0.0
        self.optimizer.zero_grad()

        num_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(self.train_loader):
            images = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.loss_function(outputs, labels)
                    loss = loss / self.accum_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                loss = loss / self.accum_steps
                loss.backward()

            # Track full unscaled loss value
            epoch_loss += loss.item() * self.accum_steps

            # Apply optimizer step at accumulation boundaries or final batch
            if (batch_idx + 1) % self.accum_steps == 0 or (
                batch_idx + 1
            ) == num_batches:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

        return epoch_loss / num_batches

    def val(self) -> dict:
        self.model.eval()

        with torch.inference_mode():
            for batch_data in self.val_loader:
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)

                preds = (torch.sigmoid(outputs) > 0.5).float()

                self.dice_metric(y_pred=preds, y=labels)
                self.hausdorff_metric(y_pred=preds, y=labels)
                self.iou_metric(y_pred=preds, y=labels)

        val_dice = self.dice_metric.aggregate().item()
        val_hd = self.hausdorff_metric.aggregate().item()
        val_iou = self.iou_metric.aggregate().item()

        self.dice_metric.reset()
        self.hausdorff_metric.reset()
        self.iou_metric.reset()

        return {
            "val_mean_dice": val_dice,
            "val_hausdorff": val_hd if not np.isnan(val_hd) else 0.0,
            "val_mean_iou": val_iou,
        }

    def test(self, target_loader: DataLoader | None = None) -> dict:
        loader = target_loader or self.test_loader
        if loader is None:
            print("[Test] No test loader provided.")
            return {}

        print(f"\n[Test] Running independent evaluation on {len(loader)} batches...")
        self.model.eval()

        with torch.inference_mode():
            for batch_data in loader:
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)

                preds = (torch.sigmoid(outputs) > 0.5).float()

                self.dice_metric(y_pred=preds, y=labels)
                self.hausdorff_metric(y_pred=preds, y=labels)
                self.iou_metric(y_pred=preds, y=labels)

        test_results = {
            "test_mean_dice": self.dice_metric.aggregate().item(),
            "test_hausdorff": self.hausdorff_metric.aggregate().item(),
            "test_mean_iou": self.iou_metric.aggregate().item(),
        }

        self.dice_metric.reset()
        self.hausdorff_metric.reset()
        self.iou_metric.reset()

        report_path = self.test_dir / "test_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Test Results for Model: {self.model_name}\n")
            f.write("=" * 40 + "\n")
            for k, v in test_results.items():
                print(f"[Test] {k}: {v:.5f}")
                f.write(f"{k}: {v:.5f}\n")

        return test_results

    def fit(self, epochs: int, device: str | torch.device | None = None, **kwargs):
        if device:
            self.device = torch.device(device)
            self.model.to(self.device)

        print(
            f"Starting training lifecycle for: {self.model_name} over {epochs} epochs."
        )
        print(
            f"Target Batch Size: {self.target_batch_size} | Physical GPU Batch Size: {self.physical_batch_size}"
        )
        print(f"Gradient Accumulation Steps internally set to: {self.accum_steps}")
        print(f"Logs storing at: {self.base_log_dir}\n")

        for epoch in range(1, epochs + 1):
            train_loss = self.train()
            val_metrics = self.val()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(train_loss)
            for k, v in val_metrics.items():
                self.history[k].append(v)

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | "
                f"Val Dice: {val_metrics['val_mean_dice']:.4f} | Val IoU: {val_metrics['val_mean_iou']:.4f}"
            )

            current_metric = val_metrics["val_mean_dice"]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                best_ckpt = self.ckpt_dir / "best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_metric": self.best_metric,
                    },
                    best_ckpt,
                )

            self._save_curves()
            self._save_visual_examples(epoch)

        print("\nTraining complete. Running final test evaluation...")
        self.test()

    def _save_curves(self):
        epochs = self.history["epochs"]
        if not epochs:
            return

        plt.figure(figsize=(10, 7))

        plt.subplot(2, 1, 1)
        plt.plot(
            epochs,
            self.history["train_loss"],
            label="Train Loss",
            color="crimson",
            lw=1.5,
        )
        plt.ylabel("Loss")
        plt.title(f"Optimization Metrics - {self.model_name}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper right")

        plt.subplot(2, 1, 2)
        plt.plot(
            epochs,
            self.history["val_mean_dice"],
            label="Val Dice",
            color="tab:blue",
            lw=1.5,
        )
        plt.plot(
            epochs,
            self.history["val_mean_iou"],
            label="Val IoU",
            color="tab:green",
            lw=1.5,
        )
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(self.plot_dir / "metrics_curve.png", dpi=500)
        plt.close()

    def _save_visual_examples(self, epoch: int):
        self.model.eval()
        with torch.no_grad():
            for batch_data in self.val_loader:
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)

                preds = (torch.sigmoid(outputs) > 0.5).float()

                img_np = images[0].detach().cpu().numpy()
                lbl_np = labels[0].detach().cpu().numpy()
                pred_np = preds[0].detach().cpu().numpy()

                if img_np.ndim == 4:
                    mid = img_np.shape[1] // 2
                    img_2d = img_np[0, mid]
                    lbl_2d = lbl_np[0, mid]
                    pred_2d = pred_np[0, mid]
                else:
                    img_2d = img_np[0]
                    lbl_2d = lbl_np[0]
                    pred_2d = pred_np[0]

                intersection = np.logical_and(pred_2d > 0.5, lbl_2d > 0.5).sum()
                union = (pred_2d > 0.5).sum() + (lbl_2d > 0.5).sum()
                sample_dice = (2.0 * intersection) / (union + 1e-8)

                fig, axes = plt.subplots(1, 3, figsize=(11, 4))
                axes[0].imshow(img_2d, cmap="gray")
                axes[0].set_title("Input Image")
                axes[0].axis("off")

                axes[1].imshow(lbl_2d, cmap="jet")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(pred_2d, cmap="jet")
                axes[2].set_title(f"Prediction (Dice: {sample_dice:.3f})")
                axes[2].axis("off")

                plt.suptitle(f"Epoch {epoch} Sample", fontsize=12)
                plt.tight_layout()
                plt.savefig(self.vis_dir / f"epoch_{epoch:03d}_vis.png", dpi=500)
                plt.close()
                break
