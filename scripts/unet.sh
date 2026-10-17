#!/bin/bash

docker run \
  --name unet \
  --ipc host \
  --ulimit memlock=-1 \
  --shm-size 32gb \
  --gpus all \
  --rm \
  --detach \
  --volume "$(pwd)":/workspace \
  -w /workspace \
  ai4mia \
  bash -c '
    augments=("none" "flip" "rotate90" "affine" "elastic" "random_crop")

    for aug in "${augments[@]}"; do
      echo "=================================================="
      echo ">>> Training with --augment $aug"
      echo "=================================================="

      python src/main.py \
        --model UNet \
        --batch_size 10 \
        --num_workers 16 \
        --epochs 100 \
        --lr 1e-5 \
        --augment "$aug"

      if [ $? -ne 0 ]; then
        echo "Error in augment: $aug"
        exit 1
      fi
    done

    echo "Every unet fitting completed!"
  '
