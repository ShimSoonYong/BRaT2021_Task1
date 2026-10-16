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
  ai4mia \
  python src/main.py \
  --model UNet \
  --batch_size 30 \
  --num_workers 10 \
  --epochs 1000 \
  --lr 1e-5 \
  --augment none
