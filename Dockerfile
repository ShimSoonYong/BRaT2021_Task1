FROM nvcr.io/nvidia/pytorch:26.09-py3

LABEL maintainer="AI4MIA"
LABEL description="Development environment for AI4MIA Course"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  wget \
  curl \
  vim \
  libgl1 \
  libglib2.0-0 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
  monai[all] \
  pandas \
  matplotlib \
  seaborn \
  scikit-learn \
  scikit-image \
  opencv-python \
  pillow \
  tqdm \
  tensorboard

WORKDIR /workspace









