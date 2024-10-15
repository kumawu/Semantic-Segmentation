# Use the NVIDIA PyTorch image as the base image
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables to prevent interactive prompts and cache installation
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/New_York

# Update and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    python3-opencv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies from the requirements file
RUN pip install --upgrade pip && pip install -r /workspace/requirements.txt

# Copy the inference script and colors.txt into the container
COPY infer.py /workspace/
COPY colors.txt /workspace/

# Set the working directory to /workspace
WORKDIR /workspace
