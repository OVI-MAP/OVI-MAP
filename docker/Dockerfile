# =============================================================================
# OVI-MAP Docker image
# =============================================================================
# Notes:
#   - Host must have nvidia-container-toolkit installed for GPU support.
#   - The repo root is mounted at /workspace — edit code on the host.
#   - The perception worker (Python 3.10+) and reconstruction (Python 3.8+ROS)
#     both run inside the container to use shared-memory IPC.
# =============================================================================

FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# ---- Prevent interactive prompts during apt installs ----
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ---- 1. System packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    lsb-release \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

# ---- 2. ROS Noetic ----
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
 && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    -o Acquire::Retries=5 \
    -o Acquire::http::Timeout=30 \
    ros-noetic-desktop \
 && rm -rf /var/lib/apt/lists/*

# ---- 3. Dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wstool \
    protobuf-compiler \
    dh-autoreconf \
    python3-catkin-tools \
    python3-osrf-pycommon \
    libflann-dev \
    libgl1-mesa-dev \
    libvtk7-dev \
    pybind11-dev \
    libgflags-dev \
    libgoogle-glog-dev \
 && rm -rf /var/lib/apt/lists/*

# ---- 4. Miniconda ----
RUN curl -fsSL --retry 3 --retry-delay 15 -o /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash /tmp/miniconda.sh -b -p /opt/conda \
 && rm /tmp/miniconda.sh \
 && /opt/conda/bin/conda init bash

ENV PATH=/opt/conda/bin:$PATH

# Accept conda Terms of Service (required by newer conda versions)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ---- 5. Conda env: ovimap-py38 (Python 3.8, reconstruction + ROS) ----
COPY envs/ovimap-py38.yaml /tmp/ovimap-py38.yaml
RUN conda env create -f /tmp/ovimap-py38.yaml && conda clean -afy \
 && rm /tmp/ovimap-py38.yaml

# ---- 6. Conda env: ovimap-perception-py310 (Python 3.10, perception) ----
COPY envs/ovimap-perception-py310.yaml /tmp/ovimap-perception-py310.yaml
RUN conda env create -f /tmp/ovimap-perception-py310.yaml && conda clean -afy \
 && rm /tmp/ovimap-perception-py310.yaml \
 && /opt/conda/envs/ovimap-perception-py310/bin/pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision==0.22.1

# ---- 7. Final cleanup (pip cache, temp files) ----
RUN rm -rf /root/.cache/pip /tmp/* /var/tmp/* \
 && conda clean -afy 2>/dev/null || true

# ---- 8. Entrypoint ----
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Workspace is the mounted repo root
WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
