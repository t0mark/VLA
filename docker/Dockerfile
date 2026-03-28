FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# ── System update ────────────────────────────────────────────────────────────
RUN apt update && apt upgrade -y && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        git curl wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ── Locale ───────────────────────────────────────────────────────────────────
RUN apt update && apt install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# ── ROS Humble ───────────────────────────────────────────────────────────────
RUN apt install -y software-properties-common && \
    add-apt-repository universe && \
    apt update && apt install -y curl

RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb && \
    rm /tmp/ros2-apt-source.deb

RUN apt update && apt upgrade -y && \
    apt install -y ros-humble-desktop ros-dev-tools ros-humble-rqt* && \
    rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# ── Python dependencies (system packages) ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ── pip upgrade ──────────────────────────────────────────────────────────────
RUN pip install --upgrade pip

# ── FlashAttention-2 ─────────────────────────────────────────────────────────
# Pre-compiled wheel for CUDA 12.x + PyTorch 2.x (cp312 for Python 3.12 in base image)
# If the base image uses Python 3.10, switch to the cp310 wheel.
RUN pip install packaging ninja && \
    pip install flash-attn --no-build-isolation

# ── NaVILA core dependencies ─────────────────────────────────────────────────
RUN pip install \
    # Vision / multimodal
    transformers==4.37.2 \
    tokenizers>=0.15.2 \
    sentencepiece==0.1.99 \
    accelerate==0.27.2 \
    peft==0.9.0 \
    bitsandbytes==0.41.0 \
    timm==0.9.12 \
    einops==0.6.1 \
    einops-exts==0.0.4 \
    # Data / vision utilities
    opencv-python==4.8.0.74 \
    decord==0.6.0 \
    pytorchvideo==0.1.5 \
    datasets==2.16.1 \
    webdataset==0.2.86 \
    # Numeric / ML utilities
    numpy==1.26.0 \
    scikit-learn==1.2.2 \
    openpyxl==3.1.2 \
    # NLP utilities
    nltk==3.3 \
    pywsd==1.2.4 \
    # API / serving
    openai==1.8.0 \
    requests \
    httpx==0.24.0 \
    uvicorn \
    fastapi \
    # Misc
    shortuuid \
    "pydantic>=1,<2" \
    "markdown2[all]" \
    gradio==3.35.2 \
    gradio_client==0.2.9 \
    tyro \
    loguru \
    hydra-core \
    pre-commit

# ── Git-based dependencies ────────────────────────────────────────────────────
RUN pip install "s2wrapper @ git+https://github.com/bfshi/scaling_on_scales"

# ── Training extras ──────────────────────────────────────────────────────────
RUN pip install \
    deepspeed==0.9.5 \
    wandb \
    ninja

# ── Evaluation extras ────────────────────────────────────────────────────────
RUN pip install \
    word2number \
    Levenshtein \
    lmms-eval \
    # evaluation/requirements.txt
    "attrs>=19.1.0" \
    dtw==1.4.0 \
    fastdtw==0.3.4 \
    gdown \
    jsonlines \
    lmdb \
    msgpack_numpy \
    networkx \
    "tqdm>=4.0.0" \
    "yacs>=0.1.5"

# ── Patch Transformers to NaVILA-compatible version ──────────────────────────
# Overrides default transformers with patched version from NaVILA source.
# Mount or COPY the NaVILA repo at /workspace/NaVILA before running this step,
# or bake the patches in at build time if the source is available:
# COPY NaVILA/llava/train/transformers_replace /tmp/transformers_replace
# RUN site=$(python -c 'import site; print(site.getsitepackages()[0])') && \
#     cp -rv /tmp/transformers_replace/* $site/transformers/

WORKDIR /workspace
