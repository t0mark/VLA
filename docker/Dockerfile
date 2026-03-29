FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# ── System update & base packages ────────────────────────────────────────────
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        git curl wget ca-certificates \
        software-properties-common \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# ── Locale ───────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8

# ── ROS Humble ───────────────────────────────────────────────────────────────
RUN add-apt-repository universe && apt-get update

RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb && \
    rm /tmp/ros2-apt-source.deb

RUN apt-get update && apt-get install -y \
        ros-humble-desktop \
        ros-dev-tools \
        ros-humble-cv-bridge \
        ros-humble-rqt* && \
    rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# ── Python 3.10 기본 도구 + 미디어/OpenCV 시스템 의존성 ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ── pip upgrade ──────────────────────────────────────────────────────────────
# NOTE: apt-get purge python3-numpy/scipy/sympy 금지
#   → ROS setup.bash가 내부 Python 스크립트로 PATH 설정; purge 시 setup.bash 동작 불가
#   → pip 패키지는 /usr/local/lib/python3.10 경로로 apt(/usr/lib/python3)보다 우선 적용됨
RUN pip3 install --upgrade pip

# ── apt sympy 충돌 해결 ───────────────────────────────────────────────────────
# apt sympy 1.9은 distutils 패키지라 pip가 제거 불가 → torch가 sympy>=1.13.3 요구
# --ignore-installed로 pip sympy를 /usr/local에 설치해 apt sympy를 덮어씀
RUN pip3 install "sympy>=1.13.3" --ignore-installed

# ── PyTorch 2.8.0 (CUDA 12.8, Python 3.10) ───────────────────────────────────
# CUDA 12.8은 Blackwell(sm_120 / RTX 5060)을 처음으로 공식 지원하는 버전
RUN pip3 install \
        torch==2.8.0 \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu128


# ── FlashAttention: 설치 안 함 ───────────────────────────────────────────────
# flash-attn 2.x: Blackwell(sm_120 / RTX 5060) 미지원 (no kernel image 오류)
# flash-attn-4: Python 3.10 미지원 (3.11+ 전용 베타)
# → 표준 어텐션(eager) 사용, ROS 추론 노드 1Hz 동작에는 문제 없음

# ── NaVILA core dependencies ─────────────────────────────────────────────────
RUN pip3 install \
    transformers==4.37.2 \
    "tokenizers>=0.14,<0.19" \
    sentencepiece==0.1.99 \
    accelerate \
    peft==0.9.0 \
    timm==0.9.12 \
    einops==0.6.1 \
    einops-exts==0.0.4

# bitsandbytes: 8GB VRAM에서 모델 실행을 위한 4-bit/8-bit 양자화
# RTX 5060 VRAM 8GB에서 Llama-3-8B BF16(~16GB) 직접 로드 불가
# 0.41.0은 CUDA 12.8 미지원 → 최신 버전 사용
RUN pip3 install bitsandbytes

RUN pip3 install \
    scipy \
    scikit-learn

RUN pip3 install \
    opencv-python==4.8.0.74 \
    decord==0.6.0 \
    pytorchvideo==0.1.5 \
    datasets==2.16.1 \
    webdataset==0.2.86 \
    openpyxl==3.1.2

RUN pip3 install \
    nltk==3.3 \
    pywsd==1.2.4 \
    openai==1.8.0 \
    requests \
    httpx==0.24.0 \
    uvicorn \
    fastapi \
    shortuuid \
    "pydantic>=1,<2" \
    "markdown2[all]" \
    gradio==3.35.2 \
    gradio_client==0.2.9 \
    tyro \
    loguru \
    hydra-core \
    pre-commit

# ── Git 기반 의존성 ──────────────────────────────────────────────────────────
RUN pip3 install "s2wrapper @ git+https://github.com/bfshi/scaling_on_scales"

# ── Training extras ──────────────────────────────────────────────────────────
RUN pip3 install \
    "deepspeed>=0.15.0" \
    wandb \
    ninja

# ── Evaluation extras ────────────────────────────────────────────────────────
RUN pip3 install \
    word2number \
    Levenshtein \
    lmms-eval \
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

# ── transformers/tokenizers 버전 고정 (lmms-eval 등이 업그레이드하므로 마지막에 재설치) ──
RUN pip3 install transformers==4.37.2 --force-reinstall --no-deps && \
    pip3 install "tokenizers>=0.14,<0.19" --force-reinstall

# ── Transformers 패치 적용 스크립트 ──────────────────────────────────────────
# environment_setup.sh 참조:
#   cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
#   cp -rv ./llava/train/deepspeed_replace/*    $site_pkg_path/deepspeed/
# 컨테이너 시작 후 /workspace 마운트가 완료된 뒤 실행 필요
RUN printf '#!/bin/bash\n\
site=$(python3 -c "import site; print(site.getsitepackages()[0])")\n\
if [ -d "/workspace/NaVILA/llava/train/transformers_replace" ]; then\n\
    cp -r /workspace/NaVILA/llava/train/transformers_replace/* $site/transformers/\n\
fi\n\
if [ -d "/workspace/NaVILA/llava/train/deepspeed_replace" ]; then\n\
    cp -r /workspace/NaVILA/llava/train/deepspeed_replace/* $site/deepspeed/\n\
fi\n' > /usr/local/bin/apply_navila_patches.sh && \
    chmod +x /usr/local/bin/apply_navila_patches.sh

RUN echo '\n# NaVILA transformers/deepspeed patch (컨테이너 시작 시 자동 적용)\napply_navila_patches.sh 2>/dev/null || true' >> /root/.bashrc

WORKDIR /workspace
