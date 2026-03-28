# na_vila_ros

NaVILA (RSS 2025) VLA 모델을 ROS 2에서 실시간으로 실행하는 추론 패키지.

카메라 이미지와 자연어 지시를 받아 내비게이션 명령(`cmd_vel`)을 출력합니다.

---

## 시스템 요구사항

| 항목 | 요구사항 |
|---|---|
| OS | Ubuntu 22.04 |
| ROS 2 | Humble |
| Python | 3.10 |
| GPU | NVIDIA GPU (VRAM 8GB 이상 권장, 4-bit 양자화 기준) |
| CUDA | 12.8 (현재 설치 기준) |
| NVIDIA 드라이버 | 570 이상 |

> **다른 GPU에서 실행하는 경우** (예: RTX 4070, CUDA 12.1):
> PyTorch와 flash-attention 버전을 해당 CUDA 버전에 맞춰 변경해야 합니다. 아래 설치 단계를 참고하세요.


## 워크스페이스 구성

```bash
# ros_ws 루트에서 실행
mkdir models && cd models

git clone https://huggingface.co/a8cheng/navila-llama3-8b-8f
git lfs install

cd .. && mkdir src && cd src
git clone https://github.com/t0mark/VLA
```

다운로드 후 구조:

```
ros_ws/
├── models/
│   └── navila-llama3-8b-8f/
│       ├── config.json
│       ├── llm/
│       ├── vision_tower/
│       └── mm_projector/
└── src/
```

런치 파일이 `ros_ws/models/navila-llama3-8b-8f`를 기본 경로로 자동 탐지합니다.

---

## 설치

### 1. Python 의존성 설치

**CUDA 12.8 환경 (RTX 5060 Ti 등 Blackwell GPU):**

```bash
# PyTorch 2.7 + CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# FlashAttention 2.8.3 (CUDA 12, Torch 2.7, Python 3.10)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

**CUDA 12.1 환경 (RTX 4070 등 Ada Lovelace GPU):**

```bash
# PyTorch 2.3 + CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

# FlashAttention: 아키텍처에 맞는 wheel 선택
# https://github.com/Dao-AILab/flash-attention/releases
# 또는 소스 빌드 (시간 소요):
# pip install flash-attn --no-build-isolation
```

**공통 Python 패키지:**

```bash
pip install \
  "transformers==4.37.2" \
  "sentencepiece==0.1.99" \
  "accelerate==0.27.2" \
  "bitsandbytes>=0.41.0" \
  "numpy==1.26.0" \
  "einops==0.6.1" \
  "decord==0.6.0" \
  "opencv-python==4.8.0.74" \
  "Pillow" \
  "s2wrapper@git+https://github.com/bfshi/scaling_on_scales"

pip install "deepspeed==0.9.5"
pip install datasets ninja
```

### 2. ROS 2 의존성 설치

```bash
sudo apt install \
  ros-humble-cv-bridge \
  ros-humble-sensor-msgs \
  ros-humble-std-msgs \
  ros-humble-geometry-msgs
```

### 3. 워크스페이스 빌드

```bash
cd ros_ws
colcon build --packages-select na_vila_ros
source install/setup.bash
```

---

## 실행

### 기본 실행 (모델 경로 자동 탐지)

```bash
ros2 launch na_vila_ros navila.launch.py

# CLI 입력 노드 (별도 터미널)
ros2 run na_vila_ros chat_prompt_node
```

---

## 파라미터 설정

`config/navila.yaml` 에서 조정합니다:

```yaml
navila_node:
  ros__parameters:
    num_frames: 4          # 추론에 사용할 프레임 수 (최대 8)
    inference_hz: 1.0      # 추론 주기 (Hz)
    max_new_tokens: 256    # 생성 최대 토큰 수
    load_4bit: true        # 4-bit NF4 양자화 (VRAM 8GB 이하 권장)
    history_interval: 1.0  # 히스토리 프레임 샘플링 간격 (초)

    image_topic: "/camera/camera/color/image_raw"
    instruction_topic: "/navila/prompt"
    command_topic: "/navila/output"
```

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `num_frames` | `4` | 히스토리 (N-1)장 + 현재 1장 |
| `inference_hz` | `1.0` | 초당 추론 횟수 |
| `max_new_tokens` | `256` | 텍스트 생성 최대 길이 |
| `load_4bit` | `true` | NF4 4-bit 양자화 활성화 |
| `history_interval` | `1.0` | 히스토리 버퍼 샘플링 간격 (초) |

---

## 모델 아키텍처

| 컴포넌트 | 내용 |
|---|---|
| 비전 인코더 | Google SigLIP (`siglip-so400m-patch14-384`) |
| 프로젝터 | MLP Downsample (2×2 spatial pooling → Linear → GELU → Linear) |
| 언어 모델 | Llama-3-8B |
| 입력 | 8프레임 (히스토리 7 + 현재 1), FP16 |
| 양자화 | NF4 4-bit (BitsAndBytes, double quant) |
| 대화 템플릿 | `llama_3` |

**VRAM 사용량 (4-bit 양자화, `num_frames=4`):**

| GPU | VRAM | 동작 여부 |
|---|---|---|
| RTX 5060 Ti (16GB) | ~6-7GB | 정상 동작 |
| RTX 4070 (12GB) | ~6-7GB | 정상 동작 |
| RTX 3080 (10GB) | ~6-7GB | 정상 동작 |

