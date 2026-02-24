pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install \
  "transformers==4.37.2" \
  "sentencepiece==0.1.99" \
  "accelerate==0.27.2" \
  "numpy==1.26.0" \    # cv_bridge 호환 (반드시 <2.0)
  "einops==0.6.1" \
  "decord==0.6.0" \
  "opencv-python==4.8.0.74" \
  "Pillow" \
  "s2wrapper@git+https://github.com/bfshi/scaling_on_scales"

pip install "deepspeed==0.9.5"

pip install datasets

pip install ninja
