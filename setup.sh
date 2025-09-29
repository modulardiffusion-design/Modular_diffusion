#!/bin/bash
set -e  # 出错立即退出

ENV_NAME="d4rl_kitchen"

echo ">>> [1/6] 创建 conda 环境: $ENV_NAME (Python 3.8)"
conda create -y -n $ENV_NAME python=3.8
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo ">>> [2/6] 安装系统依赖..."
sudo apt-get update && sudo apt-get install -y \
    libgl1-mesa-dev \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    pkg-config \
    python3-dev \
    build-essential \
    ffmpeg

echo ">>> [3/6] 安装 MuJoCo 2.1.0..."
mkdir -p ~/.mujoco
wget -q https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz
tar -xvzf /tmp/mujoco210.tar.gz -C ~/.mujoco/

echo ">>> [4/6] 安装 Cython (固定版本)"
pip install "cython==0.29.36"

echo ">>> [5/6] 安装 mujoco-py 并强制编译"
MUJOCO_PY_FORCE_REBUILD=1 pip install --no-cache-dir --force-reinstall mujoco-py==2.1.2.14

echo ">>> [6/6] 安装 Python 依赖 (跳过依赖解决，避免破坏前面编译)"
pip install -r requirements.txt --no-deps

echo ">>> ✅ 安装完成！验证 mujoco-py:"
echo "    conda activate $ENV_NAME && python -c 'import mujoco_py; print(mujoco_py.__version__)'"
