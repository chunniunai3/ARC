#!/usr/bin/env bash
# 在 Linux GPU 服务器上执行
set -euo pipefail

# 1. Clone 代码（假设你的 git repo 在 GitHub）
git clone <YOUR_REPO_URL> ARC
cd ARC

# 2. 创建 conda 环境
conda create -n arc python=3.11 -y
conda activate arc

# 3. 安装 PyTorch（根据你的 CUDA 版本调整）
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 4. 安装其他依赖
pip install numpy matplotlib pillow opencv-python

# 5. 下载 ARC-AGI-1 数据
# 方法 A: 从 Kaggle 下载
# pip install kaggle
# kaggle competitions download -c arc-prize-2024
# unzip arc-prize-2024.zip -d data/ARC-AGI/

# 方法 B: 直接复制本地数据（推荐）
# 在你本地执行: scp -r data/ARC-AGI user@server:~/ARC/data/

# 6. 复制 checkpoint（本地执行）
# scp results/checkpoints/varc_exp3_base.pt user@server:~/ARC/results/checkpoints/
# scp results/checkpoints/varc_rearc.pt user@server:~/ARC/results/checkpoints/

# 7. 验证
python -c "from src.data_loader import load_tasks; tasks=load_tasks('training',5); print(f'OK: {len(tasks)} tasks')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"

echo "Setup complete!"
