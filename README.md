# Parallelization of Deep Learning Models
**Take-Home Assignment Implementation**  
*Data parallelism evaluation for ResNet-18 training on CIFAR-10 using PyTorch DDP*

---

## 📌 Overview
This repository implements and evaluates data parallelism for deep learning training on shared-memory systems. We compare:
- **Serial baseline**: Single-device training (GPU/CPU)
- **Parallel implementation**: `DistributedDataParallel` (DDP) with Gloo (CPU) and NCCL (GPU) backends
- **Key result**: 4.20× speedup on 6 CPU cores vs. serial CPU baseline with identical convergence behavior

---

## ⚙️ Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 6+ cores (Intel i7/i9 or AMD Ryzen) |
| GPU | None (CPU-only mode) | NVIDIA GPU with ≥4GB VRAM |
| RAM | 16 GB | 32 GB |
| Storage | 5 GB free space | NVMe SSD recommended |

> **Note for Windows users**: This project **requires WSL2 Ubuntu** due to PyTorch DDP limitations on native Windows. See [WSL2 Setup Guide](#wsl2-setup-guide-windows-only) below.

### Software Requirements
- Ubuntu 22.04 LTS (native or WSL2)
- Python 3.10+
- NVIDIA Driver ≥535.xx (for GPU support on WSL2)
- CUDA 11.8+ (optional but recommended for GPU)

---

## 🚀 Quick Start (WSL2 Ubuntu)

```bash
# 1. Clone repository (if not already cloned)
cd ~/projects
git clone https://github.com/hermelawesene/dl-parallel-assignment.git
cd dl-parallel-assignment

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify GPU access (optional but recommended)
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# 5. Run experiments
./run_experiments.sh  # Executes all 4 experiments below