# Parallel Deep Learning Training - Data Parallelism Implementation

## PROJECT OVERVIEW
ResNet-18 training on CIFAR-10 comparing serial baseline vs. data parallelism using PyTorch DDP.

## SETUP INSTRUCTIONS
1. Activate virtual environment:
```bash
source venv/bin/activate
```
2. Install dependencies:
```bash
pip install torch torchvision pandas matplotlib tqdm
```

## RUN EXPERIMENTS
- **Serial GPU baseline (5 epochs):**
```bash
python train.py --epochs 5 --batch-size 256 --device cuda --mode serial
```
- **DDP validation on 1 GPU (10 epochs):**
```bash
torchrun --nproc_per_node=1 train.py --epochs 10 --batch-size 256 --device cuda
```
- **CPU parallelism speedup demonstration (6 cores):**
```bash
torchrun --nproc_per_node=6 train.py --epochs 1 --batch-size 64 --device cpu
```

## GENERATE PLOTS
```bash
python plot_results.py
```
Outputs saved to:
- `report/figures/loss_accuracy_comparison.png`
- `report/figures/epoch_time_comparison.png`

## EXPECTED RESULTS (from your runs)
| Configuration | Epochs | Time per Epoch | Test Accuracy |
| --- | --- | --- | --- |
| Serial GPU (epoch 5) | 5 | 84.25s | 76.71% |
| DDP 1-GPU (epoch 5) | 5 | 99.51s | 73.14% (+18% overhead on single GPU) |
| Serial CPU (epoch 1) | 1 | 833.31s | 37.36% |
| DDP 6-Core CPU (epoch 1) | 1 |198.43s (~4.2x speedup vs serial CPU)| |

## CRITICAL NOTES
- Store project in WSL2 native filesystem (`~/projects/`) NOT `/mnt/c/` to avoid permission errors and slow I/O.
- Single-GPU DDP shows overhead (+18%) — real speedup requires >=2 physical GPUs.
- CPU parallelism demonstrates *REAL* speedup (4.2x on 6 cores), satisfying assignment requirements.
- All training logs saved to: `experiments/logs/`.

## REPRODUCIBILITY
All experiments use identical hyperparameters:
- **Model:** ResNet-18 (adapted for 32x32 images)
- **Optimizer:** SGD (`lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`)
- **Scheduler:** Cosine annealing
- **Random seed:** `42`
- **Dataset:** CIFAR-10 with standard augmentations.