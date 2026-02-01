```markdown

# Parallel Deep Learning Training

Data parallelism implementation for ResNet-18 on CIFAR-10 using PyTorch DDP.

## Setup

```bash

python3 -m venv venv

source venv/bin/activate

pip install torch torchvision pandas matplotlib tqdm

```

## Run Experiments

\*\*Serial GPU baseline (5 epochs):\*\*

```bash

python train.py --epochs 5 --batch-size 256 --device cuda --mode serial

```

\*\*DDP validation (1 GPU, 10 epochs):\*\*

```bash

torchrun --nproc\_per\_node=1 train.py --epochs 10 --batch-size 256 --device cuda

```

\*\*CPU parallelism speedup (6 cores):\*\*

```bash

torchrun --nproc\_per\_node=6 train.py --epochs 1 --batch-size 64 --device cpu

```

## Generate Plots

```bash

python plot\_results.py

```

Outputs: \`report/figures/loss\_accuracy\_comparison.png\`, \`epoch\_time\_comparison.png\`

## Expected Results

| Configuration | Epoch Time | Speedup | Accuracy (Epoch 5) |

|---------------|------------|---------|---------------------|

| Serial GPU | 84.3 s | 1.00× | 76.71% |

| DDP 1-GPU | 99.5 s | 0.85×\* | 73.14% |

| Serial CPU | 833.3 s | 1.00× | 37.36% (epoch 1) |

| DDP 6-Core CPU| 198.4 s | 4.20× | ~68% (epoch 1) |

*\\\*Overhead on single GPU; real speedup requires ≥2 physical devices\*

## Notes

- Store project in WSL2 native filesystem (\`~/projects/\`), \*\*NOT\*\* \`/mnt/c/`

- CPU parallelism demonstrates \*\*real speedup\*\* (4.20× on 6 cores) satisfying assignment requirements

- All CSV logs saved to \`experiments/logs/\`

```