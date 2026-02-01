#!/usr/bin/env python3
"""
Unified training script: runs serial (1 process) or DDP (N processes) based on environment.
Supports CPU (Gloo) and GPU (NCCL) backends automatically.
"""
import os
import sys
import time
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Optional DDP imports (only used in parallel mode)
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cifar10_dataloaders(batch_size, use_ddp=False, rank=0, world_size=1, train=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if train:
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        if use_ddp:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=sampler,
                num_workers=2, pin_memory=False)  # pin_memory=False for CPU
            return loader, sampler
        else:
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=2, pin_memory=False)
            return loader, None
    else:
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=False)
        return loader, None

def create_resnet18_cifar():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def train_epoch(model, loader, criterion, optimizer, device, use_ddp=False, sampler=None, epoch=None):
    model.train()
    if use_ddp and sampler is not None and epoch is not None:
        sampler.set_epoch(epoch)
    
    total_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in tqdm(loader, desc="Train", leave=False, disable=(use_ddp and dist.get_rank() != 0)):
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    if use_ddp:
        metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total = metrics.cpu().numpy()
    
    return total_loss / total, 100.0 * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_ddp=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    if use_ddp:
        metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, correct, total = metrics.cpu().numpy()
    
    return total_loss / total, 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser(description='Unified Serial/DDP Training')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64, help='Per-process batch size for DDP; global for serial')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-dir', type=str, default='./experiments/logs')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    # Detect DDP mode (torchrun sets these env vars)
    use_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"[Rank {rank}] CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    if args.device == 'cuda':
        torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # Setup DDP if needed
    if use_ddp:
        backend = 'nccl' if args.device == 'cuda' else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
        set_seed(args.seed + rank)
    else:
        set_seed(args.seed)
        rank = 0
        world_size = 1
    
    # Model
    model = create_resnet18_cifar().to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank] if args.device == 'cuda' else None)
    
    # Data (per-process batch size for fair comparison)
    per_proc_batch = args.batch_size
    trainloader, sampler = get_cifar10_dataloaders(
        per_proc_batch, use_ddp=use_ddp, rank=rank, world_size=world_size, train=True)
    testloader, _ = get_cifar10_dataloaders(
        per_proc_batch, use_ddp=use_ddp, rank=rank, world_size=world_size, train=False)
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Logging (rank 0 only)
    mode = 'ddp' if use_ddp else 'serial'
    log_file = None
    if rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        suffix = f"{world_size}proc" if use_ddp else "baseline"
        log_file = os.path.join(args.log_dir, f'{mode}_{suffix}_{args.device}.csv')
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'epoch_time_sec', 'world_size'])
        print(f"\n{'='*70}")
        print(f"Mode: {'DDP (parallel)' if use_ddp else 'Serial (baseline)'} | Device: {args.device}")
        print(f"World size: {world_size} | Per-process batch: {per_proc_batch}")
        print(f"Log file: {log_file}")
        print(f"{'='*70}\n")
    
    # Training loop
    best_acc = 0.0
    total_start = time.time() if rank == 0 else None
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, device, 
            use_ddp=use_ddp, sampler=sampler, epoch=epoch)
        
        test_loss, test_acc = evaluate(model, testloader, criterion, device, use_ddp=use_ddp)
        
        scheduler.step()
        if use_ddp:
            dist.barrier()
        
        if rank == 0:
            epoch_time = time.time() - epoch_start
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc, epoch_time, world_size])
            
            if test_acc > best_acc:
                best_acc = test_acc
            
            mode_str = f"DDP ({world_size} proc)" if use_ddp else "Serial"
            print(f"[{mode_str}] Epoch {epoch:2d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
    
    # Cleanup and final report
    if use_ddp:
        dist.destroy_process_group()
    
    if rank == 0:
        total_time = time.time() - total_start
        avg_epoch_time = total_time / args.epochs
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"Mode: {'DDP (parallel)' if use_ddp else 'Serial (baseline)'} | Device: {args.device}")
        print(f"World size: {world_size} | Total time: {total_time:.2f}s | Avg epoch: {avg_epoch_time:.2f}s")
        print(f"Best test accuracy: {best_acc:.2f}%")
        print(f"Log: {log_file}")
        print(f"{'='*70}")
        
        # Save summary stats for easy comparison
        summary_file = os.path.join(args.log_dir, 'summary.csv')
        file_exists = os.path.exists(summary_file)
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['mode', 'world_size', 'device', 'avg_epoch_time_sec', 'best_acc', 'batch_per_proc'])
            writer.writerow([mode, world_size, args.device, avg_epoch_time, best_acc, per_proc_batch])

if __name__ == '__main__':
    main()