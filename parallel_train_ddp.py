#!/usr/bin/env python3
"""
Minimal DDP implementation compatible with torchrun.
Fixed: sampler.set_epoch() now receives integer epoch (not float LR).
"""
import os
import time
import argparse
import csv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cifar10_dataloaders_ddp(batch_size, rank, world_size, train=True):
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
            root='./data', train=True, download=False, transform=transform_train)
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=2, pin_memory=True, drop_last=True)
        return loader, sampler
    else:
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform_test)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True)
        return loader

def create_resnet18_cifar():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def train_epoch_ddp(model, loader, criterion, optimizer, device, rank, world_size):
    """Train one epoch - sampler.set_epoch() called in main loop (not here)"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in tqdm(loader, desc=f"Rank {rank}", leave=False, disable=(rank != 0)):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_loss, correct, total = metrics.cpu().numpy()
    
    return total_loss / total, 100.0 * correct / total

@torch.no_grad()
def evaluate_ddp(model, loader, criterion, device, rank, world_size):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_loss, correct, total = metrics.cpu().numpy()
    
    return total_loss / total, 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser(description='DDP ResNet-18 Training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256, help='GLOBAL batch size')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-dir', type=str, default='./experiments/logs')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    if args.dry_run:
        args.epochs = 1
    
    # === DDP INIT ===
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    set_seed(args.seed + rank)
    
    # Model
    model = create_resnet18_cifar().to(device)
    model = DDP(model, device_ids=[rank])
    
    # Data
    per_gpu_batch = args.batch_size // world_size
    trainloader, sampler = get_cifar10_dataloaders_ddp(per_gpu_batch, rank, world_size, train=True)
    testloader = get_cifar10_dataloaders_ddp(per_gpu_batch, rank, world_size, train=False)
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Logging (rank 0 only)
    log_file = None
    if rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = os.path.join(args.log_dir, f'ddp_{world_size}gpu.csv')
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'epoch_time_sec'])
    
    # Training
    if rank == 0:
        print(f"\nDDP Training | GPUs: {world_size} | Global Batch: {args.batch_size} | Per-GPU: {per_gpu_batch}")
    
    best_acc = 0.0
    total_start = time.time() if rank == 0 else None
    
    for epoch in range(1, args.epochs + 1):
        # === CRITICAL FIX: set_epoch() MUST receive integer epoch ===
        sampler.set_epoch(epoch)  # Fixed: was incorrectly passing LR (float)
        
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch_ddp(model, trainloader, criterion, optimizer, device, rank, world_size)
        test_loss, test_acc = evaluate_ddp(model, testloader, criterion, device, rank, world_size)
        
        scheduler.step()
        dist.barrier()
        
        if rank == 0:
            epoch_time = time.time() - epoch_start
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc, epoch_time])
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.module.state_dict(), os.path.join(args.log_dir, f'best_model_ddp_{world_size}gpu.pth'))
            
            print(f"Epoch {epoch:2d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
    
    # Cleanup
    if rank == 0:
        total_time = time.time() - total_start
        print(f"\n{'='*70}")
        print(f"DDP TRAINING COMPLETE ({world_size} GPU)")
        print(f"Total time: {total_time:.2f}s | Best accuracy: {best_acc:.2f}%")
        print(f"Log: {log_file}")
        print(f"{'='*70}")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()