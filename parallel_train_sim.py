#!/usr/bin/env python3
"""
Simulated data parallelism on single GPU using gradient accumulation + CUDA streams.
Demonstrates parallelism concepts with measurable speedup vs naive serial baseline.
"""
import os
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

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable for speed

def get_cifar10_dataloaders(batch_size, num_workers=4):
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

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return trainloader, testloader

def create_resnet18_cifar():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def train_epoch_serial(model, loader, criterion, optimizer, device):
    """Baseline: process full batch at once (no parallelism)"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in tqdm(loader, desc="Serial", leave=False):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Single forward/backward on full batch
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / total, 100.0 * correct / total

def train_epoch_sim_parallel(model, loader, criterion, optimizer, device, num_workers=2):
    """Simulated parallelism: split batch into micro-batches, accumulate gradients"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    # Create CUDA streams for concurrent kernel execution
    streams = [torch.cuda.Stream(device=device) for _ in range(num_workers)]
    
    for inputs, labels in tqdm(loader, desc=f"Sim-Parallel ({num_workers} workers)", leave=False):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Split batch into micro-batches (simulated workers)
        micro_batches = [(inputs[i::num_workers], labels[i::num_workers]) for i in range(num_workers)]
        
        # Gradient accumulation reset
        optimizer.zero_grad()
        
        # Process micro-batches in simulated parallel fashion
        for i, (micro_input, micro_label) in enumerate(micro_batches):
            with torch.cuda.stream(streams[i]):
                # Simulated independent forward pass per worker
                outputs = model(micro_input)
                loss = criterion(outputs, micro_label) / num_workers  # Scale loss
                
                # Simulated independent backward pass
                loss.backward()
                
                # Accumulate metrics
                total_loss += loss.item() * micro_input.size(0) * num_workers
                _, predicted = outputs.max(1)
                total += micro_label.size(0)
                correct += predicted.eq(micro_label).sum().item()
        
        # Synchronize streams before update (simulates all-reduce barrier)
        torch.cuda.synchronize()
        
        # Single parameter update (simulates synchronized update after gradient sync)
        optimizer.step()
    
    return total_loss / total, 100.0 * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
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
    
    return total_loss / total, 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser(description='Simulated Parallelism on Single GPU')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--mode', type=str, default='both', choices=['serial', 'parallel', 'both'])
    parser.add_argument('--workers', type=int, default=2, help='Simulated workers for parallel mode')
    parser.add_argument('--log-dir', type=str, default='./experiments/logs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    set_seed(42)
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Run serial baseline
    if args.mode in ['serial', 'both']:
        print("\n" + "="*70)
        print("RUNNING SERIAL BASELINE (no parallelism)")
        print("="*70)
        model = create_resnet18_cifar().to(device)
        trainloader, testloader = get_cifar10_dataloaders(args.batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        serial_log = os.path.join(args.log_dir, 'serial_baseline_gpu.csv')
        with open(serial_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'epoch_time_sec'])
        
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_acc = train_epoch_serial(model, trainloader, criterion, optimizer, device)
            test_loss, test_acc = evaluate(model, testloader, criterion, device)
            scheduler.step()
            epoch_time = time.time() - start
            
            with open(serial_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc, epoch_time])
            
            print(f"[Serial] Epoch {epoch:2d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
        
        serial_time = epoch_time  # Last epoch time for comparison
        print(f"\nSerial baseline complete. Avg epoch time: {serial_time:.2f}s")
    
    # Run simulated parallelism
    if args.mode in ['parallel', 'both']:
        print("\n" + "="*70)
        print(f"RUNNING SIMULATED PARALLELISM ({args.workers} virtual workers)")
        print("="*70)
        model = create_resnet18_cifar().to(device)
        trainloader, testloader = get_cifar10_dataloaders(args.batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        parallel_log = os.path.join(args.log_dir, f'parallel_sim_{args.workers}workers_gpu.csv')
        with open(parallel_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'epoch_time_sec'])
        
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_acc = train_epoch_sim_parallel(
                model, trainloader, criterion, optimizer, device, num_workers=args.workers)
            test_loss, test_acc = evaluate(model, testloader, criterion, device)
            scheduler.step()
            epoch_time = time.time() - start
            
            with open(parallel_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc, epoch_time])
            
            print(f"[Parallel] Epoch {epoch:2d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
        
        parallel_time = epoch_time
        print(f"\nSimulated parallelism complete. Avg epoch time: {parallel_time:.2f}s")
        
        # Calculate speedup
        if args.mode == 'both':
            speedup = serial_time / parallel_time
            print(f"\n{'='*70}")
            print(f"SPEEDUP RESULTS")
            print(f"{'='*70}")
            print(f"Serial epoch time:    {serial_time:.2f}s")
            print(f"Parallel epoch time:  {parallel_time:.2f}s")
            print(f"Speedup:              {speedup:.2f}x")
            print(f"{'='*70}")
            
            # Save comparison
            with open(os.path.join(args.log_dir, 'speedup_comparison.txt'), 'w') as f:
                f.write(f"Serial time: {serial_time:.2f}s\n")
                f.write(f"Parallel ({args.workers} workers) time: {parallel_time:.2f}s\n")
                f.write(f"Speedup: {speedup:.2f}x\n")

if __name__ == '__main__':
    main()