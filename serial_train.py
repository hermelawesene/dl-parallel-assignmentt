#!/usr/bin/env python3
"""
Serial baseline implementation for DL parallelization assignment.
Measures training time, loss convergence, and accuracy on CIFAR-10 using ResNet-18.
"""
import os
import time
import argparse
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Reproducibility setup (critical for fair comparison)
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable for timing accuracy

def get_cifar10_dataloaders(batch_size=256, num_workers=4):
    """Load CIFAR-10 with standard augmentations."""
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
    """ResNet-18 adapted for 32x32 CIFAR-10 images."""
    model = torchvision.models.resnet18(num_classes=10)
    # Replace first conv to handle 32x32 images (original ResNet expects 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool that downsamples too aggressively
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    """Single epoch training with explicit forward/backward steps."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # --- FORWARD PASS ---
        outputs = model(inputs)              # Compute predictions
        loss = criterion(outputs, labels)    # Compute loss
        
        # --- BACKWARD PASS ---
        optimizer.zero_grad()                # Clear previous gradients
        loss.backward()                      # Compute gradients via autograd
        
        # --- PARAMETER UPDATE ---
        optimizer.step()                     # Apply gradients to weights
        
        # --- METRICS ---
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Serial ResNet-18 Training on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-dir', type=str, default='./experiments/logs', help='Logging directory')
    parser.add_argument('--dry-run', action='store_true', help='Run single batch for debugging')
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, 'serial_baseline.csv')
    
    # Initialize components
    model = create_resnet18_cifar().to(device)
    trainloader, testloader = get_cifar10_dataloaders(
        batch_size=args.batch_size, 
        num_workers=4 if device.type == 'cuda' else 2
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # CSV logger initialization
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'epoch_time_sec'])
    
    # Training loop
    print(f"\nStarting serial training for {args.epochs} epochs...")
    best_acc = 0.0
    total_start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train one epoch (explicit forward/backward/update visible in function)
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        # LR scheduler step
        scheduler.step()
        
        # Timing
        epoch_time = time.time() - epoch_start
        
        # Log results
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc, epoch_time])
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model_serial.pth'))
        
        # Progress print
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")
        
        # Dry-run exit
        if args.dry_run:
            print("[DRY RUN] Exiting after 1 epoch")
            break
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*70}")
    print(f"SERIAL TRAINING COMPLETE")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Results logged to: {log_file}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()