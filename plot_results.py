#!/usr/bin/env python3
"""
Generate loss and accuracy comparison plots between serial and parallel implementations.
Outputs: loss_comparison.png, accuracy_comparison.png
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('report/figures', exist_ok=True)

# Load data
serial_gpu = pd.read_csv('experiments/logs/serial_baseline_gpu.csv')
ddp_1gpu = pd.read_csv('experiments/logs/ddp_1gpu.csv')
serial_cpu = pd.read_csv('experiments/logs/serial_baseline_cpu.csv')

# Truncate DDP to 5 epochs for fair comparison with serial GPU
ddp_1gpu_5ep = ddp_1gpu[ddp_1gpu['epoch'] <= 5].copy()

# Create figure with two subplots (loss + accuracy)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ===== PLOT 1: LOSS CURVES =====
ax1.plot(serial_gpu['epoch'], serial_gpu['train_loss'], 
         'o-', color='#2E86AB', linewidth=2, markersize=6, label='Serial GPU (Train)')
ax1.plot(serial_gpu['epoch'], serial_gpu['test_loss'], 
         's--', color='#2E86AB', linewidth=2, markersize=6, label='Serial GPU (Test)')
ax1.plot(ddp_1gpu_5ep['epoch'], ddp_1gpu_5ep['train_loss'], 
         'o-', color='#A23B72', linewidth=2, markersize=6, label='DDP 1-GPU (Train)')
ax1.plot(ddp_1gpu_5ep['epoch'], ddp_1gpu_5ep['test_loss'], 
         's--', color='#A23B72', linewidth=2, markersize=6, label='DDP 1-GPU (Test)')

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training and Test Loss Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xticks(np.arange(1, 6, 1))
ax1.set_xlim(0.8, 5.2)

# Add convergence annotations
ax1.annotate(f'Serial GPU: {serial_gpu["test_loss"].iloc[-1]:.3f}', 
             xy=(5, serial_gpu["test_loss"].iloc[-1]), 
             xytext=(4.2, 0.9),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1),
             fontsize=9, color='#2E86AB', fontweight='bold')
ax1.annotate(f'DDP 1-GPU: {ddp_1gpu_5ep["test_loss"].iloc[-1]:.3f}', 
             xy=(5, ddp_1gpu_5ep["test_loss"].iloc[-1]), 
             xytext=(4.2, 1.05),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1),
             fontsize=9, color='#A23B72', fontweight='bold')

# ===== PLOT 2: ACCURACY CURVES =====
ax2.plot(serial_gpu['epoch'], serial_gpu['train_acc'], 
         'o-', color='#2E86AB', linewidth=2, markersize=6, label='Serial GPU (Train)')
ax2.plot(serial_gpu['epoch'], serial_gpu['test_acc'], 
         's--', color='#2E86AB', linewidth=2, markersize=6, label='Serial GPU (Test)')
ax2.plot(ddp_1gpu_5ep['epoch'], ddp_1gpu_5ep['train_acc'], 
         'o-', color='#A23B72', linewidth=2, markersize=6, label='DDP 1-GPU (Train)')
ax2.plot(ddp_1gpu_5ep['epoch'], ddp_1gpu_5ep['test_acc'], 
         's--', color='#A23B72', linewidth=2, markersize=6, label='DDP 1-GPU (Test)')

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Training and Test Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
ax2.legend(loc='lower right', fontsize=10)
ax2.set_xticks(np.arange(1, 6, 1))
ax2.set_xlim(0.8, 5.2)
ax2.set_ylim(35, 85)

# Add accuracy annotations
ax2.annotate(f'Serial GPU: {serial_gpu["test_acc"].iloc[-1]:.2f}%', 
             xy=(5, serial_gpu["test_acc"].iloc[-1]), 
             xytext=(4.2, 70),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1),
             fontsize=9, color='#2E86AB', fontweight='bold')
ax2.annotate(f'DDP 1-GPU: {ddp_1gpu_5ep["test_acc"].iloc[-1]:.2f}%', 
             xy=(5, ddp_1gpu_5ep["test_acc"].iloc[-1]), 
             xytext=(4.2, 65),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1),
             fontsize=9, color='#A23B72', fontweight='bold')

# Adjust layout and save
plt.tight_layout()
plt.savefig('report/figures/loss_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: report/figures/loss_accuracy_comparison.png")

# ===== SEPARATE PLOT: EPOCH TIME COMPARISON =====
fig2, ax = plt.subplots(figsize=(8, 5))

# Extract epoch times (first 5 epochs)
serial_times = serial_gpu['epoch_time_sec'].values[:5]
ddp_times = ddp_1gpu_5ep['epoch_time_sec'].values

x = np.arange(1, 6)
width = 0.35

bars1 = ax.bar(x - width/2, serial_times, width, label='Serial GPU', 
               color='#2E86AB', alpha=0.9, edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, ddp_times, width, label='DDP 1-GPU', 
               color='#A23B72', alpha=0.9, edgecolor='black', linewidth=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Epoch Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Per-Epoch Training Time: Serial vs DDP (1 GPU)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.8)
ax.set_ylim(0, max(serial_times.max(), ddp_times.max()) * 1.15)

# Add overhead annotation
avg_serial = serial_times[1:].mean()  # Exclude epoch 1 warmup
avg_ddp = ddp_times[1:].mean()
overhead = ((avg_ddp - avg_serial) / avg_serial) * 100
ax.annotate(f'DDP overhead:\n+{overhead:.1f}%', 
            xy=(3, avg_ddp), xytext=(3.8, avg_ddp + 15),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('report/figures/epoch_time_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: report/figures/epoch_time_comparison.png")

# ===== GENERATE RESULTS TABLE =====
print("\n" + "="*70)
print("PERFORMANCE COMPARISON (Epochs 2-5 Average)")
print("="*70)

# Exclude epoch 1 (CUDA warmup)
serial_post_warmup = serial_gpu[serial_gpu['epoch'] > 1]
ddp_post_warmup = ddp_1gpu_5ep[ddp_5ep['epoch'] > 1]

serial_avg_time = serial_post_warmup['epoch_time_sec'].mean()
ddp_avg_time = ddp_post_warmup['epoch_time_sec'].mean()
speedup = serial_avg_time / ddp_avg_time
overhead_pct = ((ddp_avg_time - serial_avg_time) / serial_avg_time) * 100

print(f"Serial GPU avg epoch time:  {serial_avg_time:.2f}s")
print(f"DDP 1-GPU avg epoch time:   {ddp_avg_time:.2f}s")
print(f"Overhead:                   {overhead_pct:+.1f}%")
print(f"Speedup:                    {speedup:.2f}x")
print(f"\nFinal Test Accuracy (Epoch 5):")
print(f"  Serial GPU:  {serial_gpu['test_acc'].iloc[-1]:.2f}%")
print(f"  DDP 1-GPU:   {ddp_1gpu_5ep['test_acc'].iloc[-1]:.2f}%")
print(f"  Difference:  {abs(serial_gpu['test_acc'].iloc[-1] - ddp_1gpu_5ep['test_acc'].iloc[-1]):.2f}%")
print("="*70)

print("\n✅ All plots generated successfully. Use these figures in Section 2.3 and Section 6.3 of your report.")