import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 无物理头 vs 有物理头（都 RGB 归一化，公平对比）
nophys = {
    'mAP': 0.2512, 'NDS': 0.3621,
    'EPA_car': 0.495, 'EPA_ped': 0.307,
    'ADE_car': 0.798, 'FDE_car': 1.100, 'MR_car': 0.120,
    'L2_1s': 0.456, 'L2_2s': 0.725, 'L2_3s': 1.054,
    'col_1s': 0.00117, 'col_2s': 0.00264, 'col_3s': 0.00456,
}
phys = {
    'mAP': 0.2360, 'NDS': 0.3426,
    'EPA_car': 0.384, 'EPA_ped': 0.263,
    'ADE_car': 0.860, 'FDE_car': 1.194, 'MR_car': 0.133,
    'L2_1s': 0.685, 'L2_2s': 1.083, 'L2_3s': 1.632,
    'col_1s': 0.00186, 'col_2s': 0.00264, 'col_3s': 0.00947,
}

labels = ['No PhysHead', 'With PhysHead']
green = '#70AD47'
red = '#C00000'

def annotate(bars, fmt='{:.3f}'):
    for bar in bars:
        ax = bar.axes
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                fmt.format(bar.get_height()), ha='center', va='bottom',
                fontsize=10, fontweight='bold')

# ============ Fig 1: Detection ============
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
vals = [nophys['mAP'], phys['mAP']]
b = axes[0].bar(labels, vals, color=[green, red], width=0.5)
annotate(b)
axes[0].set_title('Detection mAP (higher better)', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 0.30)
axes[0].grid(axis='y', alpha=0.3)

vals = [nophys['NDS'], phys['NDS']]
b = axes[1].bar(labels, vals, color=[green, red], width=0.5)
annotate(b)
axes[1].set_title('Detection NDS (higher better)', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 0.42)
axes[1].grid(axis='y', alpha=0.3)
fig.suptitle('Ablation: Physical Head Effect on Detection (fair RGB)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/ablation_detection.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 2: Motion ============
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
vals = [nophys['EPA_car'], phys['EPA_car']]
b = axes[0].bar(labels, vals, color=[green, red], width=0.5)
annotate(b)
axes[0].set_title('Motion EPA_car (higher better)', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 0.6)
axes[0].grid(axis='y', alpha=0.3)

vals = [nophys['ADE_car'], phys['ADE_car']]
b = axes[1].bar(labels, vals, color=[green, red], width=0.5)
annotate(b)
axes[1].set_title('Motion ADE_car (lower better)', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 1.0)
axes[1].grid(axis='y', alpha=0.3)
fig.suptitle('Ablation: Physical Head Effect on Motion (fair RGB)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/ablation_motion.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 3: Planning L2 ============
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(3)
w = 0.35
nophys_l2 = [nophys['L2_1s'], nophys['L2_2s'], nophys['L2_3s']]
phys_l2 = [phys['L2_1s'], phys['L2_2s'], phys['L2_3s']]
b1 = ax.bar(x - w/2, nophys_l2, w, label='No PhysHead', color=green)
b2 = ax.bar(x + w/2, phys_l2, w, label='With PhysHead', color=red)
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['L2 @ 1s', 'L2 @ 2s', 'L2 @ 3s'], fontsize=12)
ax.set_ylabel('Planning Error (m)', fontsize=12)
ax.set_title('Ablation: Physical Head Effect on Planning (lower better)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/ablation_planning.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 4: Collision ============
fig, ax = plt.subplots(figsize=(11, 5))
nophys_col = [nophys['col_1s'], nophys['col_2s'], nophys['col_3s']]
phys_col = [phys['col_1s'], phys['col_2s'], phys['col_3s']]
b1 = ax.bar(x - w/2, nophys_col, w, label='No PhysHead', color=green)
b2 = ax.bar(x + w/2, phys_col, w, label='With PhysHead', color=red)
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Collision @ 1s', 'Collision @ 2s', 'Collision @ 3s'], fontsize=12)
ax.set_ylabel('Collision Rate', fontsize=12)
ax.set_title('Ablation: Physical Head Effect on Collision (lower better)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 0.011)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/ablation_collision.png', dpi=150, bbox_inches='tight')
plt.close()

print('消融对比图已生成')
print(f"mAP: nophys={nophys['mAP']:.4f} vs phys={phys['mAP']:.4f}")
print(f"EPA_car: nophys={nophys['EPA_car']:.4f} vs phys={phys['EPA_car']:.4f}")
print(f"L2@1s: nophys={nophys['L2_1s']:.4f} vs phys={phys['L2_1s']:.4f}")
print(f"L2@3s: nophys={nophys['L2_3s']:.4f} vs phys={phys['L2_3s']:.4f}")
print(f"碰撞@3s: nophys={nophys['col_3s']:.4f} vs phys={phys['col_3s']:.4f}")
