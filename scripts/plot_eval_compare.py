import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

official = {
    'mAP': 0.2698, 'NDS': 0.3894,
    'EPA_car': 0.598, 'EPA_ped': 0.290,
    'ADE_car': 0.787, 'FDE_car': 1.075, 'MR_car': 0.121,
    'L2_1s': 0.463, 'L2_2s': 0.763, 'L2_3s': 1.122,
    'col_1s': 0.00107, 'col_2s': 0.00244, 'col_3s': 0.00423,
}
ours = {
    'mAP': 0.2360, 'NDS': 0.3426,
    'EPA_car': 0.384, 'EPA_ped': 0.263,
    'ADE_car': 0.860, 'FDE_car': 1.194, 'MR_car': 0.133,
    'L2_1s': 0.685, 'L2_2s': 1.083, 'L2_3s': 1.632,
    'col_1s': 0.00186, 'col_2s': 0.00264, 'col_3s': 0.00947,
}

labels = ['Official VAD', 'Ours (+PhysHead)']
blue = '#4472C4'
orange = '#ED7D31'

def annotate(bars, fmt='{:.3f}'):
    for bar in bars:
        ax = bar.axes
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                fmt.format(bar.get_height()), ha='center', va='bottom',
                fontsize=10, fontweight='bold')

# ============ Fig 1: Detection ============
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
vals = [official['mAP'], ours['mAP']]
b = axes[0].bar(labels, vals, color=[blue, orange], width=0.5)
annotate(b)
axes[0].set_title('3D Detection mAP (higher better)', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 0.32)
axes[0].grid(axis='y', alpha=0.3)

vals = [official['NDS'], ours['NDS']]
b = axes[1].bar(labels, vals, color=[blue, orange], width=0.5)
annotate(b)
axes[1].set_title('3D Detection NDS (higher better)', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 0.45)
axes[1].grid(axis='y', alpha=0.3)

fig.suptitle('Stage-2 Model vs Official VAD — Detection', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/eval_detection_compare.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 2: Motion ============
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
vals = [official['EPA_car'], ours['EPA_car']]
b = axes[0].bar(labels, vals, color=[blue, orange], width=0.5)
annotate(b)
axes[0].set_title('Motion EPA_car (higher better)', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 0.7)
axes[0].grid(axis='y', alpha=0.3)

vals = [official['ADE_car'], ours['ADE_car']]
b = axes[1].bar(labels, vals, color=[blue, orange], width=0.5)
annotate(b)
axes[1].set_title('Motion ADE_car (lower better)', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 1.0)
axes[1].grid(axis='y', alpha=0.3)

fig.suptitle('Stage-2 Model vs Official VAD — Motion Prediction', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/eval_motion_compare.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 3: Planning L2 ============
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(3)
w = 0.35
off_l2 = [official['L2_1s'], official['L2_2s'], official['L2_3s']]
our_l2 = [ours['L2_1s'], ours['L2_2s'], ours['L2_3s']]
b1 = ax.bar(x - w/2, off_l2, w, label='Official VAD', color=blue)
b2 = ax.bar(x + w/2, our_l2, w, label='Ours (+PhysHead)', color=orange)
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['L2 @ 1s', 'L2 @ 2s', 'L2 @ 3s'], fontsize=12)
ax.set_ylabel('Planning Error (m)', fontsize=12)
ax.set_title('Stage-2 Model vs Official VAD — Planning L2 Error (lower better)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/eval_planning_compare.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 4: Collision ============
fig, ax = plt.subplots(figsize=(11, 5))
off_col = [official['col_1s'], official['col_2s'], official['col_3s']]
our_col = [ours['col_1s'], ours['col_2s'], ours['col_3s']]
b1 = ax.bar(x - w/2, off_col, w, label='Official VAD', color=blue)
b2 = ax.bar(x + w/2, our_col, w, label='Ours (+PhysHead)', color=orange)
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Collision @ 1s', 'Collision @ 2s', 'Collision @ 3s'], fontsize=12)
ax.set_ylabel('Collision Rate', fontsize=12)
ax.set_title('Stage-2 Model vs Official VAD — Collision Rate (lower better)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 0.012)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/eval_collision_compare.png', dpi=150, bbox_inches='tight')
plt.close()

print('All charts regenerated with English labels.')
