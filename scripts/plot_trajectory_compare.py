import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mmcv

# 加载两个模型的规划结果
official_plan = mmcv.load('test/VAD_tiny_stage_2_bgr/Sun_Aug_16_17_37_16_2026/pts_bbox/results_nusc.pkl')['plan_results']
ours_plan = mmcv.load('test/VAD_tiny_stage_2/Sun_Aug_16_18_24_36_2026/pts_bbox/results_nusc.pkl')['plan_results']

# 加载 val pkl 获取 GT 轨迹
val_data = mmcv.load('/root/autodl-tmp/data/nuscenes_pkls/vad_nuscenes_infos_temporal_val.pkl')
infos = val_data['infos']
token_to_gt = {info['token']: info.get('gt_ego_fut_trajs', None) for info in infos}

# 提取规划轨迹：选择 GT 命令对应的模式
def extract_traj(plan_result):
    trajs = plan_result[0]  # [3, 6, 2]
    cmd = plan_result[1]    # [1,1,1,3] one-hot
    cmd_idx = int(np.argmax(cmd.flatten()))
    return trajs[cmd_idx].numpy()  # [6, 2]

# 选取一些有 GT 轨迹的样本
common_tokens = [t for t in ours_plan.keys() if t in official_plan and t in token_to_gt and token_to_gt[t] is not None]

print(f'可对比样本数: {len(common_tokens)}')

# 选前 8 个样本做可视化（2x4 布局）
n_samples = 8
sample_tokens = common_tokens[:n_samples]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

blue = '#4472C4'
orange = '#ED7D31'
green = '#70AD47'

for i, token in enumerate(sample_tokens):
    ax = axes[i]
    # 官方轨迹
    off_traj = extract_traj(official_plan[token])
    # 我们的轨迹
    our_traj = extract_traj(ours_plan[token])
    # GT 轨迹
    gt_traj = token_to_gt[token]

    # 画轨迹（x 为纵向向前，y 为横向向左）
    ax.plot(off_traj[:, 0], off_traj[:, 1], 'o-', color=blue, linewidth=2, markersize=5, label='Official')
    ax.plot(our_traj[:, 0], our_traj[:, 1], 's-', color=orange, linewidth=2, markersize=5, label='Ours')
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], '^-', color=green, linewidth=2, markersize=5, label='GT')

    # 起点标记（ego 位置）
    ax.plot(0, 0, 'k*', markersize=15, label='Ego')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)

    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_title(f'Sample {i+1}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

# 图例只放第一张
axes[0].legend(fontsize=9, loc='best')

fig.suptitle('Planning Trajectory Comparison: Official VAD vs Ours (+Physical Head) vs GT', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/eval_trajectory_compare.png', dpi=150, bbox_inches='tight')
plt.close()
print('保存 charts/eval_trajectory_compare.png')

# 同时计算每个样本的 L2 误差对比
print('\n=== 各样本 L2@3s 误差对比 ===')
print(f'{"Sample":<10} {"Official":<12} {"Ours":<12}')
for i, token in enumerate(sample_tokens):
    off_traj = extract_traj(official_plan[token])
    our_traj = extract_traj(ours_plan[token])
    gt_traj = token_to_gt[token]
    off_err = np.linalg.norm(off_traj[-1] - gt_traj[-1])
    our_err = np.linalg.norm(our_traj[-1] - gt_traj[-1])
    print(f'Sample {i+1:<3} {off_err:<12.3f} {our_err:<12.3f}')
