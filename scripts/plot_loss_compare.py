import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_log(path):
    """解析训练日志，提取每个 iter 的损失值"""
    losses = []
    with open(path, 'r') as f:
        for line in f:
            if 'Epoch [' not in line or 'loss:' not in line:
                continue
            # 提取 epoch 和 iter
            em = re.search(r'Epoch \[(\d+)\]\[(\d+)/(\d+)\]', line)
            if not em:
                continue
            epoch = int(em.group(1))
            it = int(em.group(2))
            per_epoch = int(em.group(3))
            global_step = (epoch - 1) * per_epoch + it
            
            # 提取所有 loss 字段
            vals = {}
            for k in ['loss_cls', 'loss_bbox', 'loss_traj', 'loss_traj_cls',
                      'loss_map_cls', 'loss_map_pts', 'loss_map_dir',
                      'loss_plan_reg', 'loss_plan_bound', 'loss_plan_col', 'loss_plan_dir',
                      'loss_phys_kappa', 'loss_phys_omega', 'loss_phys_aeb',
                      'loss_comply_kappa', 'loss_comply_omega',
                      'loss', 'grad_norm']:
                m = re.search(rf'\b{k}:\s*([\d.eE+-]+)', line)
                if m:
                    try:
                        vals[k] = float(m.group(1))
                    except:
                        pass
            if 'loss' in vals:
                vals['step'] = global_step
                vals['epoch'] = epoch
                losses.append(vals)
    return losses

phys = parse_log('work_dirs/stage2_full/20260815_094853.log')
nophys = parse_log('work_dirs/stage2_nophys/20260816_184543.log')

print(f'phys 日志: {len(phys)} 条, epoch {phys[0]["epoch"]}-{phys[-1]["epoch"]}')
print(f'nophys 日志: {len(nophys)} 条, epoch {nophys[0]["epoch"]}-{nophys[-1]["epoch"]}')

def get_arr(data, key):
    return np.array([d[key] for d in data if key in d])

def get_step(data):
    return np.array([d['step'] for d in data])

def smooth(x, alpha=0.05):
    s = np.zeros_like(x)
    s[0] = x[0]
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i-1]
    return s

# ============ 图1: 物理头损失（phys 特有） ============
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# 物理监督损失
ax = axes[0]
step = get_step(phys)
for key, color, name in [
    ('loss_phys_kappa', '#C00000', 'loss_phys_kappa'),
    ('loss_phys_omega', '#ED7D31', 'loss_phys_omega'),
    ('loss_phys_aeb', '#4472C4', 'loss_phys_aeb'),
]:
    if key in phys[0]:
        ax.plot(step, smooth(get_arr(phys, key)), color=color, linewidth=0.8, label=name)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Physical Head Supervision Losses (phys model only)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 合规损失
ax = axes[1]
for key, color, name in [
    ('loss_comply_kappa', '#C00000', 'loss_comply_kappa'),
    ('loss_comply_omega', '#ED7D31', 'loss_comply_omega'),
]:
    if key in phys[0]:
        ax.plot(step, smooth(get_arr(phys, key)), color=color, linewidth=0.8, label=name)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Compliance Losses (phys model only)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/loss_phys_head.png', dpi=150, bbox_inches='tight')
plt.close()
print('保存 charts/loss_phys_head.png')

# ============ 图2: 规划损失对比 ============
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

ax = axes[0]
ax.plot(get_step(phys), smooth(get_arr(phys, 'loss_plan_reg')), color='#C00000', linewidth=0.8, label='phys (with head)')
ax.plot(get_step(nophys), smooth(get_arr(nophys, 'loss_plan_reg')), color='#70AD47', linewidth=0.8, label='nophys (no head)')
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Planning Loss (loss_plan_reg)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(get_step(phys), smooth(get_arr(phys, 'loss_plan_col')), color='#C00000', linewidth=0.8, label='phys (with head)')
ax.plot(get_step(nophys), smooth(get_arr(nophys, 'loss_plan_col')), color='#70AD47', linewidth=0.8, label='nophys (no head)')
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Planning Collision Loss (loss_plan_col)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/loss_planning_compare.png', dpi=150, bbox_inches='tight')
plt.close()
print('保存 charts/loss_planning_compare.png')

# ============ 图3: 检测损失对比 ============
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
ax = axes[0]
ax.plot(get_step(phys), smooth(get_arr(phys, 'loss_cls')), color='#C00000', linewidth=0.8, label='phys (with head)')
ax.plot(get_step(nophys), smooth(get_arr(nophys, 'loss_cls')), color='#70AD47', linewidth=0.8, label='nophys (no head)')
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Detection Classification Loss (loss_cls)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(get_step(phys), smooth(get_arr(phys, 'loss_bbox')), color='#C00000', linewidth=0.8, label='phys (with head)')
ax.plot(get_step(nophys), smooth(get_arr(nophys, 'loss_bbox')), color='#70AD47', linewidth=0.8, label='nophys (no head)')
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Detection BBox Loss (loss_bbox)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/loss_detection_compare.png', dpi=150, bbox_inches='tight')
plt.close()
print('保存 charts/loss_detection_compare.png')

# ============ 图4: 总损失对比 ============
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(get_step(phys), smooth(get_arr(phys, 'loss')), color='#C00000', linewidth=0.8, label='phys (with head)')
ax.plot(get_step(nophys), smooth(get_arr(nophys, 'loss')), color='#70AD47', linewidth=0.8, label='nophys (no head)')
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Total Loss', fontsize=12)
ax.set_title('Total Loss Comparison: phys vs nophys', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/loss_total_compare.png', dpi=150, bbox_inches='tight')
plt.close()
print('保存 charts/loss_total_compare.png')

# ============ 图5: 运动损失 + 地图损失对比 ============
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
ax = axes[0]
ax.plot(get_step(phys), smooth(get_arr(phys, 'loss_traj')), color='#C00000', linewidth=0.8, label='phys (with head)')
ax.plot(get_step(nophys), smooth(get_arr(nophys, 'loss_traj')), color='#70AD47', linewidth=0.8, label='nophys (no head)')
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Motion Prediction Loss (loss_traj)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(get_step(phys), smooth(get_arr(phys, 'loss_map_pts')), color='#C00000', linewidth=0.8, label='phys (with head)')
ax.plot(get_step(nophys), smooth(get_arr(nophys, 'loss_map_pts')), color='#70AD47', linewidth=0.8, label='nophys (no head)')
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Map Prediction Loss (loss_map_pts)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/loss_motion_map_compare.png', dpi=150, bbox_inches='tight')
plt.close()
print('保存 charts/loss_motion_map_compare.png')

# ============ 输出统计摘要 ============
print('\n=== 损失统计摘要（最后 1000 iter 平均） ===')
for key in ['loss', 'loss_cls', 'loss_bbox', 'loss_traj', 'loss_map_pts',
            'loss_plan_reg', 'loss_plan_col', 'loss_phys_kappa', 'loss_phys_omega',
            'loss_comply_kappa', 'loss_comply_omega']:
    if key in phys[0]:
        p = np.mean(get_arr(phys, key)[-1000:])
    else:
        p = float('nan')
    if key in nophys[0]:
        n = np.mean(get_arr(nophys, key)[-1000:])
    else:
        n = float('nan')
    print(f'  {key:<20} phys={p:<10.4f} nophys={n:<10.4f}')
