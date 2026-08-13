"""
model/trajectory_utils.py — 轨迹几何计算 + 合规损失

compute_curvature: 三点法计算离散轨迹上各中间点的曲率
trajectory_compliance_loss: Hinge loss, 惩罚轨迹违反物理头预测的安全边界
"""
import torch


def compute_curvature(trajs: torch.Tensor) -> torch.Tensor:
    """
    三点法离散曲率: κ = 4·Area(△) / (a·b·c)
    三角形面积用 Heron 公式。

    Args:
        trajs: [B, T, 2]  轨迹点序列 (x=forward, y=left)

    Returns:
        κ: [B, T-2]  中间 T-2 个点的曲率 (1/m)
    """
    B, T, _ = trajs.shape

    # 三边长度: a = d(p_i-1, p_i), b = d(p_i, p_i+1), c = d(p_i-1, p_i+1)
    dp1 = trajs[:, 1:] - trajs[:, :-1]           # [B, T-1, 2]
    a = dp1[:, :-1].norm(dim=-1)                  # [B, T-2]
    b = dp1[:, 1:].norm(dim=-1)                   # [B, T-2]
    c = (trajs[:, 2:] - trajs[:, :-2]).norm(dim=-1)  # [B, T-2]

    # Heron 公式: s = (a+b+c)/2, area² = s(s-a)(s-b)(s-c)
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)
    area_sq = torch.clamp(area_sq, min=0.0)  # 数值安全
    area = torch.sqrt(area_sq + 1e-12)        # [B, T-2] 避免 sqrt(0) 梯度 inf

    # κ = 4 * area / (a * b * c)
    denominator = a * b * c
    kappa = 4.0 * area / (denominator + 1e-8)

    return kappa


def trajectory_compliance_loss(
    ego_fut_preds: torch.Tensor,
    ego_fut_cmd: torch.Tensor,
    kappa_max: torch.Tensor,
    omega_max: torch.Tensor,
    dt: float = 0.5,
) -> tuple:
    """
    Hinge loss 约束规划轨迹在物理安全边界内。

    L_kappa = mean(max(0, κ_i - κ_max))       空间违规
    L_omega = mean(max(0, |Δκ/dt| - ω_max))   时间违规

    Args:
        ego_fut_preds: [B, ego_fut_mode=3, fut_ts=6, 2]  规划轨迹
        ego_fut_cmd:   [B, ego_fut_mode=3]  GT 命令 one-hot
        kappa_max:     [B, 1]  物理头预测的最大安全曲率 (1/m)
        omega_max:     [B, 1]  物理头预测的最大方向盘转速 (rad/s)
        dt:            时间步长 (默认 0.5s)

    Returns:
        (loss_kappa, loss_omega): 两个标量 loss
    """
    B, M, T = ego_fut_preds.shape[:3]

    # 选择 GT 命令对应的轨迹模式 [B, T, 2]
    cmd_idx = ego_fut_cmd.argmax(dim=-1)   # [B]
    batch_idx = torch.arange(B, device=ego_fut_preds.device)
    traj = ego_fut_preds[batch_idx, cmd_idx]  # [B, T, 2]

    # 计算曲率 [B, T-2] (中间 4 个点)
    kappa = compute_curvature(traj)         # [B, 4]

    # 曲率违规: max(0, κ_i - κ_max)
    kappa_t = kappa_max.view(B, 1)          # [B, 1]
    # inf κ_max → 不产生合规 loss
    finite_mask = torch.isfinite(kappa_t)    # [B, 1]
    violations_kappa = torch.relu(kappa - kappa_t)  # [B, 4]
    violations_kappa = violations_kappa * finite_mask.float()
    loss_kappa = violations_kappa.mean()

    # 曲率变化率违规: max(0, |Δκ|/dt - ω_max)
    dkappa_dt = torch.abs(kappa[:, 1:] - kappa[:, :-1]) / dt   # [B, 3]
    omega_t = omega_max.view(B, 1)                               # [B, 1]
    violations_omega = torch.relu(dkappa_dt - omega_t)
    loss_omega = violations_omega.mean()

    return loss_kappa, loss_omega
