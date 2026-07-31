"""
model/vad2_with_heads.py — 物理约束头网络模块

PhysicalHead: 从 ego_feats [B, 1, 512] 预测 3 维物理风险信号。
输入 ego_feats 是 VADHead.forward() 阶段 5 中融合了
ego ↔ agent ↔ map 交叉注意力的特征张量。

架构:
  ego_feats [B, 1, 512]
       │
       ├── Linear(512, 256) + ReLU + Dropout(0.1)
       ├── Linear(256, 128) + ReLU (共享 trunk)
       │
       ├──→ kappa_branch:  Linear(128, 1)   → κ_max    [B, 1]
       ├──→ omega_branch:  Linear(128, 1)   → ω_max    [B, 1]  (ReLU)
       └──→ aeb_branch:    Linear(128, 1)   → P_AEB    [B, 1]  (sigmoid)
"""

import torch
import torch.nn as nn


class PhysicalHead(nn.Module):
    """
    物理约束头 — 从融合后的 ego 特征预测安全风险信号。
    
    3 维输出:
        κ_max  — 最大安全曲率 (1/m), 范围 0.02-0.7
        ω_max  — 最大方向盘转速 (rad/s), 范围 0.1-1.0
        P_AEB  — 紧急制动概率, 范围 0.0/1.0

    Args:
        in_dim:      输入特征维度 (默认 512, 对应 VAD embed_dims*2)
        hidden_dims: 隐藏层维度列表 (默认 [256, 128])
        dropout:     Dropout 比率
    """
    def __init__(
        self,
        in_dim: int = 512,
        hidden_dims: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # 共享特征提取 trunk
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.shared_trunk = nn.Sequential(*layers)

        # 三个独立的预测分支
        last_dim = hidden_dims[-1]
        self.kappa_head = nn.Linear(last_dim, 1)   # κ_max: 回归
        self.omega_head = nn.Linear(last_dim, 1)   # ω_max: 回归 (ReLU 保证非负)
        self.aeb_head = nn.Linear(last_dim, 1)     # P_AEB: sigmoid 后输出

    def forward(self, ego_feats: torch.Tensor) -> dict:
        """
        Args:
            ego_feats: [B, 1, 512] VADHead 中 ego-agent-map 融合特征

        Returns:
            dict: {
                'kappa_max': [B, 1],   未激活 (回归值)
                'omega_max': [B, 1],   ReLU 后非负 (回归值)
                'p_aeb':     [B, 1],   已过 sigmoid
            }
        """
        # ego_feats: [B, 1, D] → [B, D]
        x = ego_feats.squeeze(1)

        # 共享 trunk 提取
        x = self.shared_trunk(x)

        return {
            'kappa_max': self.kappa_head(x),
            'omega_max': torch.relu(self.omega_head(x)),         # 非负
            'p_aeb':     torch.sigmoid(self.aeb_head(x)),
        }
