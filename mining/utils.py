"""
mining/utils.py — 物理约束计算核心函数

坐标系约定（与 VAD 数据处理一致）:
  - LiDAR 帧: x=forward, y=left, z=up
  - 全局帧: nuScenes 标准全局坐标 (x=east, y=north)
  - 本模块函数接受任意 2D 坐标系，只要 ego 和 agent 在同一帧即可

作者: SafeDrive-LLM
"""

import numpy as np
from typing import List, Optional, Tuple


# ==================== 物理常数 ====================
FRICTION_COEFF = 0.7          # μ: 干燥沥青路面摩擦系数
GRAVITY = 9.8                 # g (m/s²)
A_BRAKE_MAX = -5.0            # 最大安全减速度 (m/s²), 常规路面
A_BRAKE_MAX_RAIN = -3.5       # 雨天最大减速度 (待扩展)
TTC_AEB_THRESHOLD = 0.5       # AEB 触发 TTC 阈值 (s)
TTC_WARNING_THRESHOLD = 2.0   # 危险 TTC 警告阈值 (s)
MIN_SPEED_FOR_KAPPA = 0.1     # 低于此速度 κ_max 返回 inf (无翻车风险)
EVASIVE_LANE_WIDTH = 3.0      # 横向避让所需的最小车道宽度 (m)
OMEGA_MAX_MIN = 0.1           # ω_max 下限 (rad/s)
OMEGA_MAX_MAX = 1.0           # ω_max 上限 (rad/s)
OMEGA_COEFF = 5.0             # ω_max 公式系数: ω = min(1.0, max(0.1, 5.0/v))


# ==================== TTC 计算 ====================

def calc_ttc(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    agent_pos: np.ndarray,
    agent_vel: np.ndarray,
    ego_heading: Optional[float] = None,
    grace_period: float = 0.3,
) -> float:
    """
    计算自车与单个周围物体的 TTC (Time-To-Collision)。

    公式:
      TTC = |rel_pos| / max(v_rel_proj, epsilon)

    其中 v_rel_proj 是相对速度在相对位置方向上的投影。
    若 agent 正在远离 (>0)，TTC 为 inf。

    Args:
        ego_pos:   自车 2D 位置 [x, y]
        ego_vel:   自车 2D 速度 [vx, vy]
        agent_pos: agent 2D 位置 [x, y]
        agent_vel: agent 2D 速度 [vx, vy]
        ego_heading: 自车朝向角 (rad), 备用进入ego-frame模式
        grace_period: 安全缓冲时间 (s), 防止数值不稳定

    Returns:
        float: TTC (秒), 无碰撞风险返回 inf, 已碰撞返回 0
    """
    rel_pos = agent_pos[:2] - ego_pos[:2]
    rel_dist = float(np.linalg.norm(rel_pos))

    # 已重叠（碰撞已发生）
    if rel_dist < 0.01:
        return 0.0

    rel_vel = agent_vel[:2] - ego_vel[:2]
    # 相对速度在视线方向上的投影
    # 正值: agent 远离 ego (安全), 负值: agent 靠近 ego (危险)
    los_dir = rel_pos / max(rel_dist, 1e-6)  # line-of-sight unit vector
    v_rel_proj = float(np.dot(rel_vel, los_dir))

    # Agent 正在远离，无碰撞风险
    if v_rel_proj >= 0:
        return float('inf')

    # TTC = 距离 / 接近速度 + 缓冲时间
    ttc = rel_dist / (-v_rel_proj) + grace_period

    return max(ttc, 0.0)


def calc_ttc_forward_projection(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    agent_pos: np.ndarray,
    agent_vel: np.ndarray,
    ego_heading: float,
) -> float:
    """
    计算 TTC 的前向投影变体（仅沿自车前进方向）。

    适用于: 直道行驶，主要关注前方碰撞风险。
    使用场景: 论文中的 "仅前方物体 TTC < 0.5s" 判断。

    Args:
        ego_pos:     自车 2D 位置 [x, y]
        ego_vel:     自车 2D 速度 [vx, vy]
        agent_pos:   agent 2D 位置 [x, y]
        agent_vel:   agent 2D 速度 [vx, vy]
        ego_heading: 自车朝向角 (rad)

    Returns:
        float: TTC (秒), inf 表示无碰撞风险
    """
    forward_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)])

    rel_pos = agent_pos[:2] - ego_pos[:2]
    # 仅考虑前方的 agent (前向投影 > 0)
    forward_dist = float(np.dot(rel_pos, forward_dir))
    if forward_dist <= 0:
        return float('inf')

    ego_speed = float(np.dot(ego_vel[:2], forward_dir))
    agent_speed = float(np.dot(agent_vel[:2], forward_dir))
    rel_speed = ego_speed - agent_speed

    if rel_speed <= 0:
        return float('inf')

    return forward_dist / rel_speed


def calc_all_ttc(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    agent_positions: np.ndarray,
    agent_velocities: np.ndarray,
    ego_heading: Optional[float] = None,
) -> np.ndarray:
    """
    批量计算自车与所有周围物体的 TTC。

    Args:
        ego_pos:      自车 2D 位置 [x, y]
        ego_vel:      自车 2D 速度 [vx, vy]
        agent_positions: (N, 2) agent 位置数组
        agent_velocities: (N, 2) agent 速度数组
        ego_heading:  自车朝向角 (可选)

    Returns:
        np.ndarray: (N,) 每个 agent 的 TTC 值
    """
    n = len(agent_positions)
    ttc_values = np.full(n, float('inf'))

    for i in range(n):
        ttc_values[i] = calc_ttc(
            ego_pos, ego_vel,
            agent_positions[i], agent_velocities[i],
            ego_heading=ego_heading,
        )

    return ttc_values


# ==================== 安全边界计算 ====================

def calc_kappa_max(v: float, mu: float = FRICTION_COEFF, g: float = GRAVITY) -> float:
    """
    最大安全曲率: κ_max = μ·g / v²

    物理意义: 以速度 v 转弯时，向心力 = m·v²·κ，由摩擦力 μ·m·g 提供。
    当 v²·κ > μ·g 时车辆会侧滑/翻车。

    Args:
        v:  自车速度 (m/s), 标量
        mu: 路面摩擦系数 (默认 0.7 对应干燥沥青)
        g:  重力加速度 (默认 9.8 m/s²)

    Returns:
        float: 最大安全曲率 (1/m), v 较小时返回 inf (无风险)
    """
    if abs(v) < MIN_SPEED_FOR_KAPPA:
        return float('inf')
    return mu * g / (v * v)


def calc_a_brake_max(surface: str = 'dry') -> float:
    """
    最大安全减速度。

    默认 -5.0 m/s² (约 0.51g), 对应舒适刹停。
    实际车辆最大制动约 -9.0 m/s² (紧急), 但考虑舒适性取 -5.0。

    Args:
        surface: 'dry' | 'rain' (雨天暂未细化实现)

    Returns:
        float: 最大安全减速度 (负值, m/s²)
    """
    mapping = {
        'dry': A_BRAKE_MAX,
        'rain': A_BRAKE_MAX_RAIN,
    }
    return mapping.get(surface, A_BRAKE_MAX)


# ==================== AEB 紧急制动判断 ====================

def calc_aeb_prob(
    ttc_values: np.ndarray,
    has_evasive_space: bool,
    threshold: float = TTC_AEB_THRESHOLD,
) -> float:
    """
    紧急制动概率 P_AEB。

    规则: 存在任何物体 TTC < threshold 且无横向避让空间 → 1.0，否则 → 0.0

    Args:
        ttc_values:       (N,) 每个 agent 的 TTC (秒)
        has_evasive_space: 是否有可用的横向避让空间
        threshold:        TTC 触发阈值 (秒)

    Returns:
        float: 0.0 或 1.0
    """
    # 将所有 agent 的 TTC 与阈值比较
    any_critical = np.any(ttc_values < threshold)

    if any_critical and not has_evasive_space:
        return 1.0
    return 0.0


def check_evasive_space(
    ego_pos: np.ndarray,
    ego_heading: float,
    lane_boundaries: List[np.ndarray],
    min_width: float = EVASIVE_LANE_WIDTH,
) -> bool:
    """
    检查当前车道是否有足够的横向避让空间。

    简化实现: 计算自车左右 2m 范围内的可用宽度。
    TODO: 后续可改用 nuScenes map API 精确检查相邻车道。

    Args:
        ego_pos:          自车 2D 位置 [x, y]
        ego_heading:      自车朝向角 (rad)
        lane_boundaries:  车道边界线列表，每条为 (M, 2) 点集
        min_width:        最小避让宽度 (m)

    Returns:
        bool: True 表示有足够避让空间
    """
    # 横向方向向量 (垂直于前进方向)
    lat_dir = np.array([-np.sin(ego_heading), np.cos(ego_heading)])

    # 在左右各 min_width/2 范围内是否通畅
    # 简化版本: 检查左右两侧是否有车道边界
    min_dist_left = float('inf')
    min_dist_right = float('inf')

    for boundary in lane_boundaries:
        if len(boundary) == 0:
            continue
        # 计算边界上所有点到自车的距离
        vecs = boundary - ego_pos[:2]
        for vec in vecs:
            lat_dist = float(np.dot(vec, lat_dir))
            if lat_dist > 0:
                min_dist_left = min(min_dist_left, lat_dist)
            else:
                min_dist_right = min(min_dist_right, -lat_dist)

    # 左右两侧都有至少 min_width/2 的空间
    return (
        min_dist_left >= min_width / 2
        and min_dist_right >= min_width / 2
    )


# ==================== Agent 方位统计 ====================

def count_agents_by_quadrant(
    ego_pos: np.ndarray,
    ego_heading: float,
    agent_positions: np.ndarray,
) -> dict:
    """
    按前/左/右/后四个象限统计周围 agent 数量。

    象限定义（以自车朝向为 0°）:
      - front:  -45° ~ 45°
      - right:  45° ~ 135°
      - rear:   135° ~ 225° (或 -135° ~ -225°)
      - left:   225° ~ 315° (或 -45° ~ -135°)

    Args:
        ego_pos:       自车 2D 位置 [x, y]
        ego_heading:   自车朝向角 (rad)
        agent_positions: (N, 2) agent 位置

    Returns:
        dict: {
            'num_front': int, 'num_left': int, 'num_right': int, 'num_rear': int,
            'nearest_dist': float,            # 最近物体距离 (m)
            'nearest_rel_vel': float,         # 最近物体的相对速度大小 (m/s)
        }
    """
    if len(agent_positions) == 0:
        return {
            'num_front': 0, 'num_left': 0, 'num_right': 0, 'num_rear': 0,
            'nearest_dist': float('inf'),
            'nearest_rel_vel': 0.0,
        }

    forward_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)])
    lat_dir = np.array([-np.sin(ego_heading), np.cos(ego_heading)])

    rel_vecs = agent_positions[:, :2] - ego_pos[:2]

    # 投影到前向和横向
    fwd_proj = np.dot(rel_vecs, forward_dir)   # (N,)
    lat_proj = np.dot(rel_vecs, lat_dir)        # (N,)

    num_front = int(np.sum((fwd_proj > 0) & (abs(lat_proj) <= abs(fwd_proj))))
    num_left = int(np.sum((lat_proj > 0) & (abs(lat_proj) > abs(fwd_proj))))
    num_right = int(np.sum((lat_proj < 0) & (abs(lat_proj) > abs(fwd_proj))))
    num_rear = int(np.sum((fwd_proj < 0) & (abs(lat_proj) <= abs(fwd_proj))))

    # 最近物体距离
    dists = np.linalg.norm(rel_vecs, axis=-1)
    nearest_idx = int(np.argmin(dists)) if len(dists) > 0 else -1
    nearest_dist = float(dists[nearest_idx]) if nearest_idx >= 0 else float('inf')

    return {
        'num_front': num_front,
        'num_left': num_left,
        'num_right': num_right,
        'num_rear': num_rear,
        'nearest_dist': nearest_dist,
    }


# ==================== 道路结构 ====================

def classify_lane_type(nusc_map, ego_pose: dict) -> str:
    """
    使用 nuScenes map API 获取当前车道类型。

    Args:
        nusc_map:  NuScenesMap 实例
        ego_pose:  自车位姿字典 {'translation': [x,y,z], 'rotation': [w,x,y,z]}

    Returns:
        str: 'driving' | 'turning' | 'intersection' | 'bike' | 'unknown'
    """
    try:
        from nuscenes.map_expansion.map_api import NuScenesMap

        x = ego_pose['translation'][0]
        y = ego_pose['translation'][1]
        point = (x, y)

        # 尝试获取位置所在的车道 token
        # get_closest_lane 返回 (lane_token, distance,)
        # 注意: 不同版本 nuScenes devkit API 可能有细微差异
        try:
            # nuscenes-devkit >= 1.1.0
            lane_token = nusc_map.get_closest_lane(x, y, radius=5.0)
        except (AttributeError, TypeError):
            # 降级方案: 用 layername 查询
            records_in_radius = nusc_map.get_records_in_radius(
                x, y, 5.0, ['lane']
            )
            if records_in_radius and 'lane' in records_in_radius:
                # 取最近的车道
                lanes = records_in_radius['lane']
                if lanes:
                    lane_token = lanes[0]
                else:
                    return 'unknown'
            else:
                return 'unknown'

        if lane_token == '' or lane_token is None:
            return 'unknown'

        lane = nusc_map.get('lane', lane_token)
        # lane 字段中 lane_type 可能存储为 'lane_type' 或 'lane_type0'
        lane_type = lane.get('lane_type', lane.get('lane_type0', 'unknown'))
        # 标准化
        if 'turn' in lane_type.lower():
            return 'turning'
        elif 'inter' in lane_type.lower():
            return 'intersection'
        elif 'bike' in lane_type.lower() or 'cycle' in lane_type.lower():
            return 'bike'
        elif 'driving' in lane_type.lower() or 'car' in lane_type.lower():
            return 'driving'
        else:
            return lane_type.lower()

    except Exception:
        return 'unknown'


# ==================== 可见性检查 ====================

def check_visibility(
    ego_pos: np.ndarray,
    ego_heading: float,
    agent_positions: np.ndarray,
    agent_boxes: List[dict],
    num_sectors: int = 4,
    occlusion_threshold_angle: float = 30.0,
) -> dict:
    """
    检查各象限是否存在视线遮挡。

    简化实现: 基于各象限内最近 agent 的距离判断是否遮挡。
    如果有 agent 距离 < 2m -> 被遮挡 -> 可能看不到更远的物体。

    Args:
        ego_pos:          自车 2D 位置 [x, y]
        ego_heading:      自车朝向角 (rad)
        agent_positions:  (N, 2) agent 位置
        agent_boxes:      agent 的 w/l 信息列表 [{'size': [w,l,h]}, ...]
        num_sectors:      将 360° 分为几个扇区 (默认 4: 前/右/后/左)
        occlusion_threshold_angle: 遮挡判定角度阈值 (度)

    Returns:
        dict: {
            'front_occluded': bool,
            'left_occluded': bool,
            'right_occluded': bool,
            'rear_occluded': bool,
        }
    """
    sectors = ['front', 'right', 'rear', 'left']
    result = {f'{s}_occluded': False for s in sectors}

    if len(agent_positions) == 0:
        return result

    for i in range(len(agent_positions)):
        rel_vec = agent_positions[i][:2] - ego_pos[:2]
        dist = float(np.linalg.norm(rel_vec))
        if dist < 1e-3:
            continue

        # 计算 agent 在 ego 坐标系中的方位角
        angle = np.arctan2(rel_vec[1], rel_vec[0]) - ego_heading
        angle = np.degrees(angle) % 360

        # 判断所在扇区
        if 45 <= angle < 135:
            sector = 'right'
        elif 135 <= angle < 225:
            sector = 'rear'
        elif 225 <= angle < 315:
            sector = 'left'
        else:
            sector = 'front'

        # 如果有大型 agent (< 5m) 在旁，认为可能遮挡
        if agent_boxes and i < len(agent_boxes):
            w = agent_boxes[i].get('size', [0, 0, 0])[0]
            l = agent_boxes[i].get('size', [0, 0, 0])[1]
            # 大型物体 + 距离较近 -> 可能遮挡
            if max(w, l) > 3.0 and dist < 3.0:
                result[f'{sector}_occluded'] = True

    return result


# ==================== 方向盘转速 ====================

def calc_steering_rate(
    steering_angle: Optional[float],
    prev_steering_angle: Optional[float],
    dt: float = 0.5,
) -> float:
    """
    方向盘转角变化率 (rad/s)。

    Args:
        steering_angle:      当前方向盘转角 (rad)
        prev_steering_angle: 上一帧方向盘转角 (rad)
        dt:                  时间间隔 (s)

    Returns:
        float: 方向盘转速 (rad/s), 数据缺失返回 0.0
    """
    if steering_angle is None or prev_steering_angle is None:
        return 0.0
    return (steering_angle - prev_steering_angle) / max(dt, 1e-3)


# ==================== 最大方向盘转速 ====================

def calc_omega_max(
    v: float,
    coeff: float = OMEGA_COEFF,
    min_val: float = OMEGA_MAX_MIN,
    max_val: float = OMEGA_MAX_MAX,
) -> float:
    """
    最大安全方向盘转速: ω_max = max(0.1, min(1.0, coeff / |v|))

    物理意义: 高速时方向盘微小转动即可造成大幅横向位移，
    因此允许的方向盘转速随速度增加而降低。
    v 低于 1 m/s 时 clamp 到 max_val。

    Args:
        v:     自车速度 (m/s)
        coeff: 公式系数 (默认 5.0)
        min_val: 输出下限 (默认 0.1 rad/s)
        max_val: 输出上限 (默认 1.0 rad/s)

    Returns:
        float: 最大安全方向盘转速 (rad/s), 始终在 [min_val, max_val]
    """
    abs_v = max(abs(v), 0.01)
    omega = coeff / abs_v
    return max(min_val, min(max_val, omega))


# ==================== 综合: 单帧物理标签生成 ====================

def compute_physical_labels(
    ego_speed: float,
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    agent_positions: np.ndarray,
    agent_velocities: np.ndarray,
    ego_heading: float,
    has_evasive_space: bool = True,
) -> dict:
    """
    综合计算一帧的所有物理约束标签。

    这是 build_meta_db 的核心调用函数。

    Args:
        ego_speed:        自车标量速度 (m/s)
        ego_pos:          自车 2D 位置 [x, y]
        ego_vel:          自车 2D 速度 [vx, vy]
        agent_positions:  (N, 2) agent 位置
        agent_velocities: (N, 2) agent 速度
        ego_heading:      自车朝向角 (rad)
        has_evasive_space: 是否有横向避让空间

    Returns:
        dict: {
            'agent_ttc':   (N,) 每个 agent 的 TTC 值
            'kappa_max':   float 最大安全曲率 (1/m)
            'omega_max':   float 最大方向盘转速 (rad/s)
            'p_aeb':       float 紧急制动概率 (0.0 / 1.0)
        }
    """
    # 1. TTC 计算
    agent_ttc = calc_all_ttc(
        ego_pos, ego_vel,
        agent_positions, agent_velocities,
        ego_heading=ego_heading,
    )

    # 2. 最大安全曲率
    kappa_max = calc_kappa_max(ego_speed)

    # 3. 最大方向盘转速
    omega_max = calc_omega_max(ego_speed)

    # 4. 紧急制动概率
    p_aeb = calc_aeb_prob(agent_ttc, has_evasive_space)

    return {
        'agent_ttc': agent_ttc.tolist(),
        'kappa_max': kappa_max,
        'omega_max': omega_max,
        'p_aeb': p_aeb,
    }
