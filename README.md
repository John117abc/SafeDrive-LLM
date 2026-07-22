# SafeDrive-LLM

基于结构化常识注入的端到端安全驾驶模型。

## 项目结构

```
SafeDrive-LLM/
├── configs/          # 配置文件（基础配置、VAD2配置、数据挖掘、LLM）
├── data/             # 数据相关（原始数据软链接、处理后数据、CARLA仿真）
├── vad2/             # VAD2 基线模型（mmdet3d_plugin + 训练评测脚本）
├── mining/           # 三阶段数据挖掘管线
├── llm/              # LLM 常识标注
├── model/            # 模型改造（硬约束头 + 常识蒸馏头）
├── train/            # 联合训练
├── eval/             # 评测（开环/闭环/消融）
├── notebooks/        # Jupyter 探索性分析
├── scripts/          # 辅助脚本
└── docs/             # 文档
```

## 快速开始

### 环境配置

```bash
conda activate e2esafedrive
bash scripts/setup_env.sh
```

### 数据准备

```bash
# nuScenes 数据已软链接到 data/raw/nuScenes
# 运行数据挖掘管线
python mining/build_meta_db.py
python mining/filter_hazards.py
python mining/active_selection.py
```

### 训练

```bash
# VAD2 基线训练
bash vad2/tools/dist_train.sh vad2/configs/VAD/VAD_tiny_stage_1.py 2

# 联合训练（硬约束 + 常识蒸馏）
python train/train_joint.py
```

### 评测

```bash
# 开环评测
python eval/openloop/run_openloop.py

# 闭环评测 (CARLA)
python eval/closedloop/carla_runner.py
```

## 核心思想

将端到端自动驾驶在长尾场景下的安全泛化问题，解耦为两个独立子问题：
- **物理安全边界**：数学约束，不可用网络近似
- **驾驶常识认知**：借助 LLM 先验知识，建立在结构化感知之上

详见 `docs/design.md`。
