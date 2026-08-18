#!/bin/bash
# 消融对照：无物理头二阶段训练（12 epoch，RGB 归一化）
cd /root/autodl-tmp/codes/SafeDrive-LLM
rm -rf work_dirs/stage2_nophys
find vad2 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

export PYTHONPATH="$(pwd)/vad2:$PYTHONPATH"
export PYTHONWARNINGS=ignore

/root/miniconda3/envs/mmdet3d/bin/python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=28700 --use_env \
    vad2/tools/train.py \
    vad2/configs/VAD/VAD_tiny_stage_2_nophys.py \
    --work-dir work_dirs/stage2_nophys \
    --no-validate \
    --launcher pytorch \
    --cfg-options \
        total_epochs=12 \
        runner.max_epochs=12 \
        data.samples_per_gpu=1 \
        data.workers_per_gpu=4 \
        optimizer.lr=2e-4 \
        checkpoint_config.max_keep_ckpts=12
