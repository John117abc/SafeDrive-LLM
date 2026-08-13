#!/bin/bash
set -e
cd /root/autodl-tmp/codes/SafeDrive-LLM
rm -rf work_dirs/stage2_full
find vad2 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

export PYTHONPATH="$(pwd)/vad2:$PYTHONPATH"

/root/miniconda3/envs/mmdet3d/bin/python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=28720 \
    --use_env \
    vad2/tools/train.py \
    vad2/configs/VAD/VAD_tiny_stage_2.py \
    --work-dir work_dirs/stage2_full \
    --no-validate \
    --launcher pytorch \
    --cfg-options \
        total_epochs=24 \
        runner.max_epochs=24 \
        data.samples_per_gpu=1 \
        optimizer.lr=2e-4 \
        checkpoint_config.max_keep_ckpts=10
