#!/bin/bash

# LTX-2.3 distilled two-stage with ComfyUI-style ancestral sampling:
#   stage 1 (8 steps, 512x768)   -> euler_ancestral
#   stage 2 (3 steps, 1024x1536) -> euler_ancestral_cfg_pp
#
# Note cfg_pp always runs an extra unconditional forward per step, so stage 2
# costs roughly 2x the forwards of a plain-euler stage 2. Set a real
# --negative_prompt for it to be meaningful.

# set path and first
lightx2v_path=/path/to/LightX2V
model_path=Lightricks/LTX-2.3

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls ltx2 \
--task t2av \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/ltx2/ltx2_3_upsample_ancestral.json \
--prompt "A beautiful sunset over the ocean" \
--negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, extra limbs, disfigured hands, inconsistent perspective, camera shake, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, distorted voice, robotic voice, echo, background noise, off-sync audio, jittery movement, unnatural transitions, AI artifacts." \
--save_result_path ${lightx2v_path}/save_results/output_ltx2_3_upsample_ancestral.mp4
