#!/bin/bash

# LTX-2.3 distilled two-stage using the x1.5 spatial upscaler:
#   stage 1 (8 steps, 640x960)  -> euler
#   stage 2 (3 steps, 960x1440) -> euler
#
# The x1.5 ratio needs stage-1 sizes that are multiples of 64, so reachable
# finals are multiples of 96. The runner logs the size it actually produces if
# target_height/target_width are not reachable.

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
--config_json ${lightx2v_path}/configs/ltx2/ltx2_3_upsample_x15.json \
--prompt "A beautiful sunset over the ocean" \
--negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, extra limbs, disfigured hands, inconsistent perspective, camera shake, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, distorted voice, robotic voice, echo, background noise, off-sync audio, jittery movement, unnatural transitions, AI artifacts." \
--save_result_path ${lightx2v_path}/save_results/output_ltx2_3_upsample_x15.mp4
