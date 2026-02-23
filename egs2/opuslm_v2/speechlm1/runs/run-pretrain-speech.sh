#!/bin/bash

train_sets=""
train_sets+="dialogue:audio_edit-prompts_v1-add_v1-input_caption-0k-30k "
train_sets+="dialogue:audio_edit-prompts_v1-remove_v1-input_caption-30k-60k "
train_sets+="dialogue:audio_edit-prompts_v1-effect_v1-input_caption-60k-90k "

valid_sets=""
valid_sets+="dialogue:audio_edit-prompts_v1-add_v1-input_caption-dev "
valid_sets+="dialogue:audio_edit-prompts_v1-remove_v1-input_caption-dev "
valid_sets+="dialogue:audio_edit-prompts_v1-effect_v1-input_caption-dev "

test_sets=""
test_sets+="dialogue:part2_4-debug-step6-concat_split-default "
test_sets+="dialogue:part2_4-debug-step6-main-default "

bash launch_opuslm_stage3_sft.sh \
    --train_registered_specifier "$train_sets" \
    --valid_registered_specifier "$valid_sets" \
    --test_registered_specifier "$test_sets" \
    --train_config conf/stage2_qwen3.yaml --inference_nj 8 --inference_workers 2 \
    --resume_path /mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_350000 \
    --inference_config conf/inference_speech.yaml \
    --exp_dir exp/opuslm_v2_stage2_pretrain_base \
    --stage 3 --stop_stage 3 \
    "$@"
