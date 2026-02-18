#!/bin/bash

train_sets=""
train_sets+="dialogue:audio_edit-prompts_v1-add_v1-no_caption-0k-30k "
train_sets+="dialogue:audio_edit-prompts_v1-remove_v1-no_caption-30k-60k "
train_sets+="dialogue:audio_edit-prompts_v1-effect_v1-no_caption-60k-90k "

valid_sets=""
valid_sets+="dialogue:audio_edit-prompts_v1-add_v1-no_caption-dev "
valid_sets+="dialogue:audio_edit-prompts_v1-remove_v1-no_caption-dev "
valid_sets+="dialogue:audio_edit-prompts_v1-effect_v1-no_caption-dev "

test_sets=""
test_sets+="dialogue:audio_edit-prompts_v1-add_v1-no_caption-test "
test_sets+="dialogue:audio_edit-prompts_v1-remove_v1-no_caption-test "
test_sets+="dialogue:audio_edit-prompts_v1-effect_v1-no_caption-test "

bash launch_opuslm_stage3_sft.sh \
    --train_registered_specifier "$train_sets" \
    --valid_registered_specifier "$valid_sets" \
    --test_registered_specifier "$test_sets" \
    --train_config conf/stage3_v1.yaml --inference_nj 16 \
    --resume_path /mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_350000 \
    --exp_dir exp/edit-prompt_v1-all_v1-30k-no_caption \
    --stop_stage 3 \
    "$@"
