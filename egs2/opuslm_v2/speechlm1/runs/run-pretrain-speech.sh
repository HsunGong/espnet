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
# test_sets+="dialogue:mmau-mini-speech-a2t "
# test_sets+="dialogue:mmau-mini-speech-t2a "
# test_sets+="dialogue:part2_4-debug-step2-concat "
# test_sets+="dialogue:part2_4-debug-step2-main "
# test_sets+="dialogue:part2_4-debug-step3-concat "
# test_sets+="dialogue:part2_4-debug-step3-main " # <- same as step2-main, no need to evaluate
# test_sets+="dialogue:part2_4-debug-step4-concat "
# test_sets+="dialogue:part2_4-debug-step4-main "
# test_sets+="dialogue:part2_4-debug-step5-concat "
# test_sets+="dialogue:part2_4-debug-step5-main "
# test_sets+="dialogue:debug "
# test_sets+="dialogue:part2_4-debug-step4-concat-only_text "
# test_sets+="dialogue:part2_4-debug-step4-main-only_text "
# test_sets+="dialogue:part2_4-debug-step5-concat-only_text "
# test_sets+="dialogue:part2_4-debug-step5-main-only_text "
# test_sets+="dialogue:part2_4-debug-step4-concat-only_text-v2 "
# test_sets+="dialogue:part2_4-debug-step4-main-only_text-v2 "
# test_sets+="dialogue:part2_4-debug-step5-concat-only_text-v2 "
# test_sets+="dialogue:part2_4-debug-step5-main-only_text-v2 "
test_sets+="dialogue:part2_4-debug-step4-main-default "
test_sets+="dialogue:part2_4-debug-step4-concat_split-default "
test_sets+="dialogue:part2_4-debug-step4-caption_main-default "
test_sets+="dialogue:part2_4-debug-step4-main-silence0.5 "
test_sets+="dialogue:part2_4-debug-step4-concat_split-silence0.5 "
test_sets+="dialogue:part2_4-debug-step4-caption_main-silence0.5 "
test_sets+="dialogue:part2_4-debug-step4-main-silence2.0 "
test_sets+="dialogue:part2_4-debug-step4-concat_split-silence2.0 "
test_sets+="dialogue:part2_4-debug-step4-caption_main-silence2.0 "

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
