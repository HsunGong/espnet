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
test_sets+="dialogue:part2_4_debug-novad-music.min_0.max_10-dialogue.step0.main2main "
test_sets+="dialogue:part2_4_debug-novad-sound.min_10-dialogue.step0.main2main "
test_sets+="dialogue:part2_4_debug-novad-speech.min_0.max_3-dialogue.step0.main2main "
test_sets+="dialogue:part2_4_debug-novad-music.min_0.max_10-dialogue.step4.cat2main.default "
test_sets+="dialogue:part2_4_debug-novad-sound.min_10-dialogue.step4.cat2main.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_8.max_20-dialogue.step0.main2main "
test_sets+="dialogue:part2_4_debug-novad-music.min_0.max_10-dialogue.step4.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-sound.min_10-dialogue.step4.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_0.max_3-dialogue.step4.cat2main.default "
test_sets+="dialogue:part2_4_debug-novad-music.min_0.max_10-dialogue.step4.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-sound.min_10-dialogue.step4.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_0.max_3-dialogue.step4.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_8.max_20-dialogue.step4.cat2main.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_0.max_3-dialogue.step4.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_8.max_20-dialogue.step4.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-music.min_0.max_10-dialogue.step6.main2main.default "
test_sets+="dialogue:part2_4_debug-novad-sound.min_10-dialogue.step6.main2main.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_5.max_8-dialogue.step0.main2main "
test_sets+="dialogue:part2_4_debug-novad-speech.min_8.max_20-dialogue.step4.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_0.max_3-dialogue.step6.main2main.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_5.max_8-dialogue.step4.cat2main.default "
test_sets+="dialogue:part2_4_debug-novad-music.min_0.max_10-dialogue.step8.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-sound.min_10-dialogue.step8.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_5.max_8-dialogue.step4.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-music.min_0.max_10-dialogue.step8.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-sound.min_10-dialogue.step8.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_0.max_3-dialogue.step8.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_5.max_8-dialogue.step4.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_0.max_3-dialogue.step8.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-speech.min_5.max_8-dialogue.step6.main2main.default "
test_sets+="dialogue:part2_4_debug-novad-sound.min_0.max_7-dialogue.step0.main2main "
test_sets+="dialogue:part2_4_debug-novad-sound.min_0.max_7-dialogue.step4.cat2main.default "
test_sets+="dialogue:part2_4_debug-novad-sound.min_0.max_7-dialogue.step4.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-sound.min_0.max_7-dialogue.step4.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-sound.min_0.max_7-dialogue.step6.main2main.default "
test_sets+="dialogue:part2_4_debug-novad-sound.min_0.max_7-dialogue.step8.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-novad-sound.min_0.max_7-dialogue.step8.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-vad-sound.min_7.max_10-dialogue.step1.main2main "
test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step1.main2main "
test_sets+="dialogue:part2_4_debug-vad-sound.min_10-dialogue.step1.main2main "
test_sets+="dialogue:part2_4_debug-vad-sound.min_7.max_10-dialogue.step2.cat2main "
test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step2.cat2main "
test_sets+="dialogue:part2_4_debug-vad-sound.min_7.max_10-dialogue.step2.t2a_t2a "
test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step2.t2a_t2a "
test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step1.main2main "
test_sets+="dialogue:part2_4_debug-vad-sound.min_7.max_10-dialogue.step2.a2t_t2a "
test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step2.a2t_t2a "
test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step2.cat2main "
test_sets+="dialogue:part2_4_debug-vad-sound.min_7.max_10-dialogue.step4.cat2main.default "
test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step4.cat2main.default "
test_sets+="dialogue:part2_4_debug-vad-speech.min_25-dialogue.step1.main2main "
test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step2.t2a_t2a "
test_sets+="dialogue:part2_4_debug-vad-sound.min_7.max_10-dialogue.step4.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step4.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step2.a2t_t2a "
test_sets+="dialogue:part2_4_debug-vad-sound.min_7.max_10-dialogue.step4.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step4.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-vad-speech.min_25-dialogue.step2.cat2main "
test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step4.cat2main.default "
test_sets+="dialogue:part2_4_debug-vad-speech.min_25-dialogue.step2.t2a_t2a "
test_sets+="dialogue:part2_4_debug-vad-sound.min_7.max_10-dialogue.step5.main2main.default "
test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step5.main2main.default "
test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step4.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step4.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-vad-sound.min_7.max_10-dialogue.step6.main2main.default "
test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step6.main2main.default "
test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step5.main2main.default "
test_sets+="dialogue:part2_4_debug-vad-sound.min_7.max_10-dialogue.step8.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step8.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step6.main2main.default "
test_sets+="dialogue:part2_4_debug-vad-sound.min_7.max_10-dialogue.step8.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step8.a2t_t2a.default "
test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step8.t2a_t2a.default "
test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step8.a2t_t2a.default "


bash launch_opuslm_stage3_sft.sh \
    --train_registered_specifier "$train_sets" \
    --valid_registered_specifier "$valid_sets" \
    --test_registered_specifier "$test_sets" \
    --train_config conf/stage2_qwen3.yaml --inference_nj 8 --inference_workers 2 \
    --resume_path /mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_350000 \
    --inference_config conf/inference_audio.yaml \
    --exp_dir exp/opuslm_v2_stage2_pretrain_base \
    --stage 3 --stop_stage 3 \
    "$@"


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
