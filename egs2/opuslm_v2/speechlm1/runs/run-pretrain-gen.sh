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
# test_sets+="dialogue:eval-test_clean_audioset-v2-music_add_mix-tgt2audio "
# test_sets+="dialogue:eval-test_clean_audioset-v2-music_remove_mix-tgt2audio "
# test_sets+="dialogue:eval-test_clean_audioset-v2-music_replace_mix-tgt2audio "
# test_sets+="dialogue:eval-test_clean_audioset-v2-sound_add_mix-tgt2audio "
# test_sets+="dialogue:eval-test_clean_audioset-v2-sound_remove_mix-tgt2audio "
# test_sets+="dialogue:eval-test_clean_audioset-v2-sound_replace_mix-tgt2audio "
# test_sets+="dialogue:eval-test_clean_audioset-v2-speech_add_mix-tgt2audio "
# test_sets+="dialogue:eval-test_clean_audioset-v2-speech_remove_mix-tgt2audio "
# test_sets+="dialogue:eval-test_clean_audioset-v2-speech_replace_mix-tgt2audio "
# test_sets+="dialogue:eval-test_clean-v1-audio_effect_dereverb-tgt2audio "
# test_sets+="dialogue:eval-test_clean-v1-audio_effect_pitch-tgt2audio "
# test_sets+="dialogue:eval-test_clean-v1-audio_effect_reverb-tgt2audio "
# test_sets+="dialogue:eval-test_clean-v1-audio_effect_speed-tgt2audio "
# test_sets+="dialogue:eval-test_clean-v1-audio_effect_volume-tgt2audio "
# test_sets+="dialogue:eval-test_clean-v1-style_emotion-tgt2audio "
# test_sets+="dialogue:eval-test_clean-v1-style_whisper-tgt2audio "
# test_sets+="dialogue:eval-test_clean-v1-transcription_add_paralinguistic-tgt2audio "
# test_sets+="dialogue:eval-test_clean-v1-transcription_del-tgt2audio "
# test_sets+="dialogue:eval-test_clean-v1-transcription_ins-tgt2audio "
# test_sets+="dialogue:eval-test_clean-v1-transcription_replace_sentence-tgt2audio "
# test_sets+="dialogue:eval-test_clean-v1-transcription_sub-tgt2audio "
# test_sets+="dialogue:eval-test_clean_audioset-v3-music_creative_edit-tgt2audio "
# test_sets+="dialogue:eval-test_clean_audioset-v3-sing_creative_edit-tgt2audio "
# test_sets+="dialogue:eval-test_clean_audioset-v3-sound_creative_edit-tgt2audio "
# test_sets+="dialogue:eval-test_clean_audioset-v3-speech_creative_edit-tgt2audio "

test_sets+="dialogue:eval-test_clean-v1-transcription_del-t2a_t2a "
test_sets+="dialogue:eval-test_clean-v1-transcription_ins-t2a_t2a "
test_sets+="dialogue:eval-test_clean-v1-transcription_sub-t2a_t2a "
test_sets+="dialogue:eval-test_clean-v1-transcription_replace_sentence-t2a_t2a "
# test_sets+="dialogue:eval-test_clean-v1-transcription_add_paralinguistic-t2a_t2a "

test_sets+="dialogue:eval-test_clean_audioset-v2-music_add_mix-t2a_t2a "
test_sets+="dialogue:eval-test_clean_audioset-v2-music_remove_mix-t2a_t2a "
# test_sets+="dialogue:eval-test_clean_audioset-v2-music_replace_mix-t2a_t2a "
test_sets+="dialogue:eval-test_clean_audioset-v2-sound_add_mix-t2a_t2a "
test_sets+="dialogue:eval-test_clean_audioset-v2-sound_remove_mix-t2a_t2a "
# test_sets+="dialogue:eval-test_clean_audioset-v2-sound_replace_mix-t2a_t2a "
test_sets+="dialogue:eval-test_clean_audioset-v2-speech_add_mix-t2a_t2a "
test_sets+="dialogue:eval-test_clean_audioset-v2-speech_remove_mix-t2a_t2a "
# test_sets+="dialogue:eval-test_clean_audioset-v2-speech_replace_mix-t2a_t2a "

test_sets+="dialogue:eval-test_clean_audioset-v3-music_creative_edit-t2a_t2a "
test_sets+="dialogue:eval-test_clean_audioset-v3-sing_creative_edit-t2a_t2a "
test_sets+="dialogue:eval-test_clean_audioset-v3-sound_creative_edit-t2a_t2a "
test_sets+="dialogue:eval-test_clean_audioset-v3-speech_creative_edit-t2a_t2a "


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
