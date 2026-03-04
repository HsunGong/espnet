#!/bin/bash

train_sets=""

# 50k -> 1 epoch -> 5k~15k is enough for evaluation ?
# music : sound : speech ~ 1 : 1 : 2~3
# COMPARE | general data｜ continual pattern : repeat pattern | sft data
# v1      ｜     2      ｜          1        ｜       2       ｜
# v2      |      2      |          1         |       <1?      |    0.1?(~100k)

# original step0/step1 -> follow pretrain recipe totally (w/o vad, just name changed to fit the naming convention)
# caption -> audio
# music=100k + sound=60k + speech~=200k -> 350k?
train_sets+="dialogue:sft-part2_4-novad-music.min_0.max_10-dialogue.step0.main2main " # 45,923
train_sets+="dialogue:sft-part2_4-vad-music.min_10-dialogue.step1.main2main:0.5 " # 100,000
train_sets+="dialogue:sft-part2_4-vad-sound.min_10-dialogue.step1.main2main:0.3 " # 95,467
train_sets+="dialogue:sft-part2_4-vad-sound.min_7.max_10-dialogue.step1.main2main:0.5 " # 59,236
train_sets+="dialogue:sft-part2_4-novad-speech.min_3.max_5-dialogue.step0.main2main:0.1 " # 100,000
train_sets+="dialogue:sft-part2_4-vad-speech.min_5.max_8-dialogue.step1.main2main:0.4 " # 100,000
train_sets+="dialogue:sft-part2_4-novad-speech.min_8.max_20-dialogue.step0.main2main:0.6 " # 100,000
train_sets+="dialogue:sft-part2_4-vad-speech.min_20.max_25-dialogue.step1.main2main " # 68,666
train_sets+="dialogue:sft-part2_4-vad-speech.min_25-dialogue.step1.main2main:0.5 " # 100,000

# step2 continual pattern #<- important?
# audio1 -> caption1 -> caption2 -> audio2 (audio1 -> continue -> audio2)
# music=100k + sound=150k + speech=250k -> 500k
train_sets+="dialogue:sft-part2_4-vad-music.min_10-dialogue.step2.cat2main " # 100,000
train_sets+="dialogue:sft-part2_4-vad-sound.min_10-dialogue.step2.cat2main " # 95,467
train_sets+="dialogue:sft-part2_4-vad-sound.min_7.max_10-dialogue.step2.cat2main " # 59,236
train_sets+="dialogue:sft-part2_4-vad-speech.min_5.max_8-dialogue.step2.cat2main " # 100,000
train_sets+="dialogue:sft-part2_4-vad-speech.min_20.max_25-dialogue.step2.cat2main " # 68,666
train_sets+="dialogue:sft-part2_4-vad-speech.min_25-dialogue.step2.cat2main " # 100,000

# step4 repeat pattern -> which is bad 
# audio -> caption -> caption -> audio (audio repeated to learn the `reserve` format)
# almost music=1.5k + sound=0.95k + speech=1.5k
train_sets+="dialogue:sft-part2_4-novad-music.min_10-dialogue.step4.cat2main.default:0.01 " # 157,906
train_sets+="dialogue:sft-part2_4-novad-sound.min_10-dialogue.step4.cat2main.default:0.01 " # 95,467
train_sets+="dialogue:sft-part2_4-novad-speech.min_5.max_8-dialogue.step4.cat2main.default:0.005 " # 181,581
train_sets+="dialogue:sft-part2_4-novad-speech.min_8.max_20-dialogue.step4.cat2main.default:0.005 " # 156,354

##### totally 50k+40k ~ 100k
# audio1 -> caption1 -> caption2 -> audio2 (audio1 -> edit-op -> audio2)
# remix data: music ~ 10k, sound ~ 10k, speech ~ 30k, sing ~ 3k
train_sets+="dialogue:train_clean_audioset-v2-music_add_mix-cat2main " # 3,000
train_sets+="dialogue:train_clean_audioset-v2-music_remove_mix-cat2main " # 2,998
train_sets+="dialogue:train_clean_audioset-v2-music_replace_mix-cat2main " # 2,979
train_sets+="dialogue:train_clean_audioset-v2-sing_add_mix-cat2main " # 1,000
train_sets+="dialogue:train_clean_audioset-v2-sing_remove_mix-cat2main " # 999
train_sets+="dialogue:train_clean_audioset-v2-sing_replace_mix-cat2main " # 996
train_sets+="dialogue:train_clean_audioset-v2-sound_add_mix-cat2main " # 3,000
train_sets+="dialogue:train_clean_audioset-v2-sound_remove_mix-cat2main " # 3,000
train_sets+="dialogue:train_clean_audioset-v2-sound_replace_mix-cat2main " # 2,987
train_sets+="dialogue:train_clean_audioset-v2-speech_add_mix-cat2main " # 10,000
train_sets+="dialogue:train_clean_audioset-v2-speech_remove_mix-cat2main " # 9,996
train_sets+="dialogue:train_clean_audioset-v2-speech_replace_mix-cat2main " # 9,979
# speech-edit data: ~ 40k
train_sets+="dialogue:part2-speech_edit-v2-transcription_del-cat2main " # 6,721
train_sets+="dialogue:part2-speech_edit-v2-transcription_ins-cat2main " # 9,949
train_sets+="dialogue:part2-speech_edit-v2-transcription_replace_sentence-cat2main " # 8,558
train_sets+="dialogue:part2-speech_edit-v2-transcription_sub-cat2main " # 7,740

valid_sets=""
# original performance check
valid_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step1.main2main "
valid_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step1.main2main "
valid_sets+="dialogue:part2_4_debug-novad-sound.min_10-dialogue.step0.main2main "
# step2 continual pattern check
valid_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step2.cat2main "
valid_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step2.cat2main "
valid_sets+="dialogue:part2_4_debug-novad-sound.min_10-dialogue.step2.cat2main "

test_sets=""
# # test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step2.cat2main "
# # test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step4.cat2main.default "
# # test_sets+="dialogue:part2_4_debug-vad-speech.min_20.max_25-dialogue.step8.cat2main.default "

# # test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step1.main2main "
# # test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step2.cat2main "
# # test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step4.cat2main.default "
# # test_sets+="dialogue:part2_4_debug-vad-music.min_10-dialogue.step8.cat2main.default "

# # test_sets+="dialogue:part2_4_debug-novad-sound.min_10-dialogue.step0.main2main "
# # test_sets+="dialogue:part2_4_debug-novad-sound.min_10-dialogue.step4.cat2main.default "
# # test_sets+="dialogue:part2_4_debug-novad-sound.min_10-dialogue.step8.cat2main.default "

# test_sets+="dialogue:eval-test_clean-v1-transcription_del-cat2split1 "
# test_sets+="dialogue:eval-test_clean-v1-transcription_ins-cat2split1 "
# # test_sets+="dialogue:eval-test_clean-v1-transcription_sub-cat2split1 "
# test_sets+="dialogue:eval-test_clean-v1-transcription_replace_sentence-cat2split1 "
# test_sets+="dialogue:eval-test_clean-v1-transcription_add_paralinguistic-cat2split1 "
# test_sets+="dialogue:eval-test_clean-v1-style_emotion-cat2split1 "
# test_sets+="dialogue:eval-test_clean-v1-style_whisper-cat2split1 "
# # test_sets+="dialogue:eval-test_clean-v1-audio_effect_dereverb-cat2split1 "
# # test_sets+="dialogue:eval-test_clean-v1-audio_effect_pitch-cat2split1 "
# # test_sets+="dialogue:eval-test_clean-v1-audio_effect_reverb-cat2split1 "
# # test_sets+="dialogue:eval-test_clean-v1-audio_effect_speed-cat2split1 "
# # test_sets+="dialogue:eval-test_clean-v1-audio_effect_volume-cat2split1 "

test_sets+="dialogue:eval-test_clean_audioset-v2-music_add_mix-cat2split1 "
test_sets+="dialogue:eval-test_clean_audioset-v2-music_remove_mix-cat2split1 "
# test_sets+="dialogue:eval-test_clean_audioset-v2-music_replace_mix-cat2split1 "
test_sets+="dialogue:eval-test_clean_audioset-v2-sound_add_mix-cat2split1 "
test_sets+="dialogue:eval-test_clean_audioset-v2-sound_remove_mix-cat2split1 "
# test_sets+="dialogue:eval-test_clean_audioset-v2-sound_replace_mix-cat2split1 "
test_sets+="dialogue:eval-test_clean_audioset-v2-speech_add_mix-cat2split1 "
test_sets+="dialogue:eval-test_clean_audioset-v2-speech_remove_mix-cat2split1 "
# test_sets+="dialogue:eval-test_clean_audioset-v2-speech_replace_mix-cat2split1 "

# test_sets+="dialogue:eval-test_clean_audioset-v3-music_creative_edit-cat2split1 "
# # test_sets+="dialogue:eval-test_clean_audioset-v3-sing_creative_edit-cat2split1 "
# # test_sets+="dialogue:eval-test_clean_audioset-v3-sound_creative_edit-cat2split1 "
# test_sets+="dialogue:eval-test_clean_audioset-v3-speech_creative_edit-cat2split1 "

bash launch_opuslm_stage3_sft.sh \
    --train_registered_specifier "$train_sets" \
    --valid_registered_specifier "$valid_sets" \
    --test_registered_specifier "$test_sets" \
    --inference_nj 8 --inference_workers 2 \
    --resume_path /mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_350000 \
    --train_config conf/train_stage3_ct-v2.yaml \
    --inference_config conf/inference_audio_continue.yaml \
    --exp_dir exp/ct-c2a_v2-1000k \
    "$@"
