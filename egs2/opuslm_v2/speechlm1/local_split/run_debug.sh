#!/bin/bash

# bash local_split/run_all.sh data/part2_4/debug part2_4_debug --k 10 --skip_generate true
expdir=$1
name_prefix=$2
shift 2

echo yaml at "${expdir}/data.yaml"

# not-include vad part
for data_type in music.min_0.max_10 sound.min_0.max_7 sound.min_7.max_10 sound.min_10 speech.min_0.max_3 speech.min_5.max_8 speech.min_8.max_20;do
    bash local_split/run.sh \
        --input_jsonl_raw "data/part2_4/metadata.${data_type}.jsonl" \
        --expdir "${expdir}/novad-${data_type}" \
        --yaml_path "${expdir}/data.yaml" \
        --name_prefix "${name_prefix}-novad-${data_type}" \
        --run_stages 0,4,6,8 "${@}"
done
wait

# include VAD part
for data_type in music.min_10 sound.min_7.max_10 sound.min_10 speech.min_20.max_25 speech.min_25; do
    bash local_split/run.sh \
        --input_jsonl_raw "data/part2_4/metadata.${data_type}.jsonl" \
        --expdir "${expdir}/vad-${data_type}" \
        --yaml_path "${expdir}/data.yaml" \
        --name_prefix "${name_prefix}-vad-${data_type}" \
        --run_stages 1,2,4,5,6,8 "${@}"
done
wait


# python local_split/generate_filtered_test_script.py --template-sh runs/run-pretrain-audio-continue.sh --include split1 --output-sh runs/test_all_split1.sh
# python local_split/generate_filtered_test_script.py --template-sh runs/run-pretrain-audio.sh --exclude split1 --output-sh runs/test_all_normal.sh
