#!/usr/bin/env bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li, Wangyou Zhang)
# Apache 2.0
set -e
set -u
set -o pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 [--min_or_max <min/max>] [--sample_rate <8k/16k>]
  optional argument:
    [--min_or_max]: min (Default), max
    [--sample_rate]: 8k (Default), 16k
EOF
)

. ./db.sh

# Path to the directory containing WHAM! noise
# (will download from the official site if not specified)
wham_noise=

min_or_max=max
sample_rate=16k
num_spk=2

stage=0
stop_stage=100

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ ! -e "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi

if [[ "$min_or_max" != "max" ]] && [[ "$min_or_max" != "min" ]]; then
  echo "Error: min_or_max must be either max or min: ${min_or_max}"
  exit 1
fi
if [[ "$sample_rate" != "16k" ]] && [[ "$sample_rate" != "8k" ]]; then
  echo "Error: sample rate must be either 16k or 8k: ${sample_rate}"
  exit 1
fi
if [[ "$num_spk" != "2" ]] && [[ "$num_spk" != "3" ]]; then
  echo "Error: num_spk must be either 2 or 3"
  exit 1
fi

cdir=$PWD


[ -d ./data/LibriMix ] || git clone https://github.com/JorisCos/LibriMix ./data/LibriMix

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Downloading WHAM! noise data to '${cdir}/data/wham_noise'"

    # Download WHAM noise data
    if [ -z "${wham_noise}" ]; then
        # 17.65 GB unzipping to 35 GB
        mkdir -p ${cdir}/data/wham_noise
        wham_noise_url=https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip # https://storage.googleapis.com/whisper-public/wham_noise.zip
        wget --continue -O "${cdir}/data/wham_noise.zip" ${wham_noise_url}
        num_wavs=$(find "${cdir}/data/wham_noise" -iname "*.wav" | wc -l)
        if [ "${num_wavs}" -ge "4" ]; then
            echo "'${cdir}/data/wham_noise/' already exists. Skipping..."
        else
            unzip "${cdir}/data/wham_noise.zip" -d "${cdir}/data/"
        fi
        wham_noise="${cdir}/data/wham_noise"
    else
        # The simulation program will write data to wham_noie,
        # so copy it to user directory in case of permission issues.
        rsync -r -P "${wham_noise}" "${cdir}/data/wham_noise"
    fi
fi

if [ -z "${wham_noise}" ]; then
    wham_noise="${cdir}/data/wham_noise"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data simulation"
    (
    cd ./data/LibriMix
    librimix_outdir=./libri_mix

    python scripts/augment_train_noise.py --wham_dir ${cdir}/data/wham_noise
    # shellcheck disable=SC2043
    metadata_dir="metadata/Libri${num_spk}Mix"
    python scripts/create_librimix_from_metadata.py --librispeech_dir $LIBRISPEECH \
        --wham_dir "${cdir}/data/wham_noise" \
        --metadata_dir $metadata_dir \
        --librimix_outdir $librimix_outdir \
        --n_src $num_spk \
        --freqs $sample_rate \
        --modes $min_or_max \
        --types mix_clean mix_both mix_single
    )
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: LibriMix data preparation (mix_both)"

    librimix="data/LibriMix/libri_mix/Libri2Mix"
    for dset in dev test train; do
        mkdir -p "data_mix_both/${dset}"
        if [ "$dset" = "train" ]; then
            cat ${librimix}/wav${sample_rate}/${min_or_max}/metadata/mixture_train-*_mix_both.csv | grep -v mixture_ID | sort -u > "data_mix_both/${dset}/tmp"
        else
            grep -v mixture_ID ${librimix}/wav${sample_rate}/${min_or_max}/metadata/mixture_${dset}_mix_both.csv | sort -u > "data_mix_both/${dset}/tmp"
        fi
        awk -F ',' '{print $1, $1}' "data_mix_both/${dset}/tmp" > "data_mix_both/${dset}/utt2spk"
        awk -F ',' '{print $1, $1}' "data_mix_both/${dset}/tmp" > "data_mix_both/${dset}/spk2utt"
        awk -F ',' '{print $1, $2}' "data_mix_both/${dset}/tmp" > "data_mix_both/${dset}/wav.scp"
        awk -F ',' '{print $1, $3}' "data_mix_both/${dset}/tmp" > "data_mix_both/${dset}/spk1.scp"
        awk -F ',' '{print $1, $4}' "data_mix_both/${dset}/tmp" > "data_mix_both/${dset}/spk2.scp"
        if [ $num_spk -eq 2 ]; then
            awk -F ',' '{print $1, $5}' "data_mix_both/${dset}/tmp" > "data_mix_both/${dset}/noise1.scp"
        else
            awk -F ',' '{print $1, $5}' "data_mix_both/${dset}/tmp" > "data_mix_both/${dset}/spk3.scp"
            awk -F ',' '{print $1, $6}' "data_mix_both/${dset}/tmp" > "data_mix_both/${dset}/noise1.scp"
        fi
        rm "data_mix_both/${dset}/tmp"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: LibriMix data preparation (mix_single)"

    librimix="data/LibriMix/libri_mix/Libri2Mix"
    for dset in dev test train; do
        mkdir -p "data_mix_single/${dset}"
        if [ "$dset" = "train" ]; then
            cat ${librimix}/wav${sample_rate}/${min_or_max}/metadata/mixture_train-*_mix_single.csv | grep -v mixture_ID | sort -u > "data_mix_single/${dset}/tmp"
        else
            grep -v mixture_ID ${librimix}/wav${sample_rate}/${min_or_max}/metadata/mixture_${dset}_mix_single.csv | sort -u > "data_mix_single/${dset}/tmp"
        fi
        awk -F ',' '{print $1, $1}' "data_mix_single/${dset}/tmp" > "data_mix_single/${dset}/utt2spk"
        awk -F ',' '{print $1, $1}' "data_mix_single/${dset}/tmp" > "data_mix_single/${dset}/spk2utt"
        awk -F ',' '{print $1, $2}' "data_mix_single/${dset}/tmp" > "data_mix_single/${dset}/wav.scp"
        awk -F ',' '{print $1, $3}' "data_mix_single/${dset}/tmp" > "data_mix_single/${dset}/spk1.scp"
        awk -F ',' '{print $1, $4}' "data_mix_single/${dset}/tmp" > "data_mix_single/${dset}/noise1.scp"
        rm "data_mix_single/${dset}/tmp"
    done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: Prepare data files for train-100 and train-360"

    for mix_type in mix_both mix_single; do
        mkdir -p data_${mix_type}/{train-100,train-360}

        for subset in "train-100" "train-360"; do
            grep -e "${subset}" "data_${mix_type}/train/wav.scp" > "data_${mix_type}/${subset}/wav.scp"
            for f in data_${mix_type}/train/*.scp; do
                [ "$f" = "data_${mix_type}/train/wav.scp" ] || utils/filter_scp.pl "data_${mix_type}/${subset}/wav.scp" "$f" > "data_${mix_type}/${subset}/$(basename $f)"
            done
            utils/filter_scp.pl "data_${mix_type}/${subset}/wav.scp" data_${mix_type}/train/utt2spk > data_${mix_type}/${subset}/utt2spk
        done
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: LibriMix Transcripts Files Prepration."

    mkdir -p data/local
    for dset in "train-clean-100" "train-clean-360" "dev-clean" "test-clean"; do
        local/asr_data_prep.sh ${LIBRISPEECH}/${dset} data/local/${dset}
        cp data/local/${dset}/text data/local/${dset}.txt
    done
    cat data/local/train-clean-100.txt data/local/train-clean-360.txt > data/local/train_clean_460.txt

    for mix_type in mix_both mix_single; do
        for dset in "dev" "test" "train" "train-100" "train-360"; do
            if [[ " ${dset} " =~ train* ]]; then
                src_ls_dset_txt="train_clean_460.txt"
            else
                src_ls_dset_txt="${dset}-clean.txt"
            fi

            if [ "${mix_type}" = "mix_single" ]; then
                nspk=1
            else
                nspk=${num_spk}
            fi

            for i in $(seq ${nspk}); do
                awk -v idx=${i} '(FILENAME==ARGV[1]) {text[$1]=$0} (FILENAME==ARGV[2]) {split($1, lst, "_"); uttname=lst[idx]; print($1, text[uttname])}' \
                    data/local/${src_ls_dset_txt} data_${mix_type}/${dset}/wav.scp | \
                    cut -d" " -f1,3- > data_${mix_type}/${dset}/text_spk${i}
            done
        done
    done

    rm -rf data/local
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
