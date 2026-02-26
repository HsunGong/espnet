MAIN_ROOT=$PWD/../../..

export PATH=$PWD/utils/:$PATH
export LC_ALL=C
export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=${MAIN_ROOT}:PYTHONPATH

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

# NOTE(kamo): Source at the last to overwrite the setting
. local/path.sh

# NOTE(Jinchuan): avoid pytorch memory segmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ESPNET_DATASET_REGISTRY=
# # NOTE(Jinchuan): selectively enable this for wavlab internal usage.
# if [[ "$(hostname)" == dt* ]] || [[ "$(hostname)" == gh* ]] || [[ "$(hostname)" == gpu* ]] ; then # For Delta/DeltaAI
#     ESPNET_DATASET_REGISTRY+=":/work/nvme/bbjs/shared/data_registry/train_shared.yaml"
#     ESPNET_DATASET_REGISTRY+=":/work/nvme/bbjs/shared/data_registry/valid_shared.yaml"

#     # OpusLM v2
#     ESPNET_DATASET_REGISTRY+=":/work/nvme/bbjs/shared/opuslm_v2_data/data_jsons/opuslm_v2.yaml"

#     ESPNET_DATASET_REGISTRY+=":/work/nvme/bbjs/shared/opuslm_v2_data/data_curation/stage5_curated_sound_und/registry.yaml"
#     ESPNET_DATASET_REGISTRY+=":/work/nvme/bbjs/shared/opuslm_v2_data/data_curation/stage5_curated_speech_und/registry.yaml"
#     ESPNET_DATASET_REGISTRY+=":/work/nvme/bbjs/shared/opuslm_v2_data/data_curation/stage5_curated_music_und/registry.yaml"

#     ESPNET_DATASET_REGISTRY+=":/work/nvme/bbjs/shared/opuslm_v2_data/data_curation/stage5_curated_sound_gen/registry.yaml"
#     ESPNET_DATASET_REGISTRY+=":/work/nvme/bbjs/shared/opuslm_v2_data/data_curation/stage5_curated_speech_gen/registry.yaml"
#     ESPNET_DATASET_REGISTRY+=":/work/nvme/bbjs/shared/opuslm_v2_data/data_curation/stage5_curated_music_gen/registry.yaml"
# fi

export ESPNET_DATASET_REGISTRY="/mnt/home/jinchuat-andr-d6b58f/jinchuat/data/data_jsons/opuslm_v2.yaml:/mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_sft/egs2/opuslm_v2/speechlm1/data/sft.yaml"

export ESPNET_DATASET_REGISTRY="${ESPNET_DATASET_REGISTRY}:./data/debug.yaml"
export ESPNET_DATASET_REGISTRY="${ESPNET_DATASET_REGISTRY}:/mnt/home/xungong-andr-1766e0/opuslm_sft/egs2/opuslm_v2/speechlm1/data/part2_4/debug/data.yaml"
export ESPNET_DATASET_REGISTRY="${ESPNET_DATASET_REGISTRY}:/mnt/home/xungong-andr-1766e0/opuslm_sft/egs2/opuslm_v2/speechlm1/data/part2_4/full/data.yaml"


export ESPNET_DATASET_REGISTRY="${ESPNET_DATASET_REGISTRY}:/mnt/home/xungong-andr-1766e0/opuslm_sft/egs2/opuslm_v2/speechlm1/data/test_clean/speech_edit/dialogues/data.yaml"
export ESPNET_DATASET_REGISTRY="${ESPNET_DATASET_REGISTRY}:/mnt/home/xungong-andr-1766e0/opuslm_sft/egs2/opuslm_v2/speechlm1/data/test_clean/audio_edit/dialogues/data.yaml"
export ESPNET_DATASET_REGISTRY="${ESPNET_DATASET_REGISTRY}:/mnt/home/xungong-andr-1766e0/opuslm_sft/egs2/opuslm_v2/speechlm1/data/test_clean/speech_edit-short/dialogues/data.yaml"
export ESPNET_DATASET_REGISTRY="${ESPNET_DATASET_REGISTRY}:/mnt/home/xungong-andr-1766e0/opuslm_sft/egs2/opuslm_v2/speechlm1/data/test_clean/speech_edit-short/dialogues/data.yaml"
export ESPNET_DATASET_REGISTRY="${ESPNET_DATASET_REGISTRY}:/mnt/home/xungong-andr-1766e0/opuslm_sft/egs2/opuslm_v2/speechlm1/data/test_clean/freeform-edit/dialogues/data.yaml"

# NOTE(Jinchuan): For DeltaAI users, un-comment this for network setup
# export NCCL_DEBUG=WARN
# export NCCL_SOCKET_IFNAME=hsn
# module load nccl # loads the nccl built with the AWS nccl plugin for Slingshot11%  