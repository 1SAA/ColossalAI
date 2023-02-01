# activate conda environment
source /fsx/software/miniconda3/bin/activate hhc

# set environment vars
export FI_PROVIDER=efa
export FI_OFI_RXR_RX_COPY_UNEXP=1
export FI_OFI_RXR_RX_COPY_OOO=1
export FI_EFA_MR_CACHE_ENABLE=1
export FI_OFI_RXR_INLINE_MR_ENABLE=1
export NCCL_DEBUG=INFO

export HUGGINGFACE_HUB_CACHE=/fsx/users/hhc/hf_cache

set -x
# distplan in ["CAI_ZeRO1", "CAI_ZeRO2", "CAI_Gemini", "Pytorch_DDP", "Pytorch_ZeRO"]
export DISTPLAN=${DISTPLAN:-"CAI_Gemini"}

# The following options only valid when DISTPLAN="colossalai"
export GPUNUM=${GPUNUM:-16}
export TPDEGREE=${TPDEGREE:-1}
export PLACEMENT=${PLACEMENT:-"cpu"}
export USE_SHARD_INIT=${USE_SHARD_INIT:-True}
export BATCH_SIZE=${BATCH_SIZE:-16}
export MODEL_TYPE=${MODEL_TYPE:-"gpt2_medium"}
export TRAIN_STEP=${TRAIN_STEP:-6}
# export PYTHONPATH=$PWD:$PYTHONPATH

if [ ${USE_SHARD_INIT} = "True" ]; then
  USE_SHARD_INIT="--shardinit"
else
  USE_SHARD_INIT=""
fi

mkdir -p gemini_logs

colossalai run --nproc_per_node=8 \
--host=192.168.2.65,192.168.2.53 \
--master_addr=192.168.2.65 ./train_gpt_demo.py \
--tp_degree=${TPDEGREE} \
--model_type=${MODEL_TYPE} \
--batch_size=${BATCH_SIZE} \
--placement=${PLACEMENT} \
${USE_SHARD_INIT} \
--distplan=${DISTPLAN} \
--train_step=${TRAIN_STEP} \
2>&1 | tee ./gemini_logs/${MODEL_TYPE}_${DISTPLAN}_gpu_${GPUNUM}_bs_${BATCH_SIZE}_tp_${TPDEGREE}_${PLACEMENT}.log
