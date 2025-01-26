MODEL_TYPE=$1
W_BITS=$2
A_BITS=$3
GROUP_NUM=$4
CALI_CKPT=$5
echo "MODEL_TYPE: $MODEL_TYPE / W_BITS: $W_BITS / A_BITS: $A_BITS / GROUP_NUM: $GROUP_NUM"
echo "CALI_CKPT: $CALI_CKPT"

# DGQ CONFIGURATIONS
export DIFFUSERS_REWRITE=$MODEL_TYPE
if [ $MODEL_TYPE = "sd" ]; then
    STEP_SIZE=25
elif [ $MODEL_TYPE = "sdxl" ]; then
    STEP_SIZE=4
fi

# ABLATION CONFIGURATIONS
T2I_LOG_QUANT=True
T2I_REAL_TIME=True
T2I_START_PEAK=True
TIME_AWARE_AQTIZER=True
if [ $GROUP_NUM = "1" ]; then
    T2I_LOG_QUANT=False
    T2I_REAL_TIME=False
    T2I_START_PEAK=False
fi
echo "TIME_AWARE_AQTIZER: $TIME_AWARE_AQTIZER / T2I_LOG_QUANT: $T2I_LOG_QUANT / T2I_REAL_TIME: $T2I_REAL_TIME / T2I_START_PEAK: $T2I_START_PEAK"


# EVALUATION DATASET CONFIGURATIONS
DATASET=COCO # Partiprompts
# Defaults
if [ "$DATASET" == "COCO" ] ; then
    SAVE_DIR=$PWD/samples
    BATCH_SZ=3
    DATA_LIST=./data/mscoco_val2014_30k/metadata.csv
    N_IMG_PER_PROMPT=1

elif [ "$DATASET" == "Partiprompts" ] ; then
    SAVE_DIR=$PWD/samples_partiprompts/
    BATCH_SZ=1
    DATA_LIST=./data/PartiPrompts/PartiPrompts.tsv
    N_IMG_PER_PROMPT=3
fi

# RUN EVALUATION
if [ "$MODEL_TYPE" == "sd" ]; then
    echo "SD Not supported"
    exit 1
fi

# SDXL
CUDA_VISIBLE_DEVICES=0 python src/gen4eval_SDXL.py --group_num $GROUP_NUM --num_inference_steps $STEP_SIZE \
    --cali_ckpt $CALI_CKPT \
    --wq $W_BITS --use_aq --aq $A_BITS \
    --time_aware_aqtizer $TIME_AWARE_AQTIZER --t2i_log_quant $T2I_LOG_QUANT --t2i_real_time $T2I_REAL_TIME --t2i_start_peak $T2I_START_PEAK \
    --gpu_rank 0 --world_size 8 \
    --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=1 python src/gen4eval_SDXL.py --group_num $GROUP_NUM --num_inference_steps $STEP_SIZE \
    --cali_ckpt $CALI_CKPT \
    --wq $W_BITS --use_aq --aq $A_BITS \
    --time_aware_aqtizer $TIME_AWARE_AQTIZER --t2i_log_quant $T2I_LOG_QUANT --t2i_real_time $T2I_REAL_TIME --t2i_start_peak $T2I_START_PEAK \
    --gpu_rank 1 --world_size 8 \
    --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=2 python src/gen4eval_SDXL.py --group_num $GROUP_NUM --num_inference_steps $STEP_SIZE \
    --cali_ckpt $CALI_CKPT \
    --wq $W_BITS --use_aq --aq $A_BITS \
    --time_aware_aqtizer $TIME_AWARE_AQTIZER --t2i_log_quant $T2I_LOG_QUANT --t2i_real_time $T2I_REAL_TIME --t2i_start_peak $T2I_START_PEAK \
    --gpu_rank 2 --world_size 8 \
    --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=3 python src/gen4eval_SDXL.py --group_num $GROUP_NUM --num_inference_steps $STEP_SIZE \
    --cali_ckpt $CALI_CKPT \
    --wq $W_BITS --use_aq --aq $A_BITS \
    --time_aware_aqtizer $TIME_AWARE_AQTIZER --t2i_log_quant $T2I_LOG_QUANT --t2i_real_time $T2I_REAL_TIME --t2i_start_peak $T2I_START_PEAK \
    --gpu_rank 3 --world_size 8 \
    --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=4 python src/gen4eval_SDXL.py --group_num $GROUP_NUM --num_inference_steps $STEP_SIZE \
    --cali_ckpt $CALI_CKPT \
    --wq $W_BITS --use_aq --aq $A_BITS \
    --time_aware_aqtizer $TIME_AWARE_AQTIZER --t2i_log_quant $T2I_LOG_QUANT --t2i_real_time $T2I_REAL_TIME --t2i_start_peak $T2I_START_PEAK \
    --gpu_rank 4 --world_size 8 \
    --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=5 python src/gen4eval_SDXL.py --group_num $GROUP_NUM --num_inference_steps $STEP_SIZE \
    --cali_ckpt $CALI_CKPT \
    --wq $W_BITS --use_aq --aq $A_BITS \
    --time_aware_aqtizer $TIME_AWARE_AQTIZER --t2i_log_quant $T2I_LOG_QUANT --t2i_real_time $T2I_REAL_TIME --t2i_start_peak $T2I_START_PEAK \
    --gpu_rank 5 --world_size 8 \
    --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=6 python src/gen4eval_SDXL.py --group_num $GROUP_NUM --num_inference_steps $STEP_SIZE \
    --cali_ckpt $CALI_CKPT \
    --wq $W_BITS --use_aq --aq $A_BITS \
    --time_aware_aqtizer $TIME_AWARE_AQTIZER --t2i_log_quant $T2I_LOG_QUANT --t2i_real_time $T2I_REAL_TIME --t2i_start_peak $T2I_START_PEAK \
    --gpu_rank 6 --world_size 8 \
    --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=7 python src/gen4eval_SDXL.py --group_num $GROUP_NUM --num_inference_steps $STEP_SIZE \
    --cali_ckpt $CALI_CKPT \
    --wq $W_BITS --use_aq --aq $A_BITS \
    --time_aware_aqtizer $TIME_AWARE_AQTIZER --t2i_log_quant $T2I_LOG_QUANT --t2i_real_time $T2I_REAL_TIME --t2i_start_peak $T2I_START_PEAK \
    --gpu_rank 7 --world_size 8 \
    --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT 