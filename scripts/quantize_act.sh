# USER CONFIGURATIONS
# export CUDA_VISIBLE_DEVICES=2
MODEL_TYPE=$1 # sd or sdxl
W_BITS=$2 # 4 or 8
A_BITS=$3 # 4 or 8
WEIGHT_ONLY_CKPT=$4
echo "MODEL_TYPE: $MODEL_TYPE / W_BITS: $W_BITS / A_BITS: $A_BITS / WEIGHT_ONLY_CKPT: $WEIGHT_ONLY_CKPT"

# DGQ CONFIGURATIONS
echo "CONFIGURATION FOR DGQ"
GROUP_NUM=$5 # 1 or 8 or 16
GROUP_MODE=minmax
echo "GROUP_NUM: $GROUP_NUM / GROUP_MODE: $GROUP_MODE"

# ABLATION CONFIGURATIONS
T2I_LOG_QUANT=True
T2I_REAL_TIME=True
T2I_START_PEAK=True
TIME_AWARE_AQTIZER=True
if [ $GROUP_NUM -eq 1 ]; then
    T2I_LOG_QUANT=False
    T2I_REAL_TIME=False
    T2I_START_PEAK=False
fi
echo "TIME_AWARE_AQTIZER: $TIME_AWARE_AQTIZER / T2I_LOG_QUANT: $T2I_LOG_QUANT / T2I_REAL_TIME: $T2I_REAL_TIME / T2I_START_PEAK: $T2I_START_PEAK"

# CONFIGURATIONS
export DIFFUSERS_REWRITE=$MODEL_TYPE
# inference step size # sd: 25 / sdxl: 4
if [ $MODEL_TYPE = "sd" ]; then
    STEP_SIZE=25
elif [ $MODEL_TYPE = "sdxl" ]; then
    STEP_SIZE=4
    # Memory and Computation Cost issue
fi

# Quantization settings
SOFTMAX_A_BIT=$A_BITS

# Default settings
CALI_DATA_PATH=./data/cali_data_${MODEL_TYPE}_${STEP_SIZE}steps # calibration data path
OUTDIR=results/${MODEL_TYPE}_activation/w${W_BITS}a${A_BITS}g${GROUP_NUM}/

python src/quantize_act.py \
    --outdir $OUTDIR \
    --weight_only_ckpt $WEIGHT_ONLY_CKPT \
    --wq $W_BITS --aq $A_BITS --softmax_a_bit $SOFTMAX_A_BIT \
    --time_aware_aqtizer $TIME_AWARE_AQTIZER \
    --t2i_log_quant $T2I_LOG_QUANT --t2i_real_time $T2I_REAL_TIME --t2i_start_peak $T2I_START_PEAK \
    --group_num $GROUP_NUM --group_mode $GROUP_MODE \
    --cali_data_path $CALI_DATA_PATH --step_size $STEP_SIZE \


