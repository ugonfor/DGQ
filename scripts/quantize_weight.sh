# USER CONFIGURATIONS
export CUDA_VISIBLE_DEVICES=0
MODEL_TYPE=$1 # sd or sdxl
W_BITS=$2 # 4 or 8
echo "MODEL_TYPE: $MODEL_TYPE / W_BITS: $W_BITS"

# CONFIGURATIONS
export DIFFUSERS_REWRITE=$MODEL_TYPE
# inference step size # sd: 25 / sdxl: 4
if [ $MODEL_TYPE = "sd" ]; then
    STEP_SIZE=25
    NO_RECON=False # No reconstruction (True: No reconstruction, False: Reconstruction)
elif [ $MODEL_TYPE = "sdxl" ]; then
    STEP_SIZE=4
    NO_RECON=True # No reconstruction (True: No reconstruction, False: Reconstruction)
    # Memory and Computation Cost issue
fi
CALI_DATA_PATH=./data/cali_data_${MODEL_TYPE}_${STEP_SIZE}steps # calibration data path
MULTI_GPU=False # Multi-GPU
TIB_RECON=False # Use TFMQ or Q-Diffusion-> for tib recon
FAST=False # Weight Initialization with minmax or mse (True: minmax, False: mse)
outdir=results/${MODEL_TYPE}_weight # output path

if [ $MULTI_GPU = True ]; then
    python src/quantize_weight.py \
        --wq $W_BITS \
        --cali_data_path $CALI_DATA_PATH \
        --step_size $STEP_SIZE \
        --tib_recon $TIB_RECON \
        --no_recon $NO_RECON --fast $FAST \
        --multi_gpu --outdir $outdir
else
    python src/quantize_weight.py \
        --wq $W_BITS \
        --cali_data_path $CALI_DATA_PATH \
        --step_size $STEP_SIZE \
        --tib_recon $TIB_RECON \
        --no_recon $NO_RECON --fast $FAST \
        --outdir $outdir
fi