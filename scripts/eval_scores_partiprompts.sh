# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Please ensure that the following libraries are successfully installed:
#   for IS, https://github.com/toshas/torch-fidelity
#   for FID, https://github.com/mseitzer/pytorch-fid
#   for CLIP score, https://github.com/mlfoundations/open_clip
# ------------------------------------------------------------------------------------

GPU_NUM=$2
MODEL_ID=$1
PATH_ROOT=$PWD/samples_partiprompts/

IMG_PATH=$PATH_ROOT/$MODEL_ID/im512

# echo "=== Inception Score (IS) ==="
# IS_TXT=$PATH_ROOT/$MODEL_ID/im512_is.txt
# fidelity --gpu $GPU_NUM --isc --input1 $IMG_PATH | tee $IS_TXT
# echo "============"

# echo "=== Fréchet Inception Distance (FID) ==="
# FID_TXT=$PATH_ROOT/$MODEL_ID/im512_fid.txt
# NPZ_NAME_gen=$PATH_ROOT/$MODEL_ID/im512_fid.npz
# NPZ_NAME_real=$PATH_ROOT/fp32/im512_fid.npz
# CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid --save-stats $IMG_PATH $NPZ_NAME_gen
# CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid $NPZ_NAME_real $NPZ_NAME_gen | tee $FID_TXT
# echo "============"

NUM_EVAL=3000

echo "=== CLIP Score ==="
CLIP_TXT=$PATH_ROOT/$MODEL_ID/im512_clip.txt
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 src/eval_clip_score.py --img_dir $IMG_PATH --save_txt $CLIP_TXT --num_imgs $NUM_EVAL --data_list ./data/PartiPrompts/PartiPrompts.tsv
echo "============"

