GPU_NUM=$2
MODEL_ID=$1
PATH_ROOT=$PWD/samples/

IMG_PATH=$PATH_ROOT/$MODEL_ID/im256

NUM_EVAL=3000

echo "=== ImageReward Score ==="
CLIP_TXT=$PATH_ROOT/$MODEL_ID/im256_ImageReward.txt
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 src/eval_ImageReward_score.py --img_dir $IMG_PATH --save_txt $CLIP_TXT --num_imgs $NUM_EVAL
echo "============"