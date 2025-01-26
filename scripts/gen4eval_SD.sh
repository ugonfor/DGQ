MODEL=fp32
DATASET=COCO

# Defaults
if [ "$DATASET" == "COCO" ] ; then
    SAVE_DIR=$PWD/samples
    BATCH_SZ=5
    DATA_LIST=./data/mscoco_val2014_30k/metadata.csv
    N_IMG_PER_PROMPT=1

elif [ "$DATASET" == "Partiprompts" ] ; then
    SAVE_DIR=$PWD/samples_partiprompts/
    BATCH_SZ=1
    DATA_LIST=./data/PartiPrompts/PartiPrompts.tsv
    N_IMG_PER_PROMPT=3
fi


CUDA_VISIBLE_DEVICES=0 python src/gen4eval_SD.py --model $MODEL --gpu_rank 0 --world_size 8 --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=1 python src/gen4eval_SD.py --model $MODEL --gpu_rank 1 --world_size 8 --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=2 python src/gen4eval_SD.py --model $MODEL --gpu_rank 2 --world_size 8 --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=3 python src/gen4eval_SD.py --model $MODEL --gpu_rank 3 --world_size 8 --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=4 python src/gen4eval_SD.py --model $MODEL --gpu_rank 4 --world_size 8 --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=5 python src/gen4eval_SD.py --model $MODEL --gpu_rank 5 --world_size 8 --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=6 python src/gen4eval_SD.py --model $MODEL --gpu_rank 6 --world_size 8 --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT &
CUDA_VISIBLE_DEVICES=7 python src/gen4eval_SD.py --model $MODEL --gpu_rank 7 --world_size 8 --save_dir $SAVE_DIR --batch_sz $BATCH_SZ --data_list $DATA_LIST --n_img_per_prompt $N_IMG_PER_PROMPT 