import torch
import os
import argparse
import logging
import time

from pytorch_lightning import seed_everything

from quant.calibration import load_cali_model
from quant.quant_layer import QMODE, Scaler
from quant.quant_block import QuantBasicTransformerBlock
from quant.quant_model import QuantModel
from quant.load_qmodel_util import get_qmodel

from dataset_generation import get_prompts
from utils import prepare_pipe, get_file_list_from_csv, change_img_size, get_file_list_from_tsv, str2bool

MODEL_TYPE = os.environ.get("DIFFUSERS_REWRITE", "sd")


logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Activation Quantization for Diffusion Models')

    # group quantization
    parser.add_argument('--group_num', type=int, default=1, help='Use group quantization')
    parser.add_argument('--num_inference_steps', type=int, default=-1, help='Number of inference steps')

    # common setting
    parser.add_argument('--cali_ckpt', type=str, default=None, help='Calibration checkpoint path')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')

    # quantization setting
    parser.add_argument('--wq', type=int, default=4, help='Weight quantization bits')
    parser.add_argument('--use_aq', action='store_true', help='Use activation quantization')
    parser.add_argument('--aq', type=int, default=8, help='Activation quantization bits')
    
    # activation quantization setting
    parser.add_argument('--time_aware_aqtizer', type=str2bool, help='Use activation quantization with qdiff')
    parser.add_argument('--t2i_log_quant', type=str2bool, help='Use log quantization for transformer')
    parser.add_argument('--t2i_real_time', type=str2bool, help='Use real time quantization for transformer')
    parser.add_argument('--t2i_start_peak', type=str2bool, help='Use start peak quantization for transformer')
    
    # reproducibility
    parser.add_argument('--gpu_rank', type=int, required=True,
                        help='for handcrafted multi_gpu, 0~(N_GPU)')
    parser.add_argument('--world_size', type=int, required=True,
                        help='for handcrafted multi_gpu, (N_GPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    
    # default setting
    parser.add_argument("--save_dir", type=str, default=f"./samples/",
                        help="$save_dir/{model}/{im256, im512} are created for saving 256x256 and 512x512 images")
    parser.add_argument('--batch_sz', type=int, default=5, help='Batch size')
    parser.add_argument("--data_list", type=str, default="./data/mscoco_val2014_30k/metadata.csv")    
    parser.add_argument('--n_img_per_prompt', type=int, required=True, help='Number of images per prompt') # 3 for PartiPrompts, 1 for mscoco
    parser.add_argument('--img_resz', type=int, default=256, help='Image resize')


    opt = parser.parse_args()
    return opt


def run_inference(opt, file_list, save_dir_src, save_dir_tgt):
    
    ############################################################################
    # model load
    # modify the following code to use the other model

    pipe = prepare_pipe(MODEL_TYPE, fp16=opt.fp16)
    num_inference_steps = 25 if MODEL_TYPE == "sd" else 4
    if opt.num_inference_steps > 0:
        num_inference_steps = opt.num_inference_steps

    # load quantized model
    wq_params = {"bits": opt.wq,
                 "channel_wise": True,
                 "scaler": Scaler.MINMAX
                 }
    
    aq_params = {"bits": opt.aq,
                 "channel_wise": False,
                 "scaler": Scaler.MINMAX,
                 "leaf_param": opt.use_aq}
    
    softmax_aq_params = {"softmax_a_bit": opt.aq,
                 "t2i_log_quant": opt.t2i_log_quant,
                 "t2i_real_time": opt.t2i_real_time,
                 "t2i_start_peak": opt.t2i_start_peak,
                 "log_max_1": False}
    use_aq = opt.use_aq
    time_aware_aqtizer = opt.time_aware_aqtizer
    use_group = opt.group_num > 1
    qnn = get_qmodel(MODEL_TYPE, pipe, opt.cali_ckpt, wq_params, use_aq, aq_params, softmax_aq_params, 
                     use_group, num_inference_steps=num_inference_steps, time_aware_aqtizer=time_aware_aqtizer)

    # fp16
    if opt.fp16:
        qnn.half()
    else:
        qnn.float()

    pipe.unet = qnn.cuda()
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.unet.disable_out_quantization()
    pipe.to('cuda')
    ############################################################################

    print(f"rank {opt.gpu_rank} seed {opt.seed + opt.gpu_rank}")
    seed_everything(opt.seed + opt.gpu_rank)

    file_list = file_list[opt.gpu_rank*len(file_list)//opt.world_size: (opt.gpu_rank+1)*len(file_list)//opt.world_size]

    seed_everything(opt.seed)

    t0 = time.perf_counter()
    for batch_start in range(0, len(file_list), opt.batch_sz):
        batch_end = batch_start + opt.batch_sz
        
        img_names = [file_info[0] for file_info in file_list[batch_start: batch_end]]
        val_prompts = [file_info[1] for file_info in file_list[batch_start: batch_end]]

        if opt.n_img_per_prompt > 1:
            img_names = [f'{img_name}_{i}.png' for img_name in img_names for i in range(opt.n_img_per_prompt)]
            val_prompts = [val_prompt for val_prompt in val_prompts for i in range(opt.n_img_per_prompt)]

        if MODEL_TYPE == "sdxl":
            imgs = pipe(prompt = val_prompts,
                        num_inference_steps = num_inference_steps,
                        guidance_scale = 0.0,
                        ).images
        else:
            imgs = pipe(prompt = val_prompts,
                        num_inference_steps = num_inference_steps,
                        ).images

        for i, (img, img_name, val_prompt) in enumerate(zip(imgs, img_names, val_prompts)):
            img.save(os.path.join(save_dir_src, img_name))
            img.close()
            print(f"{batch_start + i}/{len(file_list)} | {img_name} {val_prompt}")
            
    change_img_size(save_dir_src, save_dir_tgt, opt.img_resz)
    print(f"{(time.perf_counter()-t0):.2f} sec elapsed")


def main():
    args = parse_args()

    output_dir_name = f"w{args.wq}a{args.aq}g{args.group_num}"
    
    save_dir_src = os.path.join(args.save_dir, MODEL_TYPE, f'{output_dir_name}/im512') # for model's raw output images
    os.makedirs(save_dir_src, exist_ok=True)
    save_dir_tgt = os.path.join(args.save_dir, MODEL_TYPE, f'{output_dir_name}/im{args.img_resz}') # for resized images for ms-coco benchmark
    os.makedirs(save_dir_tgt, exist_ok=True)       

    if args.data_list.endswith('.tsv'):
        file_list = get_file_list_from_tsv(args.data_list)
    else:
        file_list = get_file_list_from_csv(args.data_list)

    run_inference(args, file_list, save_dir_src, save_dir_tgt)

    
if __name__ == '__main__':
    main()