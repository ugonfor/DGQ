import torch
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything

import json, os
import pandas as pd

import argparse
import time
from utils import get_file_list_from_csv, change_img_size, get_file_list_from_tsv

from quant.quant_block import QuantBasicTransformerBlock
from quant.quant_layer import QMODE, Scaler
from quant.quant_model import QuantModel
from quant.reconstruction_util import RLOSS

from quant.calibration import load_cali_model
from linklink import dist
import torch.multiprocessing as mp

import logging

logging.basicConfig(level=logging.INFO)

def get_prompts(path: str,
                num: int = 10000):
    '''
        COCO-Captions dataset # 10000 samples # https://arxiv.org/pdf/2302.04304
    '''
    df = pd.DataFrame(json.load(open(path))['annotations'])
    ps = df['caption'].sample(num).tolist()
    return ps

def parse_args():
    parser = argparse.ArgumentParser(description='Activation Quantization for Diffusion Models')

    # model : fp16 / tfmq_w4a16 / qdiff_w4a8s16 / ours_w4a8s16
    parser.add_argument('--model', type=str, required=True,
                        help='Model type')
    parser.add_argument('--gpu_rank', type=int, required=True,
                        help='for handcrafted multi_gpu, 0~(N_GPU)')
    parser.add_argument('--world_size', type=int, required=True,
                        help='for handcrafted multi_gpu, (N_GPU)')
    
    parser.add_argument('--n_img_per_prompt', type=int, default=1, help='Number of images per prompt') # 3 for PartiPrompts, 1 for mscoco
    

    # default setting
    parser.add_argument("--save_dir", type=str, default="./samples/",
                        help="$save_dir/{model}/{im256, im512} are created for saving 256x256 and 512x512 images")
    parser.add_argument("--data_list", type=str, default="./data/mscoco_val2014_30k/metadata.csv")    
    
    # parser.add_argument('--num_samples', type=int, default=30000, help='Number of samples')
    parser.add_argument('--batch_sz', type=int, default=5, help='Batch size')
    parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--img_resz', type=int, default=256, help='Image resize')


    opt = parser.parse_args()
    return opt


def get_model(cali_ckpt, wq=4, aq=8, softmax_a_bit=8, use_aq=False, no_forward_patch=False, 
              use_group=False, t2i_log_quant=False, t2i_real_time=False, t2i_start_peak=False,
              num_inference_steps=25):
    pipe = StableDiffusionPipeline.from_pretrained(
        "./pretrained/stable-diffusion-v1-4",
    ).to("cuda")

    # quantization setting
    aq_mode = [QMODE.NORMAL.value, QMODE.QDIFF.value]
    wq_params = {"bits": wq,
                "channel_wise": True,
                "scaler": Scaler.MINMAX}
    aq_params = {"bits": aq,
                "channel_wise": False,
                "scaler": Scaler.MINMAX,
                "leaf_param": use_aq}
    
    # quantize model
    # setattr(pipe.unet, "split", True)
    pipe.unet.float()
    qnn = QuantModel(model=pipe.unet,
                        wq_params=wq_params,
                        aq_params=aq_params,
                        cali=False,
                        softmax_a_bit=softmax_a_bit,
                        aq_mode=aq_mode,
                        t2i_log_quant=t2i_log_quant,
                        t2i_real_time=t2i_real_time,
                        t2i_start_peak=t2i_start_peak,
                        )
    qnn.to('cuda')
    qnn.eval()

    # calibration
    cali_data = (torch.randn(1, 4, 64, 64), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 768))
    if not os.path.exists(cali_ckpt):
        raise ValueError(f"Calibration checkpoint not found: {cali_ckpt}")
    load_cali_model(qnn, cali_data, use_aq=use_aq, path=cali_ckpt, forward_patch=not no_forward_patch,
                    num_inference_steps=num_inference_steps, use_group=use_group)

    # softmax quantization should be performed on float32
    for name, module in qnn.named_modules():
        if isinstance(module, QuantBasicTransformerBlock):
            module.attn1.aqtizer_w.float()
            module.attn2.aqtizer_w.float()

    # done
    # inference
    qnn.disable_out_quantization()
    pipe.unet = qnn.to('cuda')

    return pipe


def load_model_by_name(args):

    if args.model == 'fp32':
        pipe = StableDiffusionPipeline.from_pretrained(
            "./pretrained/stable-diffusion-v1-4",
        ).to("cuda")
        
    # 8bits
    elif args.model == 'qdiff_w8a32':
        pipe = get_model(cali_ckpt='pretrained/weight_only/2024-09-16-23-53-21_qdiff_w8a32/cali_ckpt.pth_weight_only',
                        wq=8, use_aq=False, no_forward_patch=False, 
                        use_group=False, t2i_log_quant=False, t2i_real_time=False, t2i_start_peak=False,
                        num_inference_steps=25)
    # 8wbits-8abits
    elif args.model == 'qdiff_w8a8':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w8a8g1/2024-09-18-12-35-32/cali_ckpt_activation_w8a8g1.pth_merged',
                        wq=8, aq=8, softmax_a_bit=8, use_aq=True, no_forward_patch=False, 
                        use_group=False, t2i_log_quant=False, t2i_real_time=False, t2i_start_peak=False)
    elif args.model == 'daq_w8a8g8':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w8a8g8/2024-09-18-12-36-15/cali_ckpt_activation_w8a8g8.pth_merged',
                        wq=8, aq=8, softmax_a_bit=8, use_aq=True, no_forward_patch=False, 
                        use_group=True, t2i_log_quant=True, t2i_real_time=True, t2i_start_peak=True)
    elif args.model == 'daq_w8a8g16':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w8a8g16/2024-09-18-12-37-19/cali_ckpt_activation_w8a8g16.pth_merged',
                        wq=8, aq=8, softmax_a_bit=8, use_aq=True, no_forward_patch=False, 
                        use_group=True, t2i_log_quant=True, t2i_real_time=True, t2i_start_peak=True)
    # 8wbits-6abits
    elif args.model == 'qdiff_w8a6':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w8a6g1/2024-09-18-12-38-13/cali_ckpt_activation_w8a6g1.pth_merged',
                        wq=8, aq=6, softmax_a_bit=6, use_aq=True, no_forward_patch=False, 
                        use_group=False, t2i_log_quant=False, t2i_real_time=False, t2i_start_peak=False)

    elif args.model == 'daq_w8a6g8':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w8a6g8/2024-09-18-12-40-51/cali_ckpt_activation_w8a6g8.pth_merged',
                        wq=8, aq=6, softmax_a_bit=6, use_aq=True, no_forward_patch=False, 
                        use_group=True, t2i_log_quant=True, t2i_real_time=True, t2i_start_peak=True)
    elif args.model == 'daq_w8a6g16':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w8a6g16/2024-09-18-12-41-09/cali_ckpt_activation_w8a6g16.pth_merged',
                        wq=8, aq=6, softmax_a_bit=6, use_aq=True, no_forward_patch=False, 
                        use_group=True, t2i_log_quant=True, t2i_real_time=True, t2i_start_peak=True)
        
    # 4bits
    elif args.model == 'qdiff_w4a32':
        pipe = get_model(cali_ckpt='pretrained/weight_only/2024-09-12-03-53-23_qdiff_w4a32/cali_ckpt.pth_weight_only',
                        wq=4, use_aq=False, no_forward_patch=False, 
                        use_group=False, t2i_log_quant=False, t2i_real_time=False, t2i_start_peak=False,
                        num_inference_steps=25)
    # 4wbits-8abits
    elif args.model == 'qdiff_w4a8':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w4a8g1/2024-09-18-10-14-39/cali_ckpt_activation_w4a8g1.pth_merged',
                        wq=4, aq=8, softmax_a_bit=8, use_aq=True, no_forward_patch=False, 
                        use_group=False, t2i_log_quant=False, t2i_real_time=False, t2i_start_peak=False)
    elif args.model == 'daq_w4a8g8':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w4a8g8/2024-09-18-12-03-50/cali_ckpt_activation_w4a8g8.pth_merged',
                        wq=4, aq=8, softmax_a_bit=8, use_aq=True, no_forward_patch=False, 
                        use_group=True, t2i_log_quant=True, t2i_real_time=True, t2i_start_peak=True)
    elif args.model == 'daq_w4a8g16':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w4a8g16/2024-09-18-12-18-21/cali_ckpt_activation_w4a8g16.pth_merged',
                        wq=4, aq=8, softmax_a_bit=8, use_aq=True, no_forward_patch=False, 
                        use_group=True, t2i_log_quant=True, t2i_real_time=True, t2i_start_peak=True)
    # 4wbits-6abits
    elif args.model == 'qdiff_w4a6':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w4a6g1/2024-09-18-12-20-35/cali_ckpt_activation_w4a6g1.pth_merged',
                        wq=4, aq=6, softmax_a_bit=6, use_aq=True, no_forward_patch=False, 
                        use_group=False, t2i_log_quant=False, t2i_real_time=False, t2i_start_peak=False)
    elif args.model == 'daq_w4a6g8':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w4a6g8/2024-09-18-12-21-15/cali_ckpt_activation_w4a6g8.pth_merged',
                        wq=4, aq=6, softmax_a_bit=6, use_aq=True, no_forward_patch=False, 
                        use_group=True, t2i_log_quant=True, t2i_real_time=True, t2i_start_peak=True)
    elif args.model == 'daq_w4a6g16':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w4a6g16/2024-09-18-12-21-35/cali_ckpt_activation_w4a6g16.pth_merged',
                        wq=4, aq=6, softmax_a_bit=6, use_aq=True, no_forward_patch=False, 
                        use_group=True, t2i_log_quant=True, t2i_real_time=True, t2i_start_peak=True)
        
    # qdiff_original
    elif args.model == 'qdiff_original_w8a8':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w8a8g1_originalqdiff/2024-09-22-10-39-23/cali_ckpt_activation_w8a8g1.pth_merged',
                        wq=8, aq=8, softmax_a_bit=8, use_aq=True, no_forward_patch=True, 
                        use_group=False, t2i_log_quant=False, t2i_real_time=False, t2i_start_peak=False)
    elif args.model == 'qdiff_original_w8a6':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w8a6g1_originalqdiff/2024-09-22-10-40-33/cali_ckpt_activation_w8a6g1.pth_merged',
                        wq=8, aq=6, softmax_a_bit=6, use_aq=True, no_forward_patch=True, 
                        use_group=False, t2i_log_quant=False, t2i_real_time=False, t2i_start_peak=False)
    elif args.model == 'qdiff_original_w4a8':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w4a8g1_originalqdiff/2024-09-22-10-41-42/cali_ckpt_activation_w4a8g1.pth_merged',
                        wq=4, aq=8, softmax_a_bit=8, use_aq=True, no_forward_patch=True, 
                        use_group=False, t2i_log_quant=False, t2i_real_time=False, t2i_start_peak=False)
    elif args.model == 'qdiff_original_w4a6':
        pipe = get_model(cali_ckpt='pretrained/weight_activation/w4a6g1_originalqdiff/2024-09-22-10-41-56/cali_ckpt_activation_w4a6g1.pth_merged',
                        wq=4, aq=6, softmax_a_bit=6, use_aq=True, no_forward_patch=True, 
                        use_group=False, t2i_log_quant=False, t2i_real_time=False, t2i_start_peak=False)

    
    else:
        raise NotImplementedError

    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    return pipe
    

def run_inference(args, file_list, save_dir_src, save_dir_tgt):
    
    ############################################################################
    # model load
    # modify the following code to use the other model

    pipe = load_model_by_name(args)
    if args.model != 'fp32':
        pipe.unet.disable_out_quantization()

    pipe.to('cuda')
    ############################################################################

    print(f"rank {args.gpu_rank} seed {args.seed + args.gpu_rank}")
    seed_everything(args.seed + args.gpu_rank)

    file_list = file_list[args.gpu_rank*len(file_list)//args.world_size: (args.gpu_rank+1)*len(file_list)//args.world_size]

    seed_everything(args.seed)

    t0 = time.perf_counter()
    for batch_start in range(0, len(file_list), args.batch_sz):
        batch_end = batch_start + args.batch_sz
        
        img_names = [file_info[0] for file_info in file_list[batch_start: batch_end]]
        val_prompts = [file_info[1] for file_info in file_list[batch_start: batch_end]]

        if args.n_img_per_prompt > 1:
            img_names = [f'{img_name}_{i}.png' for img_name in img_names for i in range(args.n_img_per_prompt)]
            val_prompts = [val_prompt for val_prompt in val_prompts for i in range(args.n_img_per_prompt)]

        imgs = pipe(prompt = val_prompts,
                    num_inference_steps = args.num_inference_steps,
                    ).images

        for i, (img, img_name, val_prompt) in enumerate(zip(imgs, img_names, val_prompts)):
            img.save(os.path.join(save_dir_src, img_name))
            img.close()
            print(f"{batch_start + i}/{len(file_list)} | {img_name} {val_prompt}")
            
    change_img_size(save_dir_src, save_dir_tgt, args.img_resz)
    print(f"{(time.perf_counter()-t0):.2f} sec elapsed")


def main():
    args = parse_args()
    
    save_dir_src = os.path.join(args.save_dir, f'{args.model}/im512') # for model's raw output images
    os.makedirs(save_dir_src, exist_ok=True)
    save_dir_tgt = os.path.join(args.save_dir, f'{args.model}/im{args.img_resz}') # for resized images for ms-coco benchmark
    os.makedirs(save_dir_tgt, exist_ok=True)       

    if args.data_list.endswith('.tsv'):
        file_list = get_file_list_from_tsv(args.data_list)
    else:
        file_list = get_file_list_from_csv(args.data_list)

    

    run_inference(args, file_list, save_dir_src, save_dir_tgt)

    
if __name__ == '__main__':
    main()