import torch
import os
import argparse
import logging

from pytorch_lightning import seed_everything

from quant.quant_layer import Scaler
from quant.quant_block import QuantBasicTransformerBlock
from utils import prepare_pipe

from quant.load_qmodel_util import get_qmodel

MODEL_TYPE = os.environ.get("DIFFUSERS_REWRITE", "sd")

def parse_args():
    parser = argparse.ArgumentParser(description='Activation Quantization for Diffusion Models')
    
    # group quantization
    parser.add_argument('--use_group', action='store_true', help='Use group quantization')
    parser.add_argument('--num_inference_steps', type=int, default=-1, help='Number of inference steps')

    # testprompt
    parser.add_argument('--prompt', type=str, default="a painting of a virus monster playing guitar", help='Test prompt path')

    # common setting
    parser.add_argument('--cali_ckpt', type=str, default=None, help='Calibration checkpoint path')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')

    # quantization setting
    parser.add_argument('--wq', type=int, default=4, help='Weight quantization bits')
    parser.add_argument('--use_aq', action='store_true', help='Use activation quantization')
    parser.add_argument('--aq', type=int, default=8, help='Activation quantization bits')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # activation quantization setting
    parser.add_argument('--t2i_log_quant', action='store_true', help='Use log quantization for transformer')
    parser.add_argument('--t2i_real_time', action='store_true', help='Use real time quantization for transformer')
    parser.add_argument('--t2i_start_peak', action='store_true', help='Use start peak quantization for transformer')
    parser.add_argument('--time_aware_aqtizer', action='store_true', help='Use activation quantization with qdiff')

    opt = parser.parse_args()

    return opt

def inference(model_type, pipe, prompt, precision, num_inference_steps, seed=42):
    seed_everything(seed)
    if model_type == "sdxl":
        images = pipe([prompt]*2, num_inference_steps=num_inference_steps, guidance_scale=0.0)[0]
    else:
        images = pipe([prompt]*2, num_inference_steps=num_inference_steps)[0]
    for i in range(len(images)):
        images[i].save(f"tmp_{model_type}_{prompt.replace(' ', '_')}_{i}_{precision}.png")


def main():
    opt = parse_args()
    
    # load model
    pipe = prepare_pipe(MODEL_TYPE)

    # setting up the environment
    seed_everything(opt.seed)
    logging.basicConfig(level=logging.INFO)
    num_inference_steps = 25 if MODEL_TYPE == "sd" else 4
    if opt.num_inference_steps > 0:
        num_inference_steps = opt.num_inference_steps

    # inference fp model
    inference(MODEL_TYPE, pipe, opt.prompt, 'fp', num_inference_steps)

    # quantize model
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
    time_aware_aqtizer = opt.time_aware_aqtizer if use_aq else False
    qnn = get_qmodel(MODEL_TYPE, pipe, opt.cali_ckpt, wq_params, use_aq, aq_params, softmax_aq_params, 
                     opt.use_group, num_inference_steps=num_inference_steps, time_aware_aqtizer=time_aware_aqtizer)

    # done
    # inference
    if opt.fp16:
        qnn.half()
    else:
        qnn.float()
    
    pipe.unet = qnn.cuda()

    pipe.unet.disable_out_quantization()
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    # "A majestic mountain range with snow-capped peaks rising high into the sky, surrounded by a lush green forest at its base, under a clear blue sky with a few wispy clouds floating by. The scene is bathed in the golden light of the setting sun, casting long shadows and a warm glow over the landscape, creating a serene and breathtaking view."
    inference(MODEL_TYPE, pipe, opt.prompt, f"w{opt.wq}a{opt.aq if opt.use_aq else 32}{'g?' if opt.use_group else 'g1'}", num_inference_steps)

    breakpoint()

if __name__ == "__main__":
    main()
