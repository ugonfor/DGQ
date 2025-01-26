from diffusers import StableDiffusionPipeline, DiffusionPipeline, StableDiffusionXLPipeline
from diffusers_rewrite.sd import UNet2DConditionModel

import torch
import os, sys
import numpy as np
import argparse

from typing import List, Tuple, Dict

from quant.calibration import load_cali_model
from quant.calibration_group_quantization import act_group_quant

from quant.quant_layer import QMODE, Scaler
from quant.quant_model import QuantModel
from quant.reconstruction_util import RLOSS
from quant.load_qmodel_util import setup_pipe_to_calibrate

from pytorch_lightning import seed_everything
import logging
import datetime
import pandas as pd

from dataset_generation import calibration_data_generation
from utils import str2bool, nested_device, nested_float, prepare_pipe

import torch.multiprocessing as mp



MODEL_TYPE = os.environ.get("DIFFUSERS_REWRITE", "sd")



def parse_args():
    parser = argparse.ArgumentParser(description='Activation Quantization for Diffusion Models')
    
    # common setting
    parser.add_argument('--outdir', type=str, default='results', help='Output directory')
    parser.add_argument('--weight_only_ckpt', type=str, default=None, help='Quantized Weight checkpoint path')

    # quantization setting
    parser.add_argument('--wq', type=int, default=4, help='Weight quantization bits')
    parser.add_argument('--aq', type=int, default=8, help='Activation quantization bits')
    parser.add_argument('--softmax_a_bit', type=int, default=8, help='Softmax activation quantization bits')

    # aqtizer setting
    parser.add_argument('--time_aware_aqtizer', type=str2bool, help='time aware quantizer')
    parser.add_argument('--t2i_log_quant', type=str2bool, help='use log quantizer for transformer block')
    parser.add_argument('--t2i_real_time', type=str2bool, help='use real time quantizer for transformer block')
    parser.add_argument('--t2i_start_peak', type=str2bool, help='start peak for transformer block')

    # group quantization
    parser.add_argument('--group_num', type=int, default=1, help='Group number')
    parser.add_argument('--group_mode', type=str, choices=['mean', 'minmax', 'test'], default='minmax', help='Group mode')

    # for reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--coco_path', type=str, default='/raid/workspace/cvml_user/rhg/dataset/COCO/annotations/captions_train2017.json', help='COCO dataset path')
    
    # calibration setting
    parser.add_argument('--cali_prompt_data_n', default=64, type=int, help='Number of calibration prompt data')
    parser.add_argument('--cali_data_path', type=str, default='./data/cali_data', help='Calibration data path')
    parser.add_argument('--cali_data_size', type=int, default=-1, help='Calibration data size (latents per each timestep)')
    parser.add_argument('--step_size', type=int, default=25)
    
    opt = parser.parse_args()
    opt.running_stat = True
    opt.asym = True

    return opt

def setup(seed, outdir):
    print("Setting up - seed, logging, outpath")

    # setting up the environment
    seed_everything(seed)

    # outpath setting
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(outpath)

    # logging setting
    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    return outpath, logger


def main():
    opt = parse_args()

    # load model
    pipe = prepare_pipe(MODEL_TYPE)

    # setup
    outpath, logger = setup(opt.seed, opt.outdir)
    
    # log setting
    logger.info(f"sys.argv: {sys.argv}")

    # calibration data generation
    cali_data_args = dict(pipe=pipe, 
                        cali_data_path = opt.cali_data_path,
                        coco_path = opt.coco_path,
                        cali_prompt_data_n = opt.cali_prompt_data_n,
                        step_size = opt.step_size,
                        time_aware_aqtizer = opt.time_aware_aqtizer,
                        cali_data_size = opt.cali_data_size)
    w_cali_data, a_cali_data, interval = calibration_data_generation(MODEL_TYPE, **cali_data_args)

    # quantization setting
    wq_params = {"bits": opt.wq,
                 "channel_wise": True,
                 "scaler": Scaler.MINMAX}
    
    aq_params = {"bits": opt.aq,
                 "channel_wise": False,
                 "scaler": Scaler.MINMAX,
                 "leaf_param": True}
    
    softmax_aq_params = {"softmax_a_bit": opt.softmax_a_bit,
                 "t2i_log_quant": opt.t2i_log_quant,
                 "t2i_real_time": opt.t2i_real_time,
                 "t2i_start_peak": opt.t2i_start_peak,
                 "log_max_1": False} # for further exp.
    
    # quantization
    setup_pipe_to_calibrate(MODEL_TYPE, pipe)
    QuantModel_args = dict(model=pipe.unet,
                            wq_params=wq_params,
                            aq_params=aq_params,
                            softmax_aq_params=softmax_aq_params,
                            aq_mode=[QMODE.NORMAL.value, QMODE.QDIFF.value],
                            tib_recon=False,)
    qnn = QuantModel(**QuantModel_args).to('cuda').eval()

    # load_weight_only_ckpt
    dummy_cali_data = tuple([data[0:1] for data in w_cali_data])
    cali_model_args = dict(init_data=dummy_cali_data, 
                           use_aq=False,
                           path=opt.weight_only_ckpt)
    load_cali_model(qnn, **cali_model_args)


    # activation quantization (DGQ)
    group_quant_args = dict(a_cali_data=a_cali_data,
                            path=os.path.join(outpath, f"cali_ckpt_activation_w{opt.wq}a{opt.aq}g{opt.group_num}.pth"),
                            group_num=opt.group_num,
                            group_mode=opt.group_mode,
                            interval=interval,)
    act_group_quant(MODEL_TYPE, qnn, **group_quant_args)
    
    logger.info("Activation quantization is done")

if __name__ == "__main__":
    main()
