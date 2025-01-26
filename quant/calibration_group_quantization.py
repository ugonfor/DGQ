from typing import Any, Dict, Tuple, Union
import torch.nn as nn
import torch
import numpy as np
from quant.quant_block import BaseQuantBlock
from quant.quant_model import QuantModel
from quant.quant_layer import QuantLayer
from quant.adaptive_rounding import AdaRoundQuantizer, RMODE
from quant.quant_layer import UniformAffineQuantizer
from tqdm import trange

from quant.reconstruction import block_reconstruction, layer_reconstruction, tib_reconstruction
import linklink as dist
import logging
import os
import yaml
logger = logging.getLogger(__name__)

def uaq2adar(model: nn.Module):
    for _, child in model.named_children():
        if isinstance(child, QuantLayer):
            if not child.ignore_recon:
                child.wqtizer = AdaRoundQuantizer(child.wqtizer,
                                                rmode = RMODE.LEARNED_HARD_SIGMOID,
                                                w = child.original_w.data)
        elif isinstance(child, BaseQuantBlock):
            if not child.ignore_recon:
                for _, sub_child in child.named_modules():
                    if isinstance(sub_child, QuantLayer):
                        if not hasattr(sub_child, 'wqtizer1'):
                            sub_child.wqtizer = AdaRoundQuantizer(sub_child.wqtizer,
                                                                rmode = RMODE.LEARNED_HARD_SIGMOID,
                                                                w = sub_child.original_w.data)
                        else:
                            sub_child.wqtizer = AdaRoundQuantizer(sub_child.wqtizer,
                                                                rmode = RMODE.LEARNED_HARD_SIGMOID,
                                                                w = sub_child.original_w.data[:, :sub_child.split, ...])
                            sub_child.wqtizer1 = AdaRoundQuantizer(sub_child.wqtizer1,
                                                                rmode = RMODE.LEARNED_HARD_SIGMOID,
                                                                w = sub_child.original_w.data[:, sub_child.split:, ...])
        else:
            uaq2adar(child)

@torch.no_grad()
def cali_model_aq(model_type, qnn: QuantModel, a_cali_data, model_dict, group_num, interval, group_mode):
    qnn.cuda()
    qnn.eval()
    cali_data = a_cali_data
    for time in range(cali_data[0].shape[0] // interval):
        t_cali_data = tuple([x[time * interval: (time + 1) * interval] for x in cali_data])
        
        qnn.set_quant_state(use_wq = True, use_aq = True)
        qnn.disable_out_quantization()

        for name, module in qnn.model.named_modules():
            if 'aqtizer' in name:
                if isinstance(module, UniformAffineQuantizer):
                    del module.delta
                    del module.zero_point
                    module.delta = None
                    module.zero_point = None
                    module.init = False
                else:
                    del module.delta
                    module.delta = None
                    module.init = False

        # --------- activation quantization -------- #
        # batch_size
        if model_type == 'sd':
            batch_size = min(8, t_cali_data[0].shape[0]) 
        elif model_type == 'sdxl':
            batch_size = min(4, t_cali_data[0].shape[0])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # calibrate activation quantization
        with torch.no_grad():
            inds = np.random.choice(t_cali_data[0].shape[0], batch_size, replace=False)
            inputs = (x[inds].cuda() for x in t_cali_data)
            _ = qnn(*inputs)
            logger.info(f'group_num: {group_num} running stat for activation calibration...')
            inds = np.arange(t_cali_data[0].shape[0])
            np.random.shuffle(inds)
            qnn.set_group_num(group_num)
            for i in trange(0, t_cali_data[0].shape[0], batch_size):
                inputs = (x[inds[i: i + batch_size]].cuda() for x in t_cali_data)
                _ = qnn(*inputs)
            qnn.done_group_num(group_num, mode=group_mode)
            logger.info(f'group_num: {group_num} running stat for activation calibration done.')
            torch.cuda.empty_cache()

        # save the quantization parameters
        for name, module in qnn.model.named_modules():
            if 'aqtizer' in name:
                if isinstance(module, UniformAffineQuantizer) and module.delta is not None:
                    if not torch.is_tensor(module.zero_point):
                        module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                    else:
                        module.zero_point = nn.Parameter(module.zero_point)
    
        temp = {}
        for name, module in qnn.model.named_modules():
            if 'aqtizer' in name and len(list(module.cpu().state_dict().keys())) == 2:
                temp['model.' + name + '.delta'] = module.cpu().state_dict()['delta']
                temp['model.' + name + '.zero_point'] = module.cpu().state_dict()['zero_point']
        model_dict['act_{}'.format(time)] = temp
    
    return model_dict

def act_group_quant(model_type,
                    qnn: QuantModel,
                      a_cali_data: Tuple[torch.Tensor],
                      path: str = None,
                      group_num: int = 1,
                      interval: int = 128,
                      group_mode = 'minmax',
                      **kwargs
                      ) -> None:
    logger.info("Calibrating...")

    # --------- activation quantization -------- #
    model_dict = {}
    model_dict = cali_model_aq(model_type, qnn, a_cali_data, model_dict, group_num, interval, group_mode=group_mode)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model_dict, path)
    logger.info("calibrated model saved to {}".format(path))
    logger.info("Calibration done.")
    