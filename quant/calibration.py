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
def cali_model_aq(qnn, a_cali_data, model_dict, running_stat, interval):
    qnn.cuda()
    qnn.eval()
    cali_data = a_cali_data
    for time in range(cali_data[0].shape[0] // interval):
        t_cali_data = tuple([x[time * interval: (time + 1) * interval] for x in cali_data])
        qnn.set_quant_state(use_wq = True, use_aq = True)
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

        batch_size = min(8, t_cali_data[0].shape[0])
        with torch.no_grad():
            inds = np.random.choice(t_cali_data[0].shape[0], batch_size, replace=False)
            inputs = (x[inds].cuda() for x in t_cali_data)
            _ = qnn(*inputs)
            if running_stat:
                logger.info('running stat for activation calibration...')
                inds = np.arange(t_cali_data[0].shape[0])
                np.random.shuffle(inds)
                qnn.set_running_stat(True)
                for i in trange(0, t_cali_data[0].shape[0], batch_size):
                    inputs = (x[inds[i: i + batch_size]].cuda() for x in t_cali_data)
                    _ = qnn(*inputs)
                qnn.set_running_stat(False)
                logger.info('running stat for activation calibration done.')
            torch.cuda.empty_cache()
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


def cali_model(qnn: QuantModel,
                      w_cali_data: Tuple[torch.Tensor],
                      a_cali_data: Tuple[torch.Tensor],
                      use_aq: bool = False,
                      path: str = None,
                      running_stat: bool = False,
                      interval: int = 128,
                      tib_recon: bool = False,
                      **kwargs
                      ) -> None:
    logger.info("Calibrating...")
    def recon_model(model: nn.Module, prev_name=None) -> bool:
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            logger.info(f'block name: {prev_name}.{name} | quantblock: {isinstance(module, BaseQuantBlock)} | quantlayer: {isinstance(module, QuantLayer)}')

            if 'down_blocks' in prev_name:
                kwargs['keep_gpu'] = True # default
            else:
                kwargs['keep_gpu'] = False
            
            if name == 'tib':
                continue
            if name == 'time_embedding' and tib_recon:
                logger.info('Reconstruction for time embedding')
                tib_reconstruction(qnn.tib, cali_data=cali_data, **kwargs)
                continue
            if isinstance(module, QuantLayer): 
                if not module.ignore_recon:
                    logger.info(f'Reconstruction for layer {prev_name}.{name}')
                    layer_reconstruction(qnn, module, cali_data=cali_data, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if not module.ignore_recon:
                    logger.info(f'Reconstruction for block {prev_name}.{name}')
                    block_reconstruction(qnn, module, cali_data=cali_data, **kwargs)
            else:
                recon_model(module, prev_name=f"{prev_name}.{name}")

    # --------- weight initialization -------- #
    logger.info('weight initialization...')
    cali_data = w_cali_data
    qnn.set_quant_state(use_wq = True, use_aq = False)
    batch_size = min(1, cali_data[0].shape[0])
    inputs = (x[: batch_size].cuda() for x in cali_data) 
    with torch.no_grad():
        qnn(*inputs)
    qnn.disable_out_quantization()
    logger.info('weight initialization done.')

    # --------- weight quantization -------- #
    if kwargs['resume_w']:
        qnn.set_quant_state(use_wq = True, use_aq = False)
        delattr(qnn, 'tib')

        model_dict = torch.load(kwargs['resume_w'])['weight']
        for key in model_dict.keys():
            if 'alpha' in key: # have used brecq
                uaq2adar(qnn)
                break

        for name, module in qnn.model.named_modules():
            if 'wqtizer' in name:
                if not torch.is_tensor(module.zero_point):
                    module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                else:
                    module.zero_point = nn.Parameter(module.zero_point)
                module.delta = nn.Parameter(module.delta)
        
        tmp = qnn.load_state_dict(model_dict, strict=False)
        logger.info(f"quantized model loaded from {kwargs['resume_w']}")
        logger.info(f"keys not loaded: {tmp}")
        logger.info("weight quantization done.")

    else:   
        del kwargs['resume_w']
        no_recon = kwargs.pop('no_recon', False)

        if not no_recon:
            recon_model(qnn, 'unet')

        qnn.set_quant_state(use_wq = True, use_aq = False)
        if hasattr(qnn, 'tib'): delattr(qnn, 'tib')

        for name, module in qnn.model.named_modules():
            if 'wqtizer' in name:
                if not torch.is_tensor(module.zero_point):
                    module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                else:
                    module.zero_point = nn.Parameter(module.zero_point)
                module.delta = nn.Parameter(module.delta)
        model_dict = {'weight': qnn.cpu().state_dict()}

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model_dict, f'{path}_weight_only')
        logger.info("calibrated model saved to {}".format(f'{path}_weight_only'))
        logger.info("Calibration done.")

    if use_aq:
        model_dict = {'weight': model_dict}
        model_dict = cali_model_aq(qnn, a_cali_data, model_dict, running_stat, interval)
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model_dict, path)
        logger.info("calibrated model saved to {}".format(path))
        logger.info("Calibration done.")
    
@torch.no_grad()
def load_cali_model(qnn: QuantModel,
                    init_data: Tuple[torch.Tensor],
                    use_aq: bool = False,
                    path: str = None,
                    time_aware_aqtizer: bool = False,
                    num_inference_steps: int = 25,
                    use_group: bool = False
                    ) -> None:
    
    logger.info("Loading calibration model...")
    
    ckpt = torch.load(path, map_location='cpu')
    if 'weight' in ckpt.keys():
        ckpt = ckpt['weight']
        
    qnn.set_quant_state(use_wq = True, use_aq = False)
    _ = qnn(*(_.cuda() for _ in init_data))
    qnn.disable_out_quantization()
    for key in ckpt.keys():
        if 'alpha' in key: # have used brecq
            uaq2adar(qnn)
            break
        
    for name, module in qnn.model.named_modules():
        if "wqtizer" in name:
            if isinstance(module, (UniformAffineQuantizer, AdaRoundQuantizer)):
                if not torch.is_tensor(module.zero_point):
                    module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                else:
                    module.zero_point = nn.Parameter(module.zero_point)
                module.delta = nn.Parameter(module.delta)
            
    keys = [key for key in ckpt.keys() if "aqtizer" in key]
    for key in keys:
        del ckpt[key]

    if 'model' in  list(ckpt.keys())[0]:
        tmp_keys = qnn.load_state_dict(ckpt, strict=False) # TODO: recon ?
        logger.info(f"keys not loaded: {tmp_keys}")

    else:
        tmp_keys = qnn.model.load_state_dict(ckpt, strict=False)
        logger.info(f"keys not loaded: {tmp_keys}")

    qnn.set_quant_state(use_wq=True, use_aq=False)
    
    if use_aq:
        qnn.set_quant_state(use_wq=True, use_aq=True)
        _ = qnn(*(_.cuda() for _ in init_data))
        
        for module in qnn.model.modules():
            if isinstance(module, (UniformAffineQuantizer, AdaRoundQuantizer)) and module.delta is not None:
                if not torch.is_tensor(module.zero_point):
                    module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                else:
                    module.zero_point = nn.Parameter(module.zero_point)
            if isinstance(module, AdaRoundQuantizer):
                module.delta = nn.Parameter(module.delta)

        def load_act_ckpt_with_difference_shape(ckpt, qnn):
            non_loaded_keys = list(ckpt.keys())

            for name, module in qnn.named_modules():
                # use_group_num on QuantLayer (for Conv2D)
                if f'{name}.aqtizer.delta' in ckpt and not module.use_group_num:
                    # check if the shape of delta, zero_point is the same
                    if module.aqtizer.delta.shape == ckpt[f'{name}.aqtizer.delta'].shape:
                        continue
                    else:
                        module.use_group_num = True
                
                # delta, zero_point shape difference
                if f'{name}.delta' in ckpt:
                    if hasattr(module, 'delta') and module.delta is not None:
                        module.delta.data = ckpt[f'{name}.delta'].to(module.delta.device)
                    if hasattr(module, 'zero_point') and module.zero_point is not None:
                        module.zero_point.data = ckpt[f'{name}.zero_point'].to(module.zero_point.device)
                    
                    non_loaded_keys.remove(f'{name}.delta')
                    non_loaded_keys.remove(f'{name}.zero_point')
            
            if not len(non_loaded_keys) == 0:
                print(f"keys not loaded: {non_loaded_keys}")

        if use_group:
            act_ckpt = torch.load(path, map_location='cpu')['act_0']
            load_act_ckpt_with_difference_shape(act_ckpt, qnn)

        if time_aware_aqtizer: # TFMQ
            def forward(
                self, sample, timesteps, encoder_hidden_states, **kwargs
                ):
                if timesteps.size() == torch.Size([]):
                    act = self.ckpt[f'act_{int((1000 - timesteps.item())//(1000//num_inference_steps))}']
                else:
                    act = self.ckpt[f'act_{int((1000 - timesteps[0].item())//(1000//num_inference_steps))}']
                
                load_act_ckpt_with_difference_shape(act, self)
                # self.load_state_dict(act, strict=False)

                return self.model(sample, timesteps, encoder_hidden_states, **kwargs)
            
            setattr(qnn, "ckpt", torch.load(path))
            qnn.forward = forward.__get__(qnn)
        else: # QDiff
            ckpt = torch.load(path)
            if 'act_0' in list(ckpt.keys()):
                ckpt = ckpt['act_0']
            
            if 'model' in list(ckpt.keys())[0]:
                tmp_keys = qnn.load_state_dict(ckpt, strict=False)
                tmp_keys = [key for key in tmp_keys if 'aqtizer' in key]
                logger.info(f"keys not loaded: {tmp_keys}")
            else:
                tmp_keys = qnn.model.load_state_dict(ckpt, strict=False)
                tmp_keys = [key for key in tmp_keys if 'aqtizer' in key]
                logger.info(f"keys not loaded: {tmp_keys}")
    
    logger.info("Loading calibration model done.")


#  ------------- multi-gpu calibration -------------- #
def cali_model_multi(gpu: int,
                      dist_backend: str,
                      world_size: int,
                      dist_url: str,
                      rank: int,
                      ngpus_per_node: int,
                      unet: nn.Module,
                      use_aq: bool,
                      path: str,
                      w_cali_data: Tuple[torch.Tensor],
                      a_cali_data: Tuple[torch.Tensor],
                      interval: int, # samples per t
                      running_stat: bool,
                      qdiff: bool,
                      kwargs: Dict[str, Any]
                      ) -> None:
    # raise NotImplementedError
    
    log_path = os.path.join(os.path.dirname(path), "run.log")
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

    if gpu is not None:
        logger.info("Use GPU: {} for calibration.".format(gpu))

    rank = rank * ngpus_per_node + gpu
    dist.init_process_group(backend=dist_backend,
                            init_method=dist_url,
                            world_size=world_size,
                            rank=rank)
    torch.cuda.set_device(gpu)
    unet.cuda()
    unet.eval()
    qnn = QuantModel(unet,
                     wq_params=kwargs['wq_params'],
                     aq_params=kwargs['aq_params'],
                     softmax_a_bit=kwargs['softmax_a_bit'],
                     aq_mode=kwargs['aq_mode'],
                     qdiff=qdiff,
                     t2i_log_quant=kwargs.get('t2i_log_quant', False),
                     t2i_real_time=kwargs.get('t2i_real_time', False),
                     t2i_start_peak=kwargs.get('t2i_start_peak', False),
                     )
                     
    # if 'no_grad_ckpt' in kwargs.keys() and kwargs['no_grad_ckpt']:
    #     qnn.set_grad_ckpt(False)
    #     del kwargs['no_grad_ckpt']
    del kwargs['wq_params']
    del kwargs['aq_params']
    del kwargs['softmax_a_bit']
    del kwargs['aq_mode']

    if 'target_info_path' in kwargs.keys():
        target_info = yaml.load(open(kwargs['target_info_path'], 'r'), Loader=yaml.FullLoader)
        for name, module in qnn.model.named_modules():
            if isinstance(module, (QuantLayer, BaseQuantBlock)):
                in_names = target_info['in_names'] if target_info['in_names'] else []

                _in = True
                for in_name in in_names:
                    if not in_name in name:
                        _in = False
                        break

                if not _in:
                    module.ignore_recon = True

        del kwargs['target_info_path']
    
    qnn.cuda()
    qnn.eval()
    torch.backends.cudnn.benchmark = False

    c = []
    for i in range(len(w_cali_data)):
        d = []
        if type(w_cali_data[i]) == torch.Tensor:
            for j in range(w_cali_data[i].shape[0] // interval):
                d.append(w_cali_data[i][j * interval + gpu * interval // world_size: j * interval + (gpu + 1) * interval // world_size])
            c.append(torch.cat(d, dim=0))
        elif type(w_cali_data[i]) == dict:
            for key in w_cali_data[i].keys():
                e = []
                for j in range(w_cali_data[i][key].shape[0] // interval):
                    e.append(w_cali_data[i][key][j * interval + gpu * interval // world_size: j * interval + (gpu + 1) * interval // world_size])
                d.append({key: torch.cat(e, dim=0)})
            c.append(d)
    w_cali_data = tuple(c)
    c = []

    

    logger.info("Calibrating...")

    def recon_model(model: nn.Module, prev_name=None):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if rank == 0:
                logger.info(f'block name: {prev_name}.{name} | quantblock: {isinstance(module, BaseQuantBlock)} | quantlayer: {isinstance(module, QuantLayer)}')

            kwargs['keep_gpu'] = True # default
            
            if name == 'tib':
                continue
            if name == 'time_embedding' and not qdiff:
                if rank == 0:
                    logger.info('Reconstruction for time embedding')
                tib_reconstruction(qnn.tib, cali_data=cali_data, **kwargs)
                continue
            if isinstance(module, QuantLayer):
                if not module.ignore_recon:
                    if rank == 0:
                        logger.info(f'Reconstruction for layer {prev_name}.{name}')
                    layer_reconstruction(qnn, module, cali_data=cali_data, **kwargs)
                    if rank == 0:
                        logger.info(f"layer {prev_name}.{name} done. -> saved to {os.path.dirname(path)}/{prev_name}.{name}.pth")
                        torch.save(module.state_dict(), f'{os.path.dirname(path)}/{prev_name}.{name}.pth')
            elif isinstance(module, BaseQuantBlock):
                if not module.ignore_recon:
                    if rank == 0:
                        logger.info(f'Reconstruction for block {prev_name}.{name}')
                    block_reconstruction(qnn, module, cali_data=cali_data, **kwargs)
                    if rank == 0:
                        logger.info(f"block {prev_name}.{name} done. -> saved to {os.path.dirname(path)}/{prev_name}.{name}.pth")
                        torch.save(module.state_dict(), f'{os.path.dirname(path)}/{prev_name}.{name}.pth')
            else:
                recon_model(module, prev_name=f"{prev_name}.{name}")

    # --------- weight initialization -------- #
    cali_data = w_cali_data
    qnn.set_quant_state(use_wq = True, use_aq = False)
    for name, module in qnn.model.named_modules():
        if 'wqtizer' in name:
            module: Union[UniformAffineQuantizer, AdaRoundQuantizer]
            module.init = False
    batch_size = min(4, cali_data[0].shape[0])
    inputs = (x[: batch_size].cuda() for x in cali_data)
    qnn(*inputs)
    qnn.disable_out_quantization()

    # --------- weight quantization -------- #
    recon_model(qnn, 'unet')
    qnn.set_quant_state(use_wq = True, use_aq = False)
    delattr(qnn, 'tib')

    if rank == 0:
        for name, module in qnn.model.named_modules():
            if 'wqtizer' in name:
                if not torch.is_tensor(module.zero_point):
                    module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                else:
                    module.zero_point = nn.Parameter(module.zero_point)
                module.delta = nn.Parameter(module.delta)
        model_dict = {'weight': qnn.cpu().state_dict()}
    
    if path and rank == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model_dict, f'{path}_weight_only')
        logger.info("calibrated model saved to {}".format(path))

    if use_aq:
        for i in range(len(a_cali_data)):
            d = []
            for j in range(a_cali_data[i].shape[0] // interval):
                d.append(a_cali_data[i][j * interval + gpu * interval // world_size: j * interval + (gpu + 1) * interval // world_size])
            c.append(torch.cat(d, dim=0))
        a_cali_data = tuple(c)
        
        qnn.cuda()
        qnn.eval()
        cali_data = a_cali_data
        interval = interval // world_size
        for time in range(cali_data[0].shape[0] // interval):
            t_cali_data = tuple([x[time * interval: (time + 1) * interval] for x in cali_data])
            qnn.set_quant_state(use_wq = True, use_aq = True)
            for name, module in qnn.model.named_modules():
                if 'aqtizer' in name:
                    del module.delta
                    del module.zero_point
                    module.delta = None
                    module.zero_point = None
                    module.init = False

            batch_size = min(16, t_cali_data[0].shape[0])
            with torch.no_grad():
                inds = np.random.choice(t_cali_data[0].shape[0], 16, replace=False)
                inputs = (x[inds].cuda() for x in t_cali_data)
                _ = qnn(*inputs)
                if running_stat:
                    logger.info('running stat for activation calibration...')
                    inds = np.arange(t_cali_data[0].shape[0])
                    np.random.shuffle(inds)
                    qnn.set_running_stat(True)
                    for i in trange(0, t_cali_data[0].shape[0], batch_size):
                        inputs = (x[inds[i: i + batch_size]].cuda() for x in t_cali_data)
                        _ = qnn(*inputs)
                    qnn.set_running_stat(False)
                    logger.info('running stat for activation calibration done.')
                if ngpus_per_node > 1:
                    qnn.synchorize_activation_statistics()
                torch.cuda.empty_cache()
            for name, module in qnn.model.named_modules():
                if 'aqtizer' in name:
                    if isinstance(module, UniformAffineQuantizer) and module.delta is not None:
                        if not torch.is_tensor(module.zero_point):
                            module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                        else:
                            module.zero_point = nn.Parameter(module.zero_point)
            if rank == 0:
                temp = {}
                for name, module in qnn.model.named_modules():
                    if 'aqtizer' in name and len(list(module.cpu().state_dict().keys())) == 2:
                        temp['model.' + name + '.delta'] = module.cpu().state_dict()['delta']
                        temp['model.' + name + '.zero_point'] = module.cpu().state_dict()['zero_point']
                model_dict['act_{}'.format(time)] = temp
                    
        if path and rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model_dict, path)
            logger.info("calibrated model saved to {}".format(path))
    logger.info("Calibration done.")