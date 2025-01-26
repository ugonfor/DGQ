import logging
import torch
from diffusers import DiffusionPipeline
import pandas as pd
import json

from typing import Dict
import os
import einops


def get_prompts(path: str,
                num: int = 64):
    '''
        COCO-Captions dataset
    '''
    df = pd.DataFrame(json.load(open(path))['annotations'])
    ps = df['caption'].sample(num).tolist()
    return ps

def collect_data(model_type, pipe, coco_path, cali_data_path, cali_prompt_data_n=64, step_size=-1, callback_keys=[]):
    logger = logging.getLogger(__name__)

    cali_data = {"timesteps": [],}
    for k in callback_keys:
        cali_data[k] = []
        
    def callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
        callback_kwargs: Dict):
        
        for k in callback_keys:
            cali_data[k].append(callback_kwargs[k].cpu())
        cali_data["timesteps"].append(timestep.expand(callback_kwargs["latents"].shape[0]).cpu()) # timestep is a scalar tensor, expand to match the batch size
        torch.cuda.empty_cache()
        return callback_kwargs

    prompts = get_prompts(coco_path, cali_prompt_data_n)
    batch_size = 8
    for i in range(0, len(prompts), batch_size):
        prompts_batch = prompts[i:i+batch_size]
        if model_type == "sdxl":
            pipe(prompts_batch, num_inference_steps=step_size,
                 guidance_scale=0.0,
                 callback_on_step_end=callback_on_step_end, 
                 callback_on_step_end_tensor_inputs=callback_keys)
        else:
            pipe(prompts_batch,  num_inference_steps=step_size,
                 callback_on_step_end=callback_on_step_end, 
                 callback_on_step_end_tensor_inputs=callback_keys) 
        
    logger.info("Calibration data generated.")
    torch.cuda.empty_cache()
    
    os.makedirs(os.path.dirname(cali_data_path), exist_ok=True)
    torch.save(cali_data, cali_data_path)
    logger.info(f"Calibration data saved to {cali_data_path}")

    return cali_data

def cali_data_preprocessing(model_type, 
                            cali_data, cali_data_size,
                            step_size, coco_path,
                            cali_prompt_data_n):
    logger = logging.getLogger(__name__)

    # max timestep  
    if model_type == "sdxl":
        T = step_size
    else:
        T = step_size + 1 # PLMS scheduler # since one more latent are saved for sdv1.4 / not SDXL

    # if use part of the calibration data,
    if (0 < cali_data_size and cali_data_size <= 12):
        raise NotImplementedError("Only for debug")
        logger.info(f"Using calibration data size: {cali_data_size}")
        cali_data = {k: v[:T] for k, v in cali_data.items()}
        keys = ["latents", "negative_prompt_embeds", "timesteps", "negative_pooled_prompt_embeds", "negative_add_time_ids"]
        _cali_data = {k: [x[:cali_data_size] for x in v] for k, v in cali_data.items() if k in keys}
            
        key = ["prompt_embeds", "latent_model_input", "add_text_embeds", "add_time_ids"]
        for k in key:
            if k in cali_data.keys():
                _cali_data[k] = [torch.cat((x[:cali_data_size], x[x.shape[0]//2: x.shape[0]//2+cali_data_size]), dim=0) for x in cali_data[k]]

        cali_data = _cali_data
    elif 12 < cali_data_size:
        logger.info("using more than 24 latents per timestep are not supported")


    # rearrange by timestep
    _cali_data = {k: [] for k in cali_data.keys()}
    for i in range(T): # since last latents are not used
        for j in range(0, len(cali_data["latents"]), T):
            for k in cali_data.keys():
                _cali_data[k].append(cali_data[k][j+i])
    cali_data = _cali_data
    
    
    '''
    # predict the noise residual
    noise_pred = self.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=timestep_cond,
        cross_attention_kwargs=self.cross_attention_kwargs,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]

    ## SDXL
    # predict the noise residual
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    if ip_adapter_image is not None:
        added_cond_kwargs["image_embeds"] = image_embeds
    noise_pred = self.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=timestep_cond,
        cross_attention_kwargs=self.cross_attention_kwargs,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]
    '''
    
    # cali_data = kwargs, latents, prompt_embeds, negative_prompt_embeds, timesteps
    # only the latent, t, prompt_embeds are needed.
    using_classifier_free_guidance = cali_data["prompt_embeds"][0].shape[0] == 2*cali_data["latents"][0].shape[0] # "classifier free guidance requires double the latents and timestep"
    if using_classifier_free_guidance:
        logger.info("using classifier free guidance -> double the latents and timestep")
        timesteps = [einops.repeat(x, 'b -> (repeat b)', repeat=2) for x in cali_data["timesteps"]]
    else:
        timesteps = cali_data["timesteps"]
    
    # sdxl vs sd
    if model_type == "sdxl":
        cali_data = (torch.cat(cali_data["latent_model_input"], dim=0),
                        torch.cat(timesteps, dim=0),
                        torch.cat(cali_data["prompt_embeds"], dim=0),
                        torch.cat(cali_data["add_text_embeds"], dim=0),
                        torch.cat(cali_data["add_time_ids"], dim=0))
        
    else:
        cali_data = (torch.cat(cali_data["latent_model_input"], dim=0), 
                    torch.cat(timesteps, dim=0),
                    torch.cat(cali_data["prompt_embeds"], dim=0),)
    
    
    # interval
    interval = len(get_prompts(coco_path, cali_prompt_data_n))
    if using_classifier_free_guidance:
        interval *= 2
    
    return cali_data, interval


def calibration_data_generation(model_type, 
                                pipe,
                                cali_data_path, 
                                coco_path,
                                cali_prompt_data_n,
                                step_size,
                                time_aware_aqtizer,
                                cali_data_size
                                ):
    logger = logging.getLogger(__name__)

    if model_type == "sdxl":
        _callback_tensor_inputs = [
            "latents", "prompt_embeds", # "negative_prompt_embeds", 
            "add_text_embeds", "add_time_ids", # "negative_pooled_prompt_embeds", 
            "negative_add_time_ids", "latent_model_input"
        ]
    elif model_type == "sd":
        _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds",
                                    "latent_model_input"]

    # collect calibration data
    if os.path.exists(cali_data_path):
        logger.info(f"Loading calibration data... {cali_data_path}")
        cali_data = torch.load(cali_data_path)
        logger.info("Calibration data loaded.")
    else:
        logger.info("Generating calibration data...")
        cali_data = collect_data(model_type, pipe, coco_path, cali_data_path, cali_prompt_data_n, step_size, callback_keys=_callback_tensor_inputs)

    # calibration data preprocessing
    cali_data, interval = cali_data_preprocessing(model_type, cali_data,
                                                  cali_data_size, step_size, coco_path, cali_prompt_data_n)

    if not time_aware_aqtizer:
        interval = cali_data[0].shape[0]

    cali_data = tuple(map(lambda x: x.to("cpu").float(), cali_data))
    w_cali_data = cali_data
    a_cali_data = cali_data
    torch.cuda.empty_cache()

    return w_cali_data, a_cali_data, interval
