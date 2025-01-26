# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

# Modified by ugonfor

import csv
import os
from PIL import Image
from collections.abc import Mapping
import torch

from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from diffusers_rewrite import UNet2DConditionModel

def prepare_pipe(model_type, fp16=False):

    if model_type == "sd" and not os.path.exists("./pretrained/stable-diffusion-v1-4"):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        unet_new = UNet2DConditionModel()
        unet_new.load_state_dict(pipe.unet.state_dict())
        pipe.unet = unet_new
        pipe.save_pretrained("./pretrained/stable-diffusion-v1-4")

    elif model_type == "sdxl" and not os.path.exists("./pretrained/sdxl-turbo"):
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo")
        unet_new = UNet2DConditionModel()
        unet_new.load_state_dict(pipe.unet.state_dict())
        pipe.unet = unet_new
        pipe.save_pretrained("./pretrained/sdxl-turbo")
        
    else:
        pass

    # load model
    if model_type == "sdxl":
        if fp16:
            pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained("./pretrained/sdxl-turbo",
                                                                            torch_dtype=torch.float16,
                                                                            variant='fp16').to("cuda")
        else:
            pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained("./pretrained/sdxl-turbo").to("cuda")
    elif model_type == "sd":
        if fp16:
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained("./pretrained/stable-diffusion-v1-4",
                                                                            torch_dtype=torch.float16,
                                                                            variant='fp16').to("cuda")
        else:
            pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained("./pretrained/stable-diffusion-v1-4").to("cuda")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return pipe

def nested_device(tensors, device):
    "Move `tensors` (even if it's a nested list/tuple/dict of tensors) to `device`."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_device(t, device) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_device(t, device) for k, t in tensors.items()})
    return tensors.to(device) if isinstance(tensors, torch.Tensor) else tensors

def nested_float(tensors):
    "Convert `tensors` (even if it's a nested list/tuple/dict of tensors) to float."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_float(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_float(t) for k, t in tensors.items()})
    return tensors.float() if isinstance(tensors, torch.Tensor) else tensors

def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach() if isinstance(tensors, torch.Tensor) else tensors

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_file_list_from_csv(csv_file_path):
    file_list = []
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)        
        next(csv_reader, None) # Skip the header row
        for row in csv_reader: # (row[0], row[1]) = (img name, txt prompt) 
            file_list.append(row)
    return file_list

def get_file_list_from_tsv(tsv_file_path):
    file_list = []
    idx = 0
    with open(tsv_file_path, newline='') as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t')
        next(tsv_reader, None) # Skip the header row
        for row in tsv_reader:
            row = [f'{idx}.png', row[0]] # (idx, row[0]) = (img name, txt prompt)
            file_list.append(row)
            idx += 1
    return file_list

    

def change_img_size(input_folder, output_folder, resz=256, num_imgs=None):
    img_list = sorted([file for file in os.listdir(input_folder) if file.endswith('.jpg')])
    if num_imgs is not None:
        img_list = img_list[:num_imgs]
    for i, filename in enumerate(img_list):
        img = Image.open(os.path.join(input_folder, filename))
        img.resize((resz, resz)).save(os.path.join(output_folder, filename))
        img.close()
        if i % 2000 == 0:
            print(f"{i}/{len(img_list)} | {filename}: resize to {resz}")
