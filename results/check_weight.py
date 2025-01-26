
# delete folders that contain only 'run.log' file.
import glob
import os
import sys
import torch

# ./results/sd_weight/2024-09-12-03-53-23/cali_ckpt.pth_weight_only
path = sys.argv[1]

ckpt = torch.load(path, map_location='cpu')['weight']
ckpt_keys = list(ckpt.keys())

for files in os.listdir(os.path.dirname(path)):
    if files.endswith(".pth"):
        tmp = torch.load(os.path.join(os.path.dirname(path), files), map_location='cpu')

        for key in tmp.keys():
            ckpt_key = files[5:-4] + '.' + key
            if torch.all(ckpt[ckpt_key] == tmp[key]):
                print(f"Found {ckpt_key} in {files}")
                ckpt_keys.remove(ckpt_key)
            else:
                print(f"Not found {ckpt_key} in {files}")        
                break

print(len(ckpt_keys))
print(ckpt_keys)
