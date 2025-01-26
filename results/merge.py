import torch
import os 
import sys

# w?a?g? path
if len(sys.argv) != 3:
    print("Usage: python merge.py <path_to_weight_ckpt> <path_to_act_ckpt>")
    sys.exit(1)

weight_path = sys.argv[1]
act_path = sys.argv[2]

ckpt1 = torch.load(weight_path)
ckpt2 = torch.load(act_path)
ckpt3 = ckpt2
ckpt3['weight'] = ckpt1['weight']

torch.save(ckpt3, act_path+"_merged")
print(f'Merged ckpt saved to {act_path}_merged')