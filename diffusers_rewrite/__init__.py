import os
if os.environ.get("DIFFUSERS_REWRITE", "sd") == "sdxl":
    from diffusers_rewrite.sdxl import *
else:
    from diffusers_rewrite.sd import *
print("Using diffusers_rewrite: ", os.environ.get("DIFFUSERS_REWRITE", "sd"))