<h1 align="center">DGQ: Distribution-Aware Group Quantization <br> for Text-to-Image Diffusion Models </h1>

<div align="center">
  <a href="https://ugonfor.kr/" target="_blank">Hyogon&nbsp;Ryu</a><sup></sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://sites.google.com/view/cvml-kaist/" target="_blank">NaHyeon&nbsp;Park</a><sup></sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://sites.google.com/view/cvml-kaist/" target="_blank">Hyunjung&nbsp;Shim</a><sup>*</sup><br>
  <sup></sup>Korea Advanced Institute of Science and Technology (KAIST)<br>
  <sup>*</sup>Corresponding Author<br>
</div>

<div align="center">[<a href="https://ugonfor.kr/DGQ">project page</a>]&emsp;[<a href="http://arxiv.org/abs/2501.04304">arXiv</a>]</div>
<br>

<div align="center">
  <img src="https://ugonfor.kr/DGQ/static/images/teaser.jpg" alt="DGQ" width="75%">
</div>
<br>

<b>TL;DR</b>: We propose a distribution-aware group quantization method to preserve both image quality and text-to-image alignment in T2I diffusion models.

---

### Update

- **2025.01.26:** Release the code for Stable Diffusion v1.4 and SDXL-turbo. 
- **2025.01.24:** DGQ has been accepted to **ICLR2025**! We are working on releasing the code.

## Requirements

```
conda env create -f ./environment.yaml
conda activate DGQ
pip install -e .
cd diffusers
pip install -e .
```
All experiments are conducted on NVIDIA RTX A6000 (CUDA v12.2)

## How to use

### Calibration and inference

1. For weight quantization, use the following commands to the quantize weight or download the prequantized weight checkpoints on [here](https://1drv.ms/f/c/51b9d1e1871d8328/EiwL7PnUdtNFqSC9fZhRv5oB8lcLHQ2A2tgrk0Zgnwf2FQ?e=8HJIZ0).
```
# check the configurations in the shell script and run
bash scripts/quantize_weight.sh <model_type: sd/sdxl> <w_bits: 4/8>
```

2. For test weight only quantized model, use following commands.
```
DIFFUSERS_REWRITE=<model_type: sd/sdxl> python src/inference_qmodel.py --cali_ckpt <path/to/cali_ckpt.pth_weight_only> --wq <w_bits: 4/8>
```

3. Now, there are the weight quantized checkpoint, activation quantization could be performed. Use following commands to quantize act. Then, activation quantizer's checkpoints will be saved.
```
# check the configurations in the shell script and run
bash scripts/quantize_act.sh <model_type: sd/sdxl> <w_bits: 4/8> <a_bits: 6/8> <path/to/cali_ckpt.pth_weight_only> <group size: 1/8/16>
```

4. Before inference the fully(weight and act.) quantized model, you should merge the two quantized checkpoints to one file. Use following command.
```
# first, copy the <cali_ckpt.pth_weight_only> file to the same folder in activation checkpoint(<cali_ckpt_activation_w?a?g?.pth>).
# Then, run the command.
python results/merge.py <path/to/cali_ckpt.pth_weight_only> <path/to/cali_ckpt_activation_w?a?g?.pth> 
```

5. Finally, you can run the fully quantized model. 
```
DIFFUSERS_REWRITE=<model_type: sd/sdxl> python src/inference_qmodel.py --cali_ckpt <path/to/cali_ckpt_activation_w?a?g?.pth_merged> --wq <w_bits: 4/8> --use_aq --aq <a_bits: 6/8> \
<--use_group> <--prompt "prompts"> <--fp16> <--t2i_log_quant> <--t2i_real_time> <--t2i_start_peak> <--time_aware_aqtizer>
```

### Acknoledgement
This project is mainly built upon [diffusers](https://github.com/huggingface/diffusers), [BK-SDM](https://github.com/Nota-NetsPresso/BK-SDM), [TFMQ-DM](https://github.com/ModelTC/TFMQ-DM) and [Q-Diffusion](https://github.com/Xiuyu-Li/q-diffusion). Thank you for sharing your codes.


### Bibtex
```
@article{ryu2025dgq,
  title={DGQ: Distribution-Aware Group Quantization for Text-to-Image Diffusion Models},
  author={Ryu, Hyogon and Park, NaHyeon and Shim, Hyunjung},
  journal={arXiv preprint arXiv:2501.04304},
  year={2025}
}
```
