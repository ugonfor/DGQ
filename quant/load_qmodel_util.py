from quant.quant_model import QuantModel, QMODE
from quant.calibration import load_cali_model
from quant.quant_block import QuantBasicTransformerBlock
import torch

def setup_pipe_to_calibrate(model_type, pipe):
    pipe.unet.float()
    if model_type == "sdxl":
        def forward(
            self, sample, timesteps, encoder_hidden_states, text_embeds, time_ids, **kwargs
        ):
            self.original_forward(sample, timesteps, encoder_hidden_states, 
                                {"text_embeds": text_embeds, "time_ids": time_ids}, **kwargs)
    
        setattr(pipe.unet, "original_forward", pipe.unet.forward)
        setattr(pipe.unet, "forward", forward.__get__(pipe.unet))
    else:
        pass

def setup_pipe_to_inference(model_type, qnn):
    if model_type == "sdxl":
        setattr(qnn.model, "forward", qnn.model.original_forward)
        delattr(qnn.model, "original_forward")

    else:
        pass

def get_qmodel(model_type, pipe, ckpt_path, 
               wq_params, 
               use_aq, aq_params, softmax_aq_params,
               use_group, num_inference_steps, time_aware_aqtizer,
               ):

    # quantization
    setup_pipe_to_calibrate(model_type, pipe)
    QuantModel_args = dict(model=pipe.unet,
                            wq_params=wq_params,
                            aq_params=aq_params,
                            softmax_aq_params=softmax_aq_params,
                            aq_mode=[QMODE.NORMAL.value, QMODE.QDIFF.value],
                            tib_recon=False,)
    qnn = QuantModel(**QuantModel_args).to('cuda').eval()
    
    # dummy_cali_data
    if model_type == 'sd':
        cali_data = (torch.randn(1, 4, 64, 64), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 768))
    elif model_type == "sdxl":
        cali_data = (torch.randn(1, 4, 128, 128), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 2048), torch.randn(1, 1280), torch.randn(1, 6))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # calibration args
    cali_model_args = dict(init_data=cali_data,
                           use_aq=use_aq,
                           path=ckpt_path,
                           time_aware_aqtizer=time_aware_aqtizer,
                            num_inference_steps=num_inference_steps,
                            use_group=use_group
                           )
    load_cali_model(qnn, **cali_model_args)
    qnn.disable_out_quantization()

    if use_aq:
        # softmax quantization should be performed on float32
        for name, module in qnn.named_modules():
            if isinstance(module, QuantBasicTransformerBlock):
                module.attn1.use_aq = True
                module.attn2.use_aq = True

    # revert to original forward
    setup_pipe_to_inference(model_type, qnn)
    return qnn