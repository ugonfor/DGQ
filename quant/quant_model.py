from typing import List
import torch.nn as nn
import torch

from collections import namedtuple
from quant.quant_block import QuantBasicTransformerBlock, QuantResnetBlock2D, QuantTemporalInformationBlock, b2qb, BaseQuantBlock
from quant.quant_layer import QMODE, QuantLayer, StraightThrough

from quant.adaptive_rounding import AdaRoundQuantizer
from quant.quant_layer import UniformAffineQuantizer
from quant.quant_block import T2ILogQuantizer
class CFG:
    in_channels = 0
    sample_size = 0
    time_cond_proj_dim = 0
    addition_time_embed_dim = 0

class QuantModel(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 wq_params: dict = {},
                 aq_params: dict = {},
                 softmax_aq_params: dict = {},
                 cali: bool = True,
                 tib_recon: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.model = model

        # for diffusers pipeline
        self.config = CFG()
        
        self.config.in_channels = model.config.in_channels
        self.config.sample_size = model.config.sample_size
        self.config.time_cond_proj_dim = model.config.time_cond_proj_dim
        if hasattr(model.config, 'addition_time_embed_dim'):
            self.config.addition_time_embed_dim = model.config.addition_time_embed_dim
        self.tib_recon = tib_recon
        

        self.B = b2qb()
        self.quant_module(self.model, wq_params, aq_params, 
                          aq_mode=kwargs.get("aq_mode", [QMODE.NORMAL.value]), 
                          prev_name=None)
        self.quant_block(self.model, wq_params, aq_params,
                         softmax_aq_params)
        if cali and tib_recon:
            self.get_tib(self.model, wq_params, aq_params)
            

    def get_tib(self,
                    module: nn.Module,
                    wq_params: dict = {},
                    aq_params: dict = {},
                    ) -> QuantTemporalInformationBlock:
        for name, child in module.named_children():
            if name == 'time_embedding':
                self.tib = QuantTemporalInformationBlock(child, aq_params)
            elif isinstance(child, QuantResnetBlock2D):
                self.tib.add_emb_layer(child.time_emb_proj)
            else:
                self.get_tib(child, wq_params, aq_params)


    def quant_module(self,
                     module: nn.Module,
                     wq_params: dict = {},
                     aq_params: dict = {},
                     aq_mode: List[int] = [QMODE.NORMAL.value],
                     prev_name: str = None,
                     ) -> None:
        for name, child in module.named_children():
            if isinstance(child, tuple(QuantLayer.QMAP.keys())):
                if name in ['time_embedding', 'time_proj', 'time_emb_proj'] and self.tib_recon: 
                    # for keep time embedding while block reconstruction
                    # refer to TFMQ-DM
                    setattr(module, name, QuantLayer(child, wq_params, aq_params, aq_mode=aq_mode, quant_emb=True))
                else:
                    setattr(module, name, QuantLayer(child, wq_params, aq_params, aq_mode=aq_mode))
                    
            elif isinstance(child, StraightThrough):
                continue
            else:
                self.quant_module(child, wq_params, aq_params, aq_mode=aq_mode, prev_name=name)

    def quant_block(self,
                    module: nn.Module,
                    wq_params: dict = {},
                    aq_params: dict = {},
                    softmax_aq_params: dict = {},
                    ) -> None:
        for name, child in module.named_children():
            if child.__class__.__name__ in self.B:
                if self.B[child.__class__.__name__] in [QuantBasicTransformerBlock]:
                    setattr(module, name, self.B[child.__class__.__name__](child, aq_params, softmax_aq_params))
                elif self.B[child.__class__.__name__] in [QuantResnetBlock2D]:
                    setattr(module, name, self.B[child.__class__.__name__](child, aq_params))
                else:
                    raise NotImplementedError
                
            else:
                self.quant_block(child, wq_params, aq_params, softmax_aq_params)

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.model.modules():
            if isinstance(m, (BaseQuantBlock, QuantLayer)):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)

    def forward(
        self, sample, timesteps, encoder_hidden_states, *args, **kwargs
        ):
        return self.model(sample, timesteps, encoder_hidden_states, *args, **kwargs)

    def disable_out_quantization(self) -> None:
        # conv_in, conv_out are too much sensitive to quantization
        self.model.conv_in.use_wq = False
        self.model.conv_in.disable_aq = True

        self.model.conv_out.use_wq = False
        self.model.conv_out.disable_aq = True


    def synchorize_activation_statistics(self):
        import linklink.dist_helper as dist
        for module in self.modules():
            if isinstance(module, QuantLayer):
                if module.aqtizer.delta is not None:
                    dist.allaverage(module.aqtizer.delta)


    def set_group_num(self,
                         group_num: int = 1
                         ) -> None:
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                m.set_group_num(group_num)
            if isinstance(m, QuantBasicTransformerBlock):
                m.attn1.aqtizer_q.group_num = group_num
                m.attn1.aqtizer_k.group_num = group_num
                m.attn1.aqtizer_v.group_num = group_num
                
                m.attn2.aqtizer_q.group_num = group_num
                m.attn2.aqtizer_k.group_num = group_num
                m.attn2.aqtizer_v.group_num = group_num
                
                
    def done_group_num(self,
                       group_num,
                       mode
                       ) -> None:
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                m.done_group_num(group_num, mode=mode)
            if isinstance(m, QuantBasicTransformerBlock):
                m.attn1.aqtizer_q.done_group_num(group_num, mode=mode)
                m.attn1.aqtizer_k.done_group_num(group_num, mode=mode)
                m.attn1.aqtizer_v.done_group_num(group_num, mode=mode)

                m.attn2.aqtizer_q.done_group_num(group_num, mode=mode)
                m.attn2.aqtizer_k.done_group_num(group_num, mode=mode)
                m.attn2.aqtizer_v.done_group_num(group_num, mode=mode)
                
    def set_running_stat(self,
                         running_stat: bool = False
                         ) -> None:
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                m.attn1.aqtizer_q.running_stat = running_stat
                m.attn1.aqtizer_k.running_stat = running_stat
                m.attn1.aqtizer_v.running_stat = running_stat
                m.attn1.aqtizer_w.running_stat = running_stat
                m.attn2.aqtizer_q.running_stat = running_stat
                m.attn2.aqtizer_k.running_stat = running_stat
                m.attn2.aqtizer_v.running_stat = running_stat
                m.attn2.aqtizer_w.running_stat = running_stat
            elif isinstance(m, QuantLayer):
                m.set_running_stat(running_stat)

    def half(self):
        print("QuantModel half")
        super().half()
        for m in self.model.modules():
            if isinstance(m, (AdaRoundQuantizer, UniformAffineQuantizer)):
                m.half()
            if isinstance(m, QuantLayer):
                m.half()
        return self
    
    def float(self):
        print("QuantModel float")
        super().float()
        for m in self.model.modules():
            if isinstance(m, (AdaRoundQuantizer, UniformAffineQuantizer)):
                m.float()
            if isinstance(m, QuantLayer):
                m.float()
        return self
        
    @property
    def device(self):
        return next(self.parameters()).device