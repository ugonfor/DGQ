from typing import Dict, Tuple

from diffusers_rewrite import Timesteps, TimestepEmbedding
from diffusers_rewrite import Attention, ResnetBlock2D, BasicTransformerBlock

import torch
import torch as th
import torch.nn as nn
from types import MethodType
from quant.quant_layer import QuantLayer, UniformAffineQuantizer, StraightThrough, Scaler
from quant.quant_layer_text import T2ILogQuantizer


class BaseQuantBlock(nn.Module):

    def __init__(self,
                 aq_params: dict = {}
                 ) -> None:
        super().__init__()
        self.use_wq = False
        self.use_aq = False
        self.act_func = StraightThrough()
        self.ignore_recon = False

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)
            if isinstance(m, Attention):
                m.use_aq = use_aq

class QuantTemporalInformationBlock(BaseQuantBlock):

    def __init__(self,
                 time_embedding: TimestepEmbedding,
                 aq_params: dict = {},
                 ) -> None:
        super().__init__(aq_params)
        self.time_proj = Timesteps()
        self.time_embedding = time_embedding
        self.emb_layers = nn.ModuleList()

    def add_emb_layer(self,
                      time_emb_proj: nn.Linear) -> None:
        self.emb_layers.append(
            time_emb_proj
        )

    def forward(self,
                x: th.Tensor,
                t: th.Tensor,
                y: th.Tensor = None
                ) -> Tuple[th.Tensor]:
        assert t is not None
        t_emb = self.time_proj(t).to(dtype=x.dtype)
        emb = self.time_embedding(t_emb)
        opts = []
        for layer in self.emb_layers:
            temb = nn.SiLU()(emb)
            temb = layer(temb)[:, :, None, None]
            opts.append(temb)
        return tuple(opts)

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)
        for emb_layer in self.emb_layers:
            for m in emb_layer.modules():
                if isinstance(m, QuantLayer):
                    m.set_quant_state(use_wq=use_wq, use_aq=use_aq)

class QuantResnetBlock2D(BaseQuantBlock):
    '''
    This block is for 1) block-wise reconstruction and 2) conv_shortcut
    '''
    def __init__(self,
            resnet: ResnetBlock2D,
            aq_params: dict = {}
        ) -> None:
        super().__init__(aq_params)
        
        self.norm1 = resnet.norm1
        self.conv1 = resnet.conv1
        self.time_emb_proj = resnet.time_emb_proj
        self.norm2 = resnet.norm2
        self.dropout = resnet.dropout
        self.conv2 = resnet.conv2
        self.nonlinearity = resnet.nonlinearity
        self.conv_shortcut = resnet.conv_shortcut

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor

class QuantBasicTransformerBlock(BaseQuantBlock):
    def __init__(
        self,
        tran: BasicTransformerBlock,
        aq_params: dict = {},
        softmax_aq_params: dict = {},
        ) -> None:
        super().__init__(aq_params)

        self.norm1 = tran.norm1
        self.attn1 = tran.attn1
        self.norm2 = tran.norm2
        self.attn2 = tran.attn2
        self.norm3 = tran.norm3
        self.ff = tran.ff

        self.attn1.aqtizer_q = UniformAffineQuantizer(**aq_params)
        self.attn1.aqtizer_k = UniformAffineQuantizer(**aq_params)
        self.attn1.aqtizer_v = UniformAffineQuantizer(**aq_params)

        self.attn2.aqtizer_q = UniformAffineQuantizer(**aq_params)
        self.attn2.aqtizer_k = UniformAffineQuantizer(**aq_params)
        self.attn2.aqtizer_v = UniformAffineQuantizer(**aq_params)

        aq_params_w = aq_params.copy()
        aq_params_w['bits'] = softmax_aq_params['softmax_a_bit']
        aq_params_w['symmetric'] = False
        aq_params_w['always_zero'] = True
        if softmax_aq_params['t2i_log_quant']:
            aq_params_w['real_time'] = softmax_aq_params['t2i_real_time']
            aq_params_w['log_max_1'] = softmax_aq_params['log_max_1']
            self.attn1.aqtizer_w = T2ILogQuantizer(**aq_params_w)
            self.attn2.aqtizer_w = T2ILogQuantizer(**aq_params_w)
        else:
            self.attn1.aqtizer_w = UniformAffineQuantizer(**aq_params_w)
            self.attn2.aqtizer_w = UniformAffineQuantizer(**aq_params_w)
        if softmax_aq_params['t2i_start_peak']:
            self.attn2.start_peak = True

        self.attn1.forward = self.attn1.Attention_forward # MethodType(Attention_forward, self.attn1)
        self.attn2.forward = self.attn2.Attention_forward # MethodType(Attention_forward, self.attn2)
        self.attn1.use_aq = False
        self.attn2.use_aq = False

    def forward(self, x, encoder_hidden_states=None):
        residual = x

        x = self.norm1(x)
        x = self.attn1(x)
        x = x + residual

        residual = x

        x = self.norm2(x)
        if encoder_hidden_states is not None:
            x = self.attn2(x, encoder_hidden_states=encoder_hidden_states)
        else:
            x = self.attn2(x)
        x = x + residual

        residual = x

        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual
        return x


def b2qb() -> Dict[nn.Module, BaseQuantBlock]:
    D = {
        ResnetBlock2D.__name__: QuantResnetBlock2D,
        BasicTransformerBlock.__name__: QuantBasicTransformerBlock,
    }
    return D
