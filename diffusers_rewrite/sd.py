# Obviously modified from the original source code
# https://github.com/huggingface/diffusers
# So has APACHE 2.0 license

# Author : Simo Ryu
# Edit for stable diffusion : Hyogon Ryu

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from collections import namedtuple
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin

import os, copy

# stable diffusion
class Timesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super(TimestepEmbedding, self).__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_shortcut=True):
        super(ResnetBlock2D, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.time_emb_proj = nn.Linear(1280, out_channels, bias=True)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if conv_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1
            )

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


class Attention(nn.Module):
    def __init__(
        self, inner_dim, cross_attention_dim=None, num_heads=None, dropout=0.0
    ):
        super(Attention, self).__init__()
        if num_heads is None:
            self.head_dim = 64
            self.num_heads = inner_dim // self.head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = inner_dim // num_heads
        self.scale = self.head_dim**-0.5
        if cross_attention_dim is None:
            cross_attention_dim = inner_dim
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim), nn.Dropout(dropout, inplace=False)]
        )

    def forward(self, hidden_states, encoder_hidden_states=None):
        q = self.to_q(hidden_states)
        k = (
            self.to_k(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_k(hidden_states)
        )
        v = (
            self.to_v(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_v(hidden_states)
        )
        b, t, c = q.size()

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)

        for layer in self.to_out:
            attn_output = layer(attn_output)

        return attn_output

    def Attention_forward(self, hidden_states, encoder_hidden_states=None):
        if hasattr(self, 'start_peak'):
            start_peak = self.start_peak
        else:
            start_peak = False

        q = self.to_q(hidden_states)
        k = (
            self.to_k(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_k(hidden_states)
        )
        v = (
            self.to_v(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_v(hidden_states)
        )
        b, t, c = q.size()

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_aq:
            q = self.aqtizer_q(q)
            if start_peak:
                start_token = k[..., 0:1, :]
                k = k[..., 1:, :]
                k = self.aqtizer_k(k)
                k = torch.cat([start_token, k], dim=-2)
            else:
                k = self.aqtizer_k(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        attn_weights = torch.softmax(scores, dim=-1)

        if self.use_aq:
            # if self.aqtizer_w.init: breakpoint()
            attn_weights = attn_weights.to(dtype=torch.float32)

            if start_peak:
                start_token = attn_weights[..., 0:1]
                attn_weights = attn_weights[..., 1:]
                attn_weights = self.aqtizer_w(attn_weights)
                attn_weights = torch.cat([start_token, attn_weights], dim=-1).to(dtype=v.dtype)
            else:
                attn_weights = self.aqtizer_w(attn_weights).to(dtype=v.dtype)

            v = self.aqtizer_v(v)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)

        for layer in self.to_out:
            attn_output = layer(attn_output)

        return attn_output


class GEGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(GEGLU, self).__init__()
        self.proj = nn.Linear(in_features, out_features * 2, bias=True)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * torch.nn.functional.gelu(x2)


class FeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeedForward, self).__init__()

        self.net = nn.ModuleList(
            [
                GEGLU(in_features, out_features * 4),
                nn.Dropout(p=0.0, inplace=False),
                nn.Linear(out_features * 4, out_features, bias=True),
            ]
        )

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super(BasicTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn1 = Attention(hidden_size, num_heads=8)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn2 = Attention(hidden_size, 768, num_heads=8)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.ff = FeedForward(hidden_size, hidden_size)

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


class Transformer2DModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(Transformer2DModel, self).__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-06, affine=True)
        self.proj_in = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(out_channels) for _ in range(n_layers)]
        )
        self.proj_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, hidden_states, encoder_hidden_states=None):
        batch, _, height, width = hidden_states.shape
        res = hidden_states
        hidden_states = self.norm(hidden_states)
        
        hidden_states = self.proj_in(hidden_states)
        inner_dim = hidden_states.shape[1] # if proj_in is Conv, inner_dim should be after the next line

        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)

        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        hidden_states = self.proj_out(hidden_states)

        return hidden_states + res


class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample2D, self).__init__()
        self.interp = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.interp(x)
        return self.conv(x)


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, has_downsamplers=True):
        super(DownBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, out_channels, conv_shortcut=False),
                ResnetBlock2D(out_channels, out_channels, conv_shortcut=False),
            ]
        )
        self.downsamplers = None
        if has_downsamplers:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels)])

    def forward(self, hidden_states, temb):
        output_states = []
        for module in self.resnets:
            hidden_states = module(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, has_downsamplers=True, has_shortcut=True):
        super(CrossAttnDownBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(out_channels, out_channels, n_layers),
                Transformer2DModel(out_channels, out_channels, n_layers),
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, out_channels, conv_shortcut=has_shortcut),
                ResnetBlock2D(out_channels, out_channels, conv_shortcut=False),
            ]
        )
        self.downsamplers = None
        if has_downsamplers:
            self.downsamplers = nn.ModuleList(
                [Downsample2D(out_channels, out_channels)]
            )

    def forward(self, hidden_states, temb, encoder_hidden_states=None):
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, n_layers, has_upsamplers=True):
        super(CrossAttnUpBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(out_channels, out_channels, n_layers),
                Transformer2DModel(out_channels, out_channels, n_layers),
                Transformer2DModel(out_channels, out_channels, n_layers),
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(prev_output_channel + out_channels, out_channels),
                ResnetBlock2D(2 * out_channels, out_channels),
                ResnetBlock2D(out_channels + in_channels, out_channels),
            ]
        )
        self.upsamplers = None
        if has_upsamplers:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])

    def forward(
        self, hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states=None
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel):
        super(UpBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(out_channels + prev_output_channel, out_channels),
                ResnetBlock2D(out_channels * 2, out_channels),
                ResnetBlock2D(out_channels + in_channels, out_channels),
            ]
        )
        self.upsamplers = nn.ModuleList(
            [
                Upsample2D(out_channels, out_channels),
            ]
        )

    def forward(self, hidden_states, res_hidden_states_tuple=None, temb=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)

        return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, in_features):
        super(UNetMidBlock2DCrossAttn, self).__init__()
        self.attentions = nn.ModuleList(
            [Transformer2DModel(in_features, in_features, n_layers=1)]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
            ]
        )

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNet2DConditionModel(ModelMixin, ConfigMixin):
    def __init__(self):
        super(UNet2DConditionModel, self).__init__()

        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        self.register_to_config(**{
            "in_channels": 4,
            "sample_size": 64,
            "time_cond_proj_dim": None,
        })

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, stride=1, padding=1)
        self.time_proj = Timesteps()
        self.time_embedding = TimestepEmbedding(in_features=320, out_features=1280)
        self.down_blocks = nn.ModuleList(
            [
                CrossAttnDownBlock2D(in_channels=320, out_channels=320, n_layers=1, has_shortcut=False),
                CrossAttnDownBlock2D(in_channels=320, out_channels=640, n_layers=1, has_shortcut=True),
                CrossAttnDownBlock2D(in_channels=640, out_channels=1280, n_layers=1, has_shortcut=True),
                DownBlock2D(in_channels=1280, out_channels=1280, has_downsamplers=False),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpBlock2D(in_channels=1280, out_channels=1280, prev_output_channel=1280),
                CrossAttnUpBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    prev_output_channel=1280,
                    n_layers=1,
                ),
                CrossAttnUpBlock2D(
                    in_channels=320,
                    out_channels=640,
                    prev_output_channel=1280,
                    n_layers=1,
                ),
                CrossAttnUpBlock2D(
                    in_channels=320,
                    out_channels=320,
                    prev_output_channel=640,
                    n_layers=1,
                    has_upsamplers=False,
                ),
            ]
        )
        self.mid_block = UNetMidBlock2DCrossAttn(1280)
        self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-05, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)

    def forward(
        self, sample, timesteps, encoder_hidden_states=None, **kwargs
    ):
        # Implement the forward pass through the model
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s7, s8, s9] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s10, s11] = self.down_blocks[3](
            sample,
            temb=emb,
        )

        # 4. mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states
        )

        # 5. up
        sample = self.up_blocks[0](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s9, s10, s11],
        )

        sample = self.up_blocks[1](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
        )

        sample = self.up_blocks[2](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
        )

        sample = self.up_blocks[3](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
            encoder_hidden_states=encoder_hidden_states,
        )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return [sample]
    

if __name__ == "__main__":
    unet = UNet2DConditionModel()

    import sys

    if len(sys.argv) < 2:
        print("Usage: python sd.py [flops|inference|statistics] [unet|decoder]")
        exit()
    
    if sys.argv[1] == 'flops' and len(sys.argv) == 3:
        
        if sys.argv[2] =='unet':
            from flops import count_ops_and_params
            example_inputs = {
                'sample': torch.randn(1, 4, 64, 64),
                'timesteps': torch.randint(0, 1000, (1,)),
                'encoder_hidden_states': torch.randn(1, 77, 768),
            }
            macs, nparams = count_ops_and_params(unet, example_inputs=example_inputs, layer_wise=False)
            print("#Params: {:.4f} M".format(nparams/1e6))
            print("#MACs: {:.4f} G".format(macs/1e9))
            print("#FLOPs: {:.4f} G".format(macs*2/1e9))
            print("#BOPs: {:.4f} T".format(macs*2*32*32/1e12)) # 4 bytes for float32

        elif sys.argv[2] == 'decoder':
            from flops import count_ops_and_params
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to('cuda')
            decoder = pipe.vae.decoder
            dummy_latents = torch.zeros(1, 4, 64, 64).to('cuda')
            
            macs, nparams = count_ops_and_params(decoder, example_inputs=dummy_latents, layer_wise=False)
            print("#Params: {:.4f} M".format(nparams/1e6))
            print("#MACs: {:.4f} G".format(macs/1e9))
            print("#FLOPs: {:.4f} G".format(macs*2/1e9))
            print("#BOPs: {:.4f} T".format(macs*2*32*32/1e12)) # 4 bytes for float32

        else:
            raise ValueError("Invalid argument")
        exit()

    elif sys.argv[1] == 'inference' and len(sys.argv) == 3:
        

        # add below code between 142~143 line
        '''bits = int(os.environ.get("BITS", 32))
        quantizer = os.environ.get("QUANTIZER", "log")
        if attn_weights.size(-1) == 77 and bits != 32:
            if quantizer == "log":
                print("Log quantization", "bits:", bits)
                attn_weights = attn_weights.to(dtype=torch.float32)
                start_token = attn_weights[..., 0:1]
                attn_weights = attn_weights[..., 1:]
                scale = attn_weights.max()
                attn_weights_q = torch.log2(attn_weights/scale) * -1
                attn_weights_q = torch.round(attn_weights_q)
                attn_weights_q = torch.clamp(attn_weights_q, 0, 2**bits - 1)
                attn_weights_q = attn_weights_q.to(dtype=torch.float32)
                attn_weights_q = torch.exp2(attn_weights_q * -1) * scale
                attn_weights = torch.cat([start_token, attn_weights_q], dim=-1).to(dtype=v.dtype)

            elif quantizer == "linear":
                # Linear quantization
                print("Linear quantization", "bits:", bits)
                attn_weights = attn_weights.to(dtype=torch.float32)
                scale, zero_point = (attn_weights.max() - attn_weights.min())/ (2**bits), attn_weights.min()
                attn_weights = (attn_weights - zero_point) / scale
                attn_weights = torch.round(attn_weights)
                attn_weights = torch.clamp(attn_weights, 0, 2**bits - 1)
                attn_weights = attn_weights * scale + zero_point
                attn_weights = attn_weights.to(dtype=v.dtype)
                print(attn_weights.max(), attn_weights.min())

            else:
                raise ValueError("Invalid quantizer")'''

        from diffusers import StableDiffusionPipeline
        from pytorch_lightning import seed_everything
        pipe = StableDiffusionPipeline.from_pretrained("pretrained/stable-diffusion-v1-4").to('cuda')
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

        for bits in ["4", "6", "8", "16"]:
            os.environ["BITS"] = bits
            os.environ["QUANTIZER"] = "linear"
            seed_everything(1)
            images = pipe([sys.argv[2]] * 4, num_inference_steps=25) # "a cute cat eating a banana"
            # images = pipe(["an astronaut riding a banana spaceship"] * 4, num_inference_steps=25)
            images[0][0].save(f"tmp1_{os.environ['BITS']}_linear.png")
            images[0][1].save(f"tmp2_{os.environ['BITS']}_linear.png")
            images[0][2].save(f"tmp3_{os.environ['BITS']}_linear.png")
            images[0][3].save(f"tmp4_{os.environ['BITS']}_linear.png")


            os.environ["QUANTIZER"] = "log"
            seed_everything(1)
            images = pipe([sys.argv[2]] * 4, num_inference_steps=25) # "a cute cat eating a banana"
            # images = pipe(["an astronaut riding a banana spaceship"] * 4, num_inference_steps=25)
            images[0][0].save(f"tmp1_{os.environ['BITS']}_log.png")
            images[0][1].save(f"tmp2_{os.environ['BITS']}_log.png")
            images[0][2].save(f"tmp3_{os.environ['BITS']}_log.png")
            images[0][3].save(f"tmp4_{os.environ['BITS']}_log.png")

    elif sys.argv[1] == 'statistics':
        
        # add below code between 142~143 line
        '''import glob
        os.makedirs("./std_max_statistics", exist_ok=True)
        num = len(glob.glob("./std_max_statistics/self_*.csv"))

        # max self, cross std
        if attn_weights.size(-1) == 77:
            with open(f"./std_max_statistics/cross_{num-1}.csv", "a") as f:
                _max = attn_weights[..., 1:].max().item()
                _min = attn_weights[..., 1:].min().item()
                _mean = attn_weights[..., 1:].mean().item()
                f.write(f'{_max},{_min},{_mean}\n')

        else:
            with open(f"./std_max_statistics/self_{num-1}.csv", "a") as f:
                _max = attn_weights.max().item()
                _min = attn_weights.min().item()
                _mean = attn_weights.mean().item()
                f.write(f'{_max},{_min},{_mean}\n')'''


        # load partiprompts
        from src.utils import get_file_list_from_tsv
        file_list = get_file_list_from_tsv("./data/PartiPrompts/PartiPrompts.tsv")
        prompt_list = [row[1] for row in file_list]

        from diffusers import StableDiffusionPipeline
        from pytorch_lightning import seed_everything
        pipe = StableDiffusionPipeline.from_pretrained("pretrained/stable-diffusion-v1-4").to('cuda')
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

        for i in range(len(prompt_list)):
            # make empty files
            prompt = prompt_list[i]

            open(f"./std_max_statistics/cross_{i}.csv", "w").write("")
            open(f"./std_max_statistics/self_{i}.csv", "w").write("")
            
            pipe(prompt, num_inference_steps=25)


    else:
        raise ValueError("Invalid argument")    
    