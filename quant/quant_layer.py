import torch.nn as nn
import torch
from enum import Enum
from typing import List, Union
import torch.nn.functional as F
import logging
import numpy as np
import os
logger = logging.getLogger(__name__)

COUNT_QLAYERS=0

# -------- quantization utils -------- #
class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def minmax(x: torch.Tensor,
            symmetric: bool = False,
            level: int = 256,
            always_zero: bool = False
            ) -> [torch.Tensor, torch.Tensor]:
    x_min, x_max = min(x.min().item(), 0), max(x.max().item(), 0)
    delta = torch.tensor(float(x_max - x_min) / (level - 1))
    if symmetric:
        x_min, x_max = -max(abs(x_min), x_max), max(abs(x_min), x_max)
        delta = torch.tensor(float(x_max - x_min) / (level - 2))
    if always_zero:
        delta = torch.tensor(float(x_max) / (level - 1))
    if delta < 1e-8:
        delta = 1e-8
    if type(x_min) == float: x_min = torch.tensor(x_min)
    zero_point = torch.round(-x_min / delta) if not (symmetric or always_zero) else 0
    return torch.tensor(delta).type_as(x), zero_point


def logminmax(x: torch.Tensor,
            symmetric: bool = False,
            level: int = 256,
            always_zero: bool = False
            ) -> [torch.Tensor, torch.Tensor]:
    x_clone = x.clone().detach().to(torch.float16) # for memory efficiency
    delta = x_clone.max()
    best_score = 1e+10
    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        new_delta = i

        x_int = torch.round( -1 * (x_clone / new_delta).log2() )
        x_q = torch.clamp(x_int, 0, level - 1)
        x_dq = new_delta * 2 ** x_q

        score = lp_loss(x_clone, x_dq, p=2, reduction=REDUCTION.ALL)
        if score < best_score:
            best_score = score
            delta = new_delta
    del x_clone
    return torch.tensor(delta).type_as(x)



def mse(x: torch.Tensor,
        symmetric: bool = False,
        level: int = 256,
        always_zero: bool = False
        ) -> [torch.Tensor, torch.Tensor]:
    x_min, x_max = x.min().item(), x.max().item()
    delta, zero_point = None, None
    s = 1e+10
    for i in range(80):
        new_min = x_min * (1. - (i * 0.01))
        new_max = x_max * (1. - (i * 0.01))
        new_delta = torch.tensor(float(new_max - new_min) / (level - 1))
        if symmetric: 
            new_min, new_max = -max(abs(new_min), new_max), max(abs(new_min), new_max)
            new_delta = (new_max - new_min) / (level - 2)
        if always_zero:
            new_delta = torch.tensor(float(new_max) / (level - 1))
        new_zero_point = torch.round(-new_min / new_delta) if not (symmetric or always_zero) else 0
        NB, PB = -level // 2 if symmetric and not always_zero else 0,\
              level // 2 - 1 if symmetric and not always_zero else level - 1
        x_q = torch.clamp(torch.round(x / new_delta) + new_zero_point, NB, PB)
        x_dq = new_delta * (x_q - new_zero_point)
        new_s = lp_loss(x_dq, x, p=2.4, reduction=REDUCTION.ALL)
        if new_s < s:
            s = new_s
            delta, zero_point = new_delta, new_zero_point 
    return delta, zero_point


def kl(x: torch.Tensor,
       symmetric: bool = False,
       level: int = 256,
       always_zero: bool = False
       ) -> [torch.Tensor, torch.Tensor]:

    def to_hist_with_orig_bins(targ_hist, targ_bins, orig_hist, orig_bins):
        targ_v = 0.0
        targ_i = 0
        targ_bin = targ_bins[0]
        ret_hist = np.zeros_like(orig_hist)

        for i, orig_bin in enumerate(orig_bins[:-1]):
            if targ_bin <= orig_bin:
                if targ_i < len(targ_bins) - 1:
                    targ_v = targ_hist[targ_i]
                    targ_i += 1
                    targ_bin = targ_bins[targ_i]
                else:
                    targ_v = 0.0
                    targ_bin = orig_bin.max() + 1.0

            ret_hist[i] = targ_v
        return ret_hist

    min_kl = 1e5
    res_clip_ratio = 1.0
    np_x = x.clone().detach().cpu().numpy()
    ref_hist, ref_bins = np.histogram(np_x, bins=level, density=True)
    sumd = np.sum(np.diff(ref_bins))
    smooth_ref_hist = (ref_hist + 1e-5) / (1.0 + sumd * 1e-5)
    for clip_ratio in np.linspace(0.5, 1.0, 50):
        clip_range = [np.min(np_x) * clip_ratio, np.max(np_x) * clip_ratio]
        q_hist, q_bins = np.histogram(np.clip(np_x, clip_range[0], clip_range[1]), bins=level, density=True)
        c_q_hist = to_hist_with_orig_bins(q_hist, q_bins, ref_hist, ref_bins)
        c_q_hist = (c_q_hist + 1e-5) / (1.0 + sumd * 1e-5)
        kl_c_q = np.sum(smooth_ref_hist * np.log(smooth_ref_hist / c_q_hist))
        if kl_c_q < min_kl:
            min_kl = kl_c_q
            res_clip_ratio = clip_ratio
    x_min, x_max = np.min(np_x) * res_clip_ratio, np.max(np_x) * res_clip_ratio
    x_clone = torch.where(x < x_min, x_min, x.clone().detach())
    x_clone = torch.where(x > x_max, x_max, x_clone)
    return minmax(x_clone, symmetric, level, always_zero)


def hist(x: torch.Tensor,
        symmetric: bool = False,
        level: int = 256,
        always_zero: bool = False
        ) -> [torch.Tensor, torch.Tensor]:
    x_min, x_max = None, None
    np_x = x.clone().detach().cpu().numpy()
    data_max = max(-np.min(np_x), np.max(np_x))
    hist, _ = np.histogram(np_x, bins=level, range=(0, data_max), density=True)
    accum = 0
    threshold = 0.9996
    hist = hist.astype(np.float32) / hist.sum()
    for i in range(len(hist)):
        accum += hist[i]
        if accum >= threshold:
            clip_value = (i + 0.5) * (data_max / level)
            x_min, x_max = max(-clip_value, np.min(np_x)), min(clip_value, np.max(np_x))
            break
    x_clone = torch.where(x < x_min, x_min, x.clone().detach())
    x_clone = torch.where(x > x_max, x_max, x_clone)
    return minmax(x_clone, symmetric, level, always_zero)

def omse(x: torch.Tensor,
        symmetric: bool = False,
        level: int = 256,
        always_zero: bool = False
        ) -> [torch.Tensor, torch.Tensor]:
    x_min, x_max = x.min().item(), x.max().item()
    delta, zero_point = None, None
    s = 1e+10
    for i in range(80):
        xrange = x_max - x_min
        x_min = 0
        x_max = xrange * (1. - (i * 0.01))
        tmp_delta = torch.tensor(float(x_max - x_min) / (level - 1))
        for j in range(level):
            tmp_zero_point = j
            x_q = torch.clamp(torch.round(x / tmp_delta) + tmp_zero_point, 0, level - 1)
            x_dq = tmp_delta * (x_q - tmp_zero_point)
            new_s = lp_loss(x_dq, x, p=2.4, reduction=REDUCTION.ALL)
            if new_s < s:
                s = new_s
                delta = tmp_delta
                zero_point = tmp_zero_point

    return delta, zero_point

class Scaler(Enum):
    MINMAX = minmax
    MSE = mse
    KL = kl
    HIST = hist
    OMSE = omse
    LOGMINMAX = logminmax


REDUCTION = Enum('REDUCTION', ('NONE', 'ALL'))


def lp_loss(pred: torch.Tensor, 
            tgt: torch.Tensor, 
            p: int = 2., 
            reduction: REDUCTION = REDUCTION.NONE
            ) -> torch.Tensor:
    if reduction == REDUCTION.NONE:
        return (pred - tgt).abs().pow(p).sum(1).mean()
    elif reduction == REDUCTION.ALL:
        return (pred - tgt).abs().pow(p).mean()
    else:
        raise NotImplementedError


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    
    def __init__(self, 
                 bits: int = 8,
                 symmetric: bool = False,
                 channel_wise: bool = False,
                 scaler: Scaler = Scaler.MINMAX,
                 leaf_param: bool = False,
                 always_zero: bool = False, # for softmax
                 quant_emb: bool = False
                 ) -> None:
        super().__init__()
        self.level = 2 ** bits
        self.symmetric = symmetric
        self.channel_wise = channel_wise
        self.scaler = scaler
        self.leaf_param = leaf_param
        if self.leaf_param:
            self.x_min, self.x_max = None, None
        self.running_stat = False
        self.always_zero = always_zero
        self.delta = None
        self.zero_point = None
        self.init = False
        self.quant_emb = quant_emb

        global COUNT_QLAYERS
        COUNT_QLAYERS += 1

        self.group_num = -1
        self.min_max_per_in_channel = []
        self.min_max_per_out_channel = []

    def _init_quantization_param(self, 
                                 x: torch.Tensor, 
                                 channel_wise: bool = False
                                 ) -> [torch.Tensor, torch.Tensor]:
        if channel_wise:
            N = x.shape[0]
            x_clone = x.clone().detach()
            x_max = x_clone.abs()
            for _ in range(len(x.shape) - 1):
                x_max = x_max.max(dim=-1)[0]
            delta, zero_point = x_max.clone(), x_max.clone()
            for c in range(N):
                delta[c], zero_point[c] = self._init_quantization_param(x_clone[c], channel_wise=False)
            D = {4: (-1, 1, 1, 1), 3: (-1, 1, 1), 2: (-1, 1)}
            delta = delta.view(D[len(x.shape)]) 
            zero_point = zero_point.view(D[len(x.shape)])
        else:
            if self.leaf_param:
                self.x_min, self.x_max = x.data.min(), x.data.max()
            delta, zero_point = self.scaler(x, self.symmetric, self.level, self.always_zero)
        return delta, zero_point
    
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        if not self.init:
            self.delta, self.zero_point = self._init_quantization_param(x, self.channel_wise)
            if self.leaf_param:
                self.delta = nn.Parameter(self.delta) 
            self.init = True
            global COUNT_QLAYERS
            COUNT_QLAYERS -= 1
            
            if COUNT_QLAYERS % 5 == 0:
                logger.info(f'Left QuantLayer : {COUNT_QLAYERS}')


        if self.running_stat:
           self.act_momentum_update(x)
           
        if 1 < self.group_num:
            if len(x.shape) <= 2: pass
            else: self.record_min_max_ema(x)
        elif self.group_num != -1:
            self.act_momentum_update(x)

        NB, PB = -self.level // 2 if self.symmetric and not self.always_zero else 0, \
            self.level // 2 - 1 if self.symmetric and not self.always_zero else self.level - 1
        x_q = torch.clamp(ste_round(x / self.delta) + self.zero_point, NB, PB)
        x_dq = self.delta * (x_q - self.zero_point)
        return x_dq
    
    def record_min_max_ema(self, x: torch.Tensor,
                           act_range_momentum: float = 0.95
    ) -> None:
        
        x_clone = x.data.clone().detach()

        if len(x.shape) == 3: # Linear
            self.min_max_per_in_channel.append((x_clone.min(dim=0)[0].min(dim=0)[0], x_clone.max(dim=0)[0].max(dim=0)[0]))
            self.min_max_per_out_channel.append((x_clone.min(dim=0)[0].min(dim=1)[0], x_clone.max(dim=0)[0].max(dim=1)[0]))

        elif len(x.shape) == 4: # MultiheadAttention
            self.min_max_per_in_channel.append((x_clone.min(dim=0)[0].min(dim=0)[0].min(dim=0)[0], x_clone.max(dim=0)[0].max(dim=0)[0].max(dim=0)[0]))
            self.min_max_per_out_channel.append((x_clone.min(dim=0)[0].min(dim=0)[0].min(dim=-1)[0], x_clone.max(dim=0)[0].max(dim=0)[0].max(dim=-1)[0]))
    
    def done_group_num(self, 
                       group_num,
                       mode) -> None:
        if self.min_max_per_in_channel == []:
            self.group_num = -1
            return
        
        # mean value vs minmax value
        self.min_max_per_in_channel = [x[0] for x in self.min_max_per_in_channel], [x[1] for x in self.min_max_per_in_channel]
        self.min_max_per_out_channel = [x[0] for x in self.min_max_per_out_channel], [x[1] for x in self.min_max_per_out_channel]
        
        # ## mean
        # self.min_max_per_in_channel = torch.stack(self.min_max_per_in_channel[0]).mean(dim=0), torch.stack(self.min_max_per_in_channel[1]).mean(dim=0)
        # self.min_max_per_out_channel = torch.stack(self.min_max_per_out_channel[0]).mean(dim=0), torch.stack(self.min_max_per_out_channel[1]).mean(dim=0)

        ## minmax: minmax is much better than mean
        self.min_max_per_in_channel = torch.stack(self.min_max_per_in_channel[0]).min(dim=0)[0], torch.stack(self.min_max_per_in_channel[1]).max(dim=0)[0]
        self.min_max_per_out_channel = torch.stack(self.min_max_per_out_channel[0]).min(dim=0)[0], torch.stack(self.min_max_per_out_channel[1]).max(dim=0)[0]

        print(self.min_max_per_in_channel[0].shape, self.min_max_per_out_channel[0].shape)

        # K-Means Based Clustering
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt
        import glob

        # choose in_channel_wise or out_channel_wise
        # Extract data
        in_channel_min = self.min_max_per_in_channel[0].cpu().numpy().flatten()
        in_channel_max = self.min_max_per_in_channel[1].cpu().numpy().flatten()
        out_channel_min = self.min_max_per_out_channel[0].cpu().numpy().flatten()
        out_channel_max = self.min_max_per_out_channel[1].cpu().numpy().flatten()

        in_channel_spread_degree = in_channel_max.max() - in_channel_max.min() + in_channel_min.max() - in_channel_min.min()
        out_channel_spread_degree = out_channel_max.max() - out_channel_max.min() + out_channel_min.max() - out_channel_min.min()


        if in_channel_spread_degree > out_channel_spread_degree or os.environ.get('IN_CHANNEL_WISE', False):
            print('in_channel_wise')
            in_channel_wise = True

            channel_data = np.column_stack((in_channel_min, in_channel_max))
            kmeans_output = KMeans(n_clusters=group_num, random_state=0).fit(channel_data)
            channel_labels = kmeans_output.labels_

        else:
            print('out_channel_wise')
            in_channel_wise = False

            channel_data = np.column_stack((out_channel_min, out_channel_max))
            kmeans_output = KMeans(n_clusters=group_num, random_state=0).fit(channel_data)
            channel_labels = kmeans_output.labels_

        # Research Question!
        # # mean value vs minmax value
        if mode == 'mean':
            center = kmeans_output.cluster_centers_
            # maximum, minimum value for each cluster
        elif mode == 'minmax':
            minmax_cluster = []
            for i in range(group_num):
                cluster = channel_data[channel_labels == i]
                # print(f'Cluster {i} : {cluster.min()} ~ {cluster.max()}')
                try:
                    minmax_cluster.append([cluster.min(), cluster.max()])
                except:
                    minmax_cluster.append([0., 1.])
            minmax_cluster = np.array(minmax_cluster)

            center = minmax_cluster
        else:
            raise NotImplementedError

        # Update delta, zero_point
        assert len(self.min_max_per_in_channel[0].shape) == 1
        assert len(self.min_max_per_out_channel[0].shape) == 1
        if in_channel_wise:
            delta = self.min_max_per_in_channel[0].clone().detach()
            zero_point = self.min_max_per_in_channel[0].clone().detach()
            
            delta = delta.view(1, 1, -1)
            zero_point = zero_point.view(1, 1, -1)
        else:
            delta = self.min_max_per_out_channel[0].clone().detach()
            zero_point = self.min_max_per_out_channel[0].clone().detach()

            delta = delta.view(1, -1, 1)
            zero_point = zero_point.view(1, -1, 1)


        for i in range(group_num):
            if in_channel_wise:
                tmp_delta = torch.tensor((center[i, 1] - center[i, 0]) / (self.level - 1))
                if tmp_delta < 1e-8:
                    tmp_delta = 1e-8
                delta[0, 0, channel_labels == i] = tmp_delta
                zero_point[0, 0, channel_labels == i] = torch.round(-center[i, 0] / tmp_delta)
                
            else:
                tmp_delta = torch.tensor((center[i, 1] - center[i, 0]) / (self.level - 1))
                if tmp_delta < 1e-8:
                    tmp_delta = torch.tensor(1e-8)
                delta[0, channel_labels == i, 0] = tmp_delta
                zero_point[0, channel_labels == i, 0] = torch.round(-center[i, 0] / tmp_delta)

        delta = delta.to(self.min_max_per_in_channel[0].device)
        zero_point = zero_point.to(self.min_max_per_in_channel[0].device)
        self.delta.data = delta
        self.zero_point.data = zero_point
        
        # reset
        self.group_num = -1
        self.min_max_per_in_channel = []
        self.min_max_per_out_channel = []
        return delta, zero_point

    def act_momentum_update(self,
                            x: torch.Tensor,
                            act_range_momentum: float = 0.95
                            ) -> None:
        assert self.init
        assert self.leaf_param
        x_min = x.data.min()
        x_max = x.data.max()
        self.x_min = self.x_min * act_range_momentum + x_min * (1. - act_range_momentum)
        self.x_max = self.x_max * act_range_momentum + x_max * (1. - act_range_momentum)
        x_clone = torch.where(x < self.x_min, self.x_min, x.clone().detach())
        x_clone = torch.where(x > self.x_max, self.x_max, x_clone)
        x_clone[..., 0] = self.x_min
        x_clone[..., 1] = self.x_max
        delta, self.zero_point = Scaler.MINMAX(x_clone, self.symmetric, self.level, self.always_zero)
        self.delta = torch.nn.Parameter(delta)

    def bitwidth_refactor(self, 
                          bits: int = 8
                          ) -> None:
        self.level = 2 ** bits

    def extra_repr(self) -> str:
        s = 'level={level}, symmetric={symmetric}, channel_wise={channel_wise}, scaler={scaler.__name__}, leaf_param={leaf_param}, group_num={group_num}'
        return s.format(**self.__dict__)
    
    def half(self):
        super().half()
        if self.delta is not None:
            self.delta = self.delta.half()
        if self.zero_point is not None:
            self.zero_point = self.zero_point.half()
        return self
    
    def float(self):
        super().float()
        if self.delta is not None:
            self.delta = self.delta.float()
        if self.zero_point is not None:
            self.zero_point = self.zero_point.float()
        return self
    


QMODE = Enum('QMODE', ('QDIFF', 'NORMAL', 'PTQD'))

def pseudo_conv2d(input, conv_weight, conv_bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Mimics 2D convolution using matrix multiplication.
    
    Parameters:
    - input: input tensor of shape (batch_size, in_channels, height, width)
    - conv_weight: convolutional weights of shape (out_channels, in_channels // groups, kernel_height, kernel_width)
    - conv_bias: optional bias term of shape (out_channels)
    - stride: stride of the convolution
    - padding: implicit zero paddings on both sides of the input
    - dilation: spacing between kernel elements
    - groups: split input into groups, in_channels should be divisible by the number of groups
    
    Returns:
    - output_matmul: the result of the convolution as a matrix multiplication
    """
    
    # Unfold the input to get sliding windows
    kernel_height, kernel_width = conv_weight.shape[2], conv_weight.shape[3]
    
    # Apply padding to the input if specified
    input_unfolded = F.unfold(input, kernel_size=(kernel_height, kernel_width),
                              dilation=dilation, padding=padding, stride=stride)
    
    # Reshape the convolutional weights for matrix multiplication
    batch_size = input.shape[0]
    out_channels = conv_weight.shape[0]
    
    # Reshape conv_weight from (out_channels, in_channels // groups, kernel_height, kernel_width)
    # to (out_channels, in_channels * kernel_height * kernel_width // groups)
    conv_weight_reshaped = conv_weight.view(out_channels, -1)
    
    # Matrix multiplication: (batch_size, in_channels * kernel_height * kernel_width, output_height * output_width)
    # @ (out_channels, in_channels * kernel_height * kernel_width)
    # Result: (batch_size, out_channels, output_height * output_width)
    output_matmul = conv_weight_reshaped @ input_unfolded
    
    # Reshape output to the expected shape (batch_size, out_channels, output_height, output_width)
    output_height = (input.shape[2] + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    output_width = (input.shape[3] + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    output_matmul = output_matmul.view(batch_size, out_channels, output_height, output_width)
    
    # Add bias if provided
    if conv_bias is not None:
        # Reshape conv_bias to (1, out_channels, 1, 1) and add to the output
        output_matmul += conv_bias.view(1, out_channels, 1, 1)
    
    return output_matmul

def input_unfolded_pseudo_conv2d(input_unfolded, input_shape,
                                 conv_weight, conv_bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Mimics 2D convolution using matrix multiplication.
    
    Parameters:
    - input: input tensor of shape (batch_size, in_channels, height, width)
    - conv_weight: convolutional weights of shape (out_channels, in_channels // groups, kernel_height, kernel_width)
    - conv_bias: optional bias term of shape (out_channels)
    - stride: stride of the convolution
    - padding: implicit zero paddings on both sides of the input
    - dilation: spacing between kernel elements
    - groups: split input into groups, in_channels should be divisible by the number of groups
    
    Returns:
    - output_matmul: the result of the convolution as a matrix multiplication
    """
    
    # Unfold the input to get sliding windows
    kernel_height, kernel_width = conv_weight.shape[2], conv_weight.shape[3]
    
    # # Apply padding to the input if specified
    # input_unfolded = F.unfold(input, kernel_size=(kernel_height, kernel_width),
    #                           dilation=dilation, padding=padding, stride=stride)
    
    # Reshape the convolutional weights for matrix multiplication
    batch_size = input_shape[0]
    out_channels = conv_weight.shape[0]
    
    # Reshape conv_weight from (out_channels, in_channels // groups, kernel_height, kernel_width)
    # to (out_channels, in_channels * kernel_height * kernel_width // groups)
    conv_weight_reshaped = conv_weight.view(out_channels, -1)
    
    # Matrix multiplication: (batch_size, in_channels * kernel_height * kernel_width, output_height * output_width)
    # @ (out_channels, in_channels * kernel_height * kernel_width)
    # Result: (batch_size, out_channels, output_height * output_width)
    output_matmul = conv_weight_reshaped @ input_unfolded
    
    # Reshape output to the expected shape (batch_size, out_channels, output_height, output_width)
    output_height = (input_shape[2] + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    output_width = (input_shape[3] + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    output_matmul = output_matmul.view(batch_size, out_channels, output_height, output_width)
    
    # Add bias if provided
    if conv_bias is not None:
        # Reshape conv_bias to (1, out_channels, 1, 1) and add to the output
        output_matmul += conv_bias.view(1, out_channels, 1, 1)
    
    return output_matmul


class QuantLayer(nn.Module):

    QMAP = {
        nn.Conv2d: F.conv2d,
        nn.Linear: F.linear,
    }

    def __init__(self,
                 layer: Union[nn.Conv2d, nn.Linear, nn.Conv1d],
                 wq_params: dict = {},
                 aq_params: dict = {}, 
                 disable_aq: bool = False,
                 aq_mode: List[int] = [QMODE.QDIFF.value],
                 quant_emb: bool = False
                 ) -> None:
        super().__init__()
        self.wq_params = wq_params
        self.aq_params = aq_params
        self.fwd_kwargs = {}
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):
            self.fwd_kwargs = dict(
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups
            )
        self.kwd_func = self.QMAP[type(layer)]
        self.w = layer.weight
        self.original_w = self.w.data.clone()
        self.b = None
        self.original_b = None
        if layer.bias is not None:
            self.b = layer.bias
            self.original_b = self.b.data.clone()
        self.use_wq = False
        self.use_aq = False
        self.disable_aq = disable_aq
        self.aq_mode = aq_mode
        self.quant_emb = quant_emb
        self.wq_params['quant_emb'] = quant_emb
        self.wqtizer = UniformAffineQuantizer(**self.wq_params)
        self.aqtizer = UniformAffineQuantizer(**self.aq_params)
        self.split = 0
        self.act_func = StraightThrough()
        self.ignore_recon = False
        self.extra_repr = layer.extra_repr

        self.use_group_num = False
    
    def forward(self, 
                x: torch.Tensor,
                split: int = 0
                ) -> torch.Tensor:
        if self.use_group_num and self.kwd_func == F.conv2d:
            # unfold the input to get sliding windows
            input_shape = x.shape
            kernel_height, kernel_width = self.w.shape[2], self.w.shape[3]
            input_unfolded = F.unfold(x, kernel_size=(kernel_height, kernel_width),
                                        dilation=self.fwd_kwargs['dilation'][0], 
                                        padding=self.fwd_kwargs['padding'][0], 
                                        stride=self.fwd_kwargs['stride'][0])
            x = input_unfolded
            
        if self.use_aq and not self.disable_aq:
            x = self.aqtizer(x)
        if self.use_wq: 
            w = self.wqtizer(self.w)
            b = self.b
        else:
            w = self.original_w
            b = self.original_b
        w = w.to(x.device)
        if type(b) == torch.Tensor:
            b = b.to(x.device)

        if self.use_group_num and self.kwd_func == F.conv2d:
            x = input_unfolded_pseudo_conv2d(x, input_shape, w, b,
                                                stride=self.fwd_kwargs['stride'][0], 
                                                padding=self.fwd_kwargs['padding'][0], 
                                                dilation=self.fwd_kwargs['dilation'][0],
                                                groups=self.fwd_kwargs['groups'])
        else:
            x = self.kwd_func(x, w, b, **self.fwd_kwargs)
        x = self.act_func(x)
        return x
    
    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        self.use_wq = use_wq if not self.ignore_recon else False
        self.use_aq = use_aq if not self.ignore_recon else False


    def set_running_stat(self,
                         running_stat: bool
                         ) -> None:
        self.aqtizer.running_stat = running_stat

    def set_group_num(self,
                         group_num: int = 1
                         ) -> None:
        self.aqtizer.group_num = group_num
        self.use_group_num = True
    
    def done_group_num(self,
                        group_num,
                        mode
                        ) -> None:
        self.aqtizer.done_group_num(group_num, mode)

    def half(self):
        super().half()
        if self.original_w is not None:
            self.original_w = self.original_w.half()
        if self.original_b is not None:
            self.original_b = self.original_b.half()
        return self
    
    def float(self):
        super().float()
        if self.original_w is not None:
            self.original_w = self.original_w.float()
        if self.original_b is not None:
            self.original_b = self.original_b.float()
        return self