import torch.nn as nn
import torch
from enum import Enum
from typing import List, Union
import torch.nn.functional as F
import logging
import numpy as np
logger = logging.getLogger(__name__)

from quant.quant_layer import Scaler, StraightThrough, COUNT_QLAYERS, ste_round, lp_loss, REDUCTION

class T2ILogQuantizer(nn.Module):
    
    def __init__(self, 
                 bits: int = 8,
                 symmetric: bool = False,
                 channel_wise: bool = False,
                 scaler: Scaler = Scaler.MINMAX,
                 leaf_param: bool = False,
                 always_zero: bool = True, # for softmax
                 quant_emb: bool = False,
                 real_time: bool = False,
                 log_max_1: bool= False,
                 ) -> None:
        super().__init__()
        self.level = 2 ** bits
        self.symmetric = symmetric
        self.channel_wise = channel_wise
        self.scaler = scaler
        self.leaf_param = leaf_param
        if self.leaf_param:
            self.x_log_max = None
        self.running_stat = False

        self.always_zero = always_zero
        self.delta = None
        self.zero_point = None
        self.init = False
        self.quant_emb = quant_emb

        global COUNT_QLAYERS
        COUNT_QLAYERS += 1
        self.real_time = real_time
        self.NB, self.PB = -self.level // 2 if self.symmetric and not self.always_zero else 0, \
            self.level // 2 - 1 if self.symmetric and not self.always_zero else self.level - 1

        self.log_max_1=log_max_1

    def _init_quantization_param(self, 
                                 x: torch.Tensor, 
                                 ) -> [torch.Tensor, torch.Tensor]:
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e+10
        for pct in [0.999, 0.9999, 0.99999]: #
            try:
                new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                new_delta = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
                

            x_q = -1 * torch.log2(x_clone / new_delta)
            x_q = torch.round(x_q)
            x_q = torch.clamp(x_q, self.NB, self.PB)
            x_q = 2**(-1 * x_q)
            x_q = x_q * new_delta

            score = lp_loss(x_clone, x_q, p=2, reduction=REDUCTION.ALL)
            if score < best_score:
                best_score = score
                delta = new_delta

        return delta
    
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        if not self.init and not self.real_time:
            self.delta = self._init_quantization_param(x)
            if self.leaf_param:
                self.delta = nn.Parameter(self.delta) 
            self.init = True
            global COUNT_QLAYERS
            COUNT_QLAYERS -= 1
            logger.info(f'Left QuantLayer : {COUNT_QLAYERS}')

        if self.log_max_1:
            self.delta.data = torch.tensor(1.)
        
        if self.running_stat and not self.real_time:
            self.act_momentum_update(x)
        
        if self.real_time:
            delta = x.max()
        else:
            delta = self.delta
        
        x_q = -1 * torch.log2(x / delta)
        x_q = torch.round(x_q)
        x_q = torch.clamp(x_q, self.NB, self.PB)
        x_q = 2**(-1 * x_q)
        x_dq = x_q * delta
        
        return x_dq
    
    def act_momentum_update(self,
                            x: torch.Tensor,
                            act_range_momentum: float = 0.95
                            ) -> None:
        assert self.init
        assert self.leaf_param
        x_clone = x.clone().detach()
        delta = x_clone.max()
        self.delta.data = act_range_momentum * self.delta.data + (1 - act_range_momentum) * delta

    def bitwidth_refactor(self, 
                          bits: int = 8
                          ) -> None:
        self.level = 2 ** bits

    def extra_repr(self) -> str:
        s = 'level={level}, symmetric={symmetric}, channel_wise={channel_wise}, scaler={scaler.__name__}, leaf_param={leaf_param}'
        return s.format(**self.__dict__)
    
    def half(self):
        super().half()
        if self.delta is not None:
            self.delta = self.delta.half()
        return self
    
    def float(self):
        super().float()
        if self.delta is not None:
            self.delta = self.delta.float()
        return self
    

