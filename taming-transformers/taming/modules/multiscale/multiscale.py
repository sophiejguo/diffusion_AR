from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F
torch.autograd.set_detect_anomaly(True)




# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['MultiScale',]


class MultiScale(nn.Module):
    def __init__(
        self, 
        Cvae, 
        beta: float = 0.25, 
        v_patch_nums=None, 
        share_resi_ratio=0, 
        resi_ratio=0.5, 
        default_resi_counts=0):
        
        super().__init__()
        self.Cvae = Cvae
        self.beta = beta
        self.v_patch_nums = v_patch_nums

        self.resi_ratio = resi_ratio
        if share_resi_ratio == 0:   # non-shared: \phi_{1 to K} for K scales
            self.resi_ratio = PhiNonShared([(Phi(Cvae, resi_ratio) if abs(resi_ratio) > 1e-6 else nn.Identity()) for _ in range(default_resi_counts or len(self.v_patch_nums))])
        elif share_resi_ratio == 1: # fully shared: only a single \phi for K scales
            self.resi_ratio = PhiShared(Phi(Cvae, resi_ratio) if abs(resi_ratio) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_resi_ratio} for K scales
            self.resi_ratio = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, resi_ratio) if abs(resi_ratio) > 1e-6 else nn.Identity()) for _ in range(share_resi_ratio)]))

    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        with torch.cuda.amp.autocast(enabled=False):
            mean_loss: torch.Tensor = 0.0
            # vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
        
                rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area')if (si != SN-1) else f_rest
        
                # calc loss
                h_BChw = F.interpolate(rest_NC, size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else rest_NC.contiguous()
                h_BChw = self.resi_ratio[si/(SN-1)](h_BChw)
                # f_hat.add_(h_BChw)
                # f_rest.sub_(h_BChw)
                # f_hat.add(h_BChw)
                # f_rest.sub(h_BChw)

                f_hat = f_hat + h_BChw
                f_rest = f_rest - h_BChw
            
                
                # mean_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
                # mean_loss += F.mse_loss(f_hat, f_BChw).mul(self.beta) + F.mse_loss(f_hat, f_no_grad)
                mean_loss += F.mse_loss(f_hat, f_BChw)
            mean_loss *= 1. / SN
            # f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
            # f_hat = (f_hat - f_no_grad).add(f_BChw)
            f_hat = f_BChw + (f_hat - f_BChw.detach())  # More stable correction
     
        return f_hat, mean_loss
    # ===================== `forward` is only used in VAE training =====================



    def f_to_fhat(self, f_BChw: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.Tensor]:
        """
        Multi-scale residual feature decomposition without quantization.
        Returns a list of reconstructed features at each scale.

        Based off of f_to_idxBl_or_fhat, but without quantization.
        """
        dtype = f_BChw.dtype
        if dtype != torch.float32:
            f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_scales: List[torch.Tensor] = []

        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # from small to large

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):  # from small to large
            # Downsample residual to current scale
            if si != SN - 1:
                rest_k = F.interpolate(f_rest, size=(ph, pw), mode='area')
            else:
                rest_k = f_rest

            # Upsample back to original resolution
            if si != SN - 1:
                h_BChw = F.interpolate(rest_k, size=(H, W), mode='bicubic').contiguous()
            else:
                h_BChw = rest_k.contiguous()

            # Apply residual transformation
            h_BChw = self.resi_ratio[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)

            # Store the reconstruction at this scale
            f_hat_scales.append(f_hat.clone())

        return f_hat_scales

class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'