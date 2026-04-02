"""
Diffusion VAE implementation following the algorithms in the mid-term report.
This implementation avoids quantization in the VAE, as opposed to the original VQVAE.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_float32_matmul_precision('medium')


from .basic_vae import Decoder, Encoder
from .multiscale import MultiScale


class MultiScaleAE(nn.Module):
    def __init__(
        self, z_channels=32, ch=128, dropout=0.0,
        beta=0.25,          # commitment loss weight
        conv_ks=3,          # convolution kernel size
        share_resi_ratio=4, # 0: non-shared, 1: fully shared, 0 < x < 1: partially shared, 4 \phi layers for K scales: partially-shared \phi
        resi_ratio=0.5,     # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        default_resi_counts=0, # if is 0: automatically set to len(v_patch_nums)
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        test_mode=True,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.Cvae = z_channels

        # Encoder and decoder configuration
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
            using_sa=True, using_mid_sa=True,
        )
        
        # Encoder (E in the algorithm)
        self.encoder = Encoder(double_z=False, **ddconfig)
        # Decoder (D in the algorithm)
        self.decoder = Decoder(**ddconfig)

        self.multiscale: MultiScale = MultiScale(Cvae=z_channels, 
                                                 beta=beta, 
                                                 v_patch_nums=v_patch_nums,
                                                 share_resi_ratio=share_resi_ratio,
                                                 resi_ratio=resi_ratio,
                                                 default_resi_counts=default_resi_counts)
        
        # Convolution layers for encoding/decoding  
        self.pre_conv = nn.Conv2d(z_channels, z_channels, conv_ks, stride=1, padding=conv_ks//2)
        self.post_conv = nn.Conv2d(z_channels, z_channels, conv_ks, stride=1, padding=conv_ks//2)
        
        # Extra convolution layers for each scale (ϕ_k functions in the paper)
        # self.phi = nn.ModuleList([
        #     nn.Conv2d(z_channels, z_channels, kernel_size=3, stride=1, padding=1)
        #     for _ in range(steps_K)
        # ])
        
        # Test mode switch
        self.test_mode = test_mode
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    




    # ===================== `forward` is only used in VAE training =====================
    def forward(self, inp):   # -> rec_B3HW, idx_N, loss
        MultiScale.forward
        f_hat, mean_loss = self.multiscale(self.pre_conv(self.encoder(inp)))
        return self.decoder(self.post_quant_conv(f_hat)), mean_loss
    # ===================== `forward` is only used in VAE training =====================
    
    # def img_to_latents(self, im):
    #     """
    #     Convert an image to multi-scale latent representations
        
    #     Args:
    #         im: Input image
            
    #     Returns:
    #         R: List of latent representations at each scale
    #     """
    #     return self.encode(im)
    
    # def latents_to_img(self, latents):
    #     """
    #     Convert multi-scale latent representations to an image
        
    #     Args:
    #         latents: List of latent representations at each scale
            
    #     Returns:
    #         im_hat: Reconstructed image
    #     """
    #     return self.decode(latents)
    
    # def img_to_reconstructed_img(self, x, last_one=False):
    #     """
    #     Convert an image to a reconstructed image
        
    #     Args:
    #         x: Input image
    #         last_one: If True, return only the final reconstruction
    #                  If False, return list of reconstructions at each scale
            
    #     Returns:
    #         List[torch.Tensor] or torch.Tensor: Reconstructed image(s)
    #     """
    #     R = self.encode(x)
        
    #     if last_one:
    #         return self.decode(R)
    #     else:
    #         # For each scale, decode up to that scale
    #         recons = []
    #         for k in range(1, self.steps_K + 1):
    #             partial_R = R[:k]
    #             recon = self.decode(partial_R)
    #             recons.append(recon)
    #         return recons
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign) 
    
