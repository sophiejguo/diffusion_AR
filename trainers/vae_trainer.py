import time
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import VQVAE
from utils import dist
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor


class VAETrainer(object):
    def __init__(self, device, vae: VQVAE, vae_opt: AmpOptimizer):
        super().__init__()
        self.vae = vae
        self.vae_opt = vae_opt
        self.device = device
        self.recon_loss = nn.L1Loss(reduction="none")

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        loss_sum = 0.0
        stt = time.time()
        training = self.vae.training
        self.vae.eval()
        for inp_B3HW, _ in ld_val:
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            rec_B3HW, _, vq_loss = self.vae(inp_B3HW)
            loss = self.recon_loss(rec_B3HW, inp_B3HW).mean() + vq_loss.mean()
            loss_sum += loss * inp_B3HW.shape[0]
            tot += inp_B3HW.shape[0]
        self.vae.train(training)

        stats = torch.tensor([float(loss_sum), float(tot)], device=dist.get_device())
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= max(tot, 1)
        return stats[0].item(), tot, time.time() - stt

    def train_step(
        self,
        it: int,
        g_it: int,
        stepping: bool,
        metric_lg: MetricLogger,
        tb_lg: TensorboardLogger,
        inp_B3HW: FTen,
        label_B=None,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        self.vae.train(True)
        self.vae.requires_grad_(True)

        with self.vae_opt.amp_ctx:
            rec_B3HW, _, vq_loss = self.vae(inp_B3HW)
            loss = self.recon_loss(rec_B3HW, inp_B3HW).mean() + vq_loss.mean()

        grad_norm, scale_log2 = self.vae_opt.backward_clip_step(loss=loss, stepping=stepping)

        if it == 0 or it in metric_lg.log_iters:
            metric_lg.update(L=loss.item(), tnm=grad_norm.item())

        if dist.is_master() and (g_it == 0 or (g_it + 1) % 500 == 0):
            tb_lg.update(head="VAE_iter_loss", L=loss.item(), step=g_it)

        return grad_norm, scale_log2

    def state_dict(self):
        return {"vae": self.vae.state_dict(), "vae_opt": self.vae_opt.state_dict()}

    def load_state_dict(self, state, strict=True):
        self.vae.load_state_dict(state["vae"], strict=strict)
        self.vae_opt.load_state_dict(state["vae_opt"], strict=strict)

