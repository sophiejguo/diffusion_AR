import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from utils import dist
from models import DiffusionVAR, VQVAE, VectorQuantizer2, DiffLoss
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class DiffusionVARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: DiffusionVAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
    ):
        super(DiffusionVARTrainer, self).__init__()
        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: DiffusionVAR = var_wo_ddp
        self.var_opt = var_opt

        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)

        self.diff_loss = var_wo_ddp.diffloss
        self.diffusion_batch_mul = var_wo_ddp.diffusion_batch_mul

        self.label_smooth = label_smooth
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean = 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B = label_B.shape[0]
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_embed_Bl = self.vae_local.quantize.embedding(torch.cat(gt_idx_Bl, dim=1)).cuda()
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            self.var_wo_ddp.forward
            x_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            mask = torch.ones_like(x_BLV).cuda()
            L_mean += self.forward_diffusion_loss(x_BLV, gt_embed_Bl, mask)
            tot += B
        self.var_wo_ddp.train(training)

        stats = L_mean.new_tensor([L_mean.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, _ = stats.tolist()
        return L_mean, tot, time.time() - stt

    def forward_diffusion_loss(self, z, target, mask):
        bsz, seq_len, _ = z.shape
        mask = torch.zeros(bsz, seq_len)
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diff_loss(z=z, target=target, mask=mask)
        return loss

    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1:
                self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog:
            prog_wp = 1
        if prog_si == len(self.patch_nums) - 1:
            prog_si = -1

        B = label_B.shape[0]
        self.var.require_backward_grad_sync = stepping

        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        gt_embed_Bl = self.vae_local.quantize.embedding(gt_BL).cuda()

        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            x_BLV = self.var(label_B, x_BLCv_wo_first_l)
            mask = torch.ones_like(x_BLV).cuda()
            loss = self.forward_diffusion_loss(x_BLV, gt_embed_Bl, mask)
            if prog_si >= 0:
                bg, ed = self.begin_ends[prog_si]
                assert x_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:
                lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()

        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        if it == 0 or it in metric_lg.log_iters:
            mask = torch.ones_like(x_BLV).cuda()
            Lmean = self.forward_diffusion_loss(x_BLV, gt_embed_Bl, mask)
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, tnm=grad_norm)

        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2
