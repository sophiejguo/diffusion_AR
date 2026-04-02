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
            # Use pre-quantization continuous residuals as diffusion targets
            gt_idx_Bl, gt_prequant_Bl = self.vae_local.img_to_idxBl_and_prequant(inp_B3HW)
            gt_prequant_BL = torch.cat(gt_prequant_Bl, dim=1)  # (B, L, Cvae)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            x_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            mask = torch.ones_like(x_BLV)
            L_mean += self.forward_diffusion_loss(x_BLV, gt_prequant_BL, mask)
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

        # Use pre-quantization continuous residuals as diffusion targets.
        # gt_prequant_Bl[k] has shape (B, pn_k*pn_k, Cvae) — the raw encoder residual at scale k
        # before the nearest-neighbour codebook lookup. This is the z_s described in Section 2.2.
        gt_idx_Bl, gt_prequant_Bl = self.vae_local.img_to_idxBl_and_prequant(inp_B3HW)
        gt_prequant_BL = torch.cat(gt_prequant_Bl, dim=1)  # (B, L, Cvae)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

        with self.var_opt.amp_ctx:
            x_BLV = self.var(label_B, x_BLCv_wo_first_l)
            mask = torch.ones_like(x_BLV)
            loss = self.forward_diffusion_loss(x_BLV, gt_prequant_BL, mask)
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
            with torch.no_grad():
                mask = torch.ones_like(x_BLV)
                Lmean = self.forward_diffusion_loss(x_BLV, gt_prequant_BL, mask)
            grad_norm = grad_norm.item()
            # Lt, Accm, Acct are not meaningful for diffusion loss; log as -1 for compatibility
            metric_lg.update(Lm=Lmean, Lt=-1, Accm=-1, Acct=-1, tnm=grad_norm)

        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }

    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k:
                continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[DiffusionVARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[DiffusionVARTrainer.load_state_dict] {k} unexpected:  {unexpected}')

        config: dict = state.pop('config', None)
        if config is not None:
            self.prog_it = config.get('prog_it', 0)
            self.last_prog_si = config.get('last_prog_si', -1)
            self.first_prog = config.get('first_prog', True)
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[DiffusionVARTrainer.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
