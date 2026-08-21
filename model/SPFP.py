import random

import torch
from torch import nn
from torch.nn import functional as F


class RandomConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, std=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.std = std
        self.padding = kernel_size // 2

    def forward(self, x):
        weight = torch.randn(
            self.in_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
            device=x.device,
            dtype=x.dtype,
        ) * self.std
        return F.conv2d(x, weight, padding=self.padding)


class MaskDilator(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, mask):
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        mask = mask.float().unsqueeze(1)
        mask = F.max_pool2d(mask, self.kernel_size, stride=1, padding=self.padding)
        return mask.squeeze(1)


class SPFP(nn.Module):
    def __init__(self, feat_dim=256, random_std=0.1, prob=1.0, mask_size=7, apply_in_eval=False):
        super().__init__()
        self.random_conv = RandomConv(feat_dim, 3, random_std)
        self.mask_dilator = MaskDilator(mask_size)
        self.prob = prob
        self.apply_in_eval = apply_in_eval
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.force_cpu_fft = False

    def _use_perturb(self):
        return self.training or self.apply_in_eval

    def forward(self, feature, mask):
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        if not self._use_perturb():
            return feature

        if random.random() < self.prob:
            feature = self.foreground_perturb(feature, mask)
        if random.random() < self.prob:
            feature = self.global_perturb(feature)
        return feature

    def _fft_fuse(self, src, style):
        alpha = torch.sigmoid(self.alpha)

        def _op(a, b, alpha_t):
            freq1 = torch.fft.fftshift(torch.fft.fft2(a))
            phase = torch.angle(freq1)
            amp1 = torch.abs(freq1)

            freq2 = torch.fft.fftshift(torch.fft.fft2(b))
            amp2 = torch.abs(freq2)

            alpha_local = alpha_t.to(device=amp1.device, dtype=amp1.dtype)
            amp_new = alpha_local * amp2 + (1 - alpha_local) * amp1
            fusion = torch.polar(amp_new, phase)
            return torch.fft.ifft2(torch.fft.ifftshift(fusion)).real

        src_fp32 = src.float().contiguous()
        style_fp32 = style.float().contiguous()

        run_on_cpu = self.force_cpu_fft and src_fp32.is_cuda
        if run_on_cpu:
            src_work = src_fp32.cpu()
            style_work = style_fp32.cpu()
            alpha_work = alpha.cpu()
        else:
            src_work = src_fp32
            style_work = style_fp32
            alpha_work = alpha

        try:
            out = _op(src_work, style_work, alpha_work)
        except RuntimeError as e:
            if src_fp32.is_cuda and "cufft" in str(e).lower():
                # Fallback for unstable CUDA/cuFFT stacks.
                self.force_cpu_fft = True
                out = _op(src_fp32.cpu(), style_fp32.cpu(), alpha.cpu()).to(src_fp32.device)
            else:
                raise

        return out.to(dtype=src.dtype, device=src.device)

    def foreground_perturb(self, feature, mask):
        b, _, h, w = feature.shape
        mask = F.interpolate(mask.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)
        mask = self.mask_dilator(mask)

        output = []
        for i in range(b):
            feat = feature[i]
            cur_mask = mask[i]
            if cur_mask.sum() < h * w * 0.01:
                output.append(feat.unsqueeze(0))
                continue

            coords = torch.nonzero(cur_mask > 0, as_tuple=False)
            y_min = int(coords[:, 0].min().item())
            x_min = int(coords[:, 1].min().item())
            y_max = int(coords[:, 0].max().item())
            x_max = int(coords[:, 1].max().item())
            patch = feat[:, y_min: y_max + 1, x_min: x_max + 1]
            ph, pw = patch.shape[-2:]
            if ph < 2 or pw < 2:
                output.append(feat.unsqueeze(0))
                continue

            rh = random.randint(max(1, ph // 4), ph)
            rw = random.randint(max(1, pw // 4), pw)
            sy = random.randint(0, ph - rh)
            sx = random.randint(0, pw - rw)

            sub_patch = patch[:, sy: sy + rh, sx: sx + rw]
            sub_patch = F.interpolate(
                sub_patch.unsqueeze(0),
                size=patch.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            patch_new = self._fft_fuse(patch.unsqueeze(0), sub_patch.unsqueeze(0))

            new_feat = feat.clone()
            new_feat[:, y_min: y_max + 1, x_min: x_max + 1] = patch_new.squeeze(0)
            output.append(new_feat.unsqueeze(0))

        return torch.cat(output, dim=0)

    def global_perturb(self, feature):
        random_feature = self.random_conv(feature)
        return self._fft_fuse(feature, random_feature)


class FEA_SPFP(nn.Module):
    def __init__(self, reduce_dim=256, random_std=0.1, prob=1.0, mask_size=7, apply_in_eval=False):
        super().__init__()
        self.spfp = SPFP(
            feat_dim=reduce_dim,
            random_std=random_std,
            prob=prob,
            mask_size=mask_size,
            apply_in_eval=apply_in_eval,
        )
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask.float(), size=x.shape[-2:], mode="nearest")
        mask = (mask > 0.5).float()

        perturbed = self.spfp(x, mask)
        beta = torch.sigmoid(self.beta)
        fused = beta * perturbed + (1 - beta) * x

        foreground = fused * mask
        proto = self.pool(foreground)
        return proto.expand_as(x)
