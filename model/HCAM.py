import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        hidden = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def _group_norm(channels, max_groups=16):
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ScaleAwareModule(nn.Module):
    def __init__(self, channels, dilation_rates=(1, 3, 5), se_reduction=16, gn_groups=16):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilation_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                    _group_norm(channels, gn_groups),
                    nn.ReLU(inplace=True),
                    SELayer(channels, reduction=se_reduction),
                    nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                    _group_norm(channels, gn_groups),
                    nn.ReLU(inplace=True),
                )
            )
        self.attn_conv = nn.Conv2d(channels, len(dilation_rates), kernel_size=1, bias=True)

    def forward(self, x):
        outs = [br(x) for br in self.branches]
        sum_feats = outs[0]
        for o in outs[1:]:
            sum_feats = sum_feats + o
        logits = self.attn_conv(sum_feats)
        attn = F.softmax(logits, dim=1)

        fused = 0
        for i, o in enumerate(outs):
            fused = fused + o * attn[:, i : i + 1, :, :]
        return fused, outs, attn


class SCPP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        se_reduction=16,
        dilation_rates=(1, 3, 5),
        gn_groups=16,
        res_gamma_init=0.1,
        global_alpha_init=-2.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.res_proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.res_gamma = nn.Parameter(torch.tensor(float(res_gamma_init)))
        self.global_alpha = nn.Parameter(torch.tensor(float(global_alpha_init)))

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            _group_norm(in_channels, gn_groups),
            nn.ReLU(inplace=True),
        )
        self.scale_module = ScaleAwareModule(
            in_channels,
            dilation_rates=dilation_rates,
            se_reduction=se_reduction,
            gn_groups=gn_groups,
        )
        self.global_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=True),
            nn.ReLU(inplace=True),
        )
        self.global_gate = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=True),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False),
            _group_norm(out_channels, gn_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        local_feat = self.conv1x1(x)
        scale_fused, _, _ = self.scale_module(x)
        gap = F.adaptive_avg_pool2d(x, 1).view(b, c)
        global_vec = self.global_fc(gap) * self.global_gate(gap)
        global_weight = torch.sigmoid(self.global_alpha)
        global_feat = (global_weight * global_vec).view(b, c, 1, 1).expand(-1, -1, h, w)
        cat_feat = torch.cat([scale_fused, local_feat, global_feat], dim=1)
        delta = self.out_conv(cat_feat)
        return self.res_proj(x) + self.res_gamma * delta



