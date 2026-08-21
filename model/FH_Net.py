import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models
from model.SPFP import FEA_SPFP
from model.HCAM import SCPP


def _upsample_concat(x, y, module):
    _, _, h, w = y.shape
    return module(torch.cat([F.interpolate(x, size=(h, w), mode="bilinear"), y], dim=1))


class CrossClassSimilarityGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
        )
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, fused_feat, query_feat, fg_proto, bg_proto):
        q = F.normalize(query_feat, dim=1)
        fg = F.normalize(fg_proto, dim=1)
        bg = F.normalize(bg_proto, dim=1)

        sim_fg = (q * fg).sum(dim=1, keepdim=True)
        sim_bg = (q * bg).sum(dim=1, keepdim=True)
        sim_delta = sim_fg - sim_bg

        spatial = torch.sigmoid(self.spatial_gate(torch.cat([sim_fg, sim_bg, sim_delta], dim=1)))
        channel = self.channel_gate(torch.cat([fg_proto, bg_proto], dim=1))
        guided = fused_feat * spatial * channel
        return self.refine(torch.cat([fused_feat, guided], dim=1))


class IdentityTwo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask = None):
        return x


class fhnet(nn.Module):
    def __init__(
        self,
        layers=50,
        classes=2,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        pretrained=True,
        shot=1,
        enhancer="spfp",
        spfp_random_std=0.1,
        spfp_prob=0.6,
        spfp_mask_size=3,
        spfp_apply_in_eval=False,
        dp_block="scpp",
        scpp_se_reduction=16,
        scpp_dilation_rates=(1, 3, 5),
        use_transformer_fusion=False,
        use_cross_class_guidance=False,
        pretrained_path=None,
    ):
        super(fhnet, self).__init__()
        from torch.nn import BatchNorm2d as BatchNorm

        self.criterion = criterion
        self.shot = shot
        self.enhancer = enhancer.lower()
        self.dp_block = dp_block.lower()
        self.use_transformer_fusion = use_transformer_fusion
        self.use_cross_class_guidance = use_cross_class_guidance
        reduce_dim = 256

        models.BatchNorm = BatchNorm
        if layers != 50:
            raise ValueError("FH-Net supports the ResNet-50 backbone used in the paper; received layers={}.".format(layers))
        print(">>>>>>>>> Using ResNet 50<<<<<<<<<")
        resnet = models.resnet50(pretrained=pretrained, pretrained_path=pretrained_path)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu1,
            resnet.conv2,
            resnet.bn2,
            resnet.relu2,
            resnet.conv3,
            resnet.bn3,
            resnet.relu3,
            resnet.maxpool,
        )
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(reduce_dim * 2 ** (3 - i), reduce_dim, kernel_size=1, padding=0, bias=False)
                )
                for i in range(4)
            ]
        )
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False))
                for _ in range(3)
            ]
        )
        if self.enhancer == "spfp":
            self.REF = nn.ModuleList(
                [
                    FEA_SPFP(
                        reduce_dim=reduce_dim,
                        random_std=spfp_random_std,
                        prob=spfp_prob,
                        mask_size=spfp_mask_size,
                        apply_in_eval=spfp_apply_in_eval,
                    )
                    for _ in range(4)
                ]
            )
        elif self.enhancer == "no":
            self.REF = nn.ModuleList([IdentityTwo() for _ in range(4)])
        else:
            raise ValueError("Unsupported enhancer '{}', expected 'spfp' or 'no'".format(enhancer))

        self.supervise_s = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(reduce_dim * 3, reduce_dim, kernel_size=1, padding=0, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(reduce_dim, classes, kernel_size=3, padding=1, bias=False),
                )
                for _ in range(4)
            ]
        )
        self.down_s = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False))
                for _ in range(4)
            ]
        )
        if self.dp_block == "scpp":
            self.DP = nn.ModuleList(
                [
                    SCPP(
                        in_channels=reduce_dim,
                        out_channels=reduce_dim,
                        se_reduction=scpp_se_reduction,
                        dilation_rates=scpp_dilation_rates,
                    )
                    for _ in range(4)
                ]
            )
        elif self.dp_block == "no":
            self.DP = nn.ModuleList([nn.Identity() for _ in range(4)])
        else:
            raise ValueError("Unsupported dp_block '{}', expected 'scpp' or 'no'".format(dp_block))
        if self.use_cross_class_guidance:
            self.cross_class_guidance = nn.ModuleList([CrossClassSimilarityGate(reduce_dim) for _ in range(4)])

        self.fuse_up = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.2),
                )
                for _ in range(3)
            ]
        )
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim + 8, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, 2, kernel_size=3, padding=1, bias=False),
        )
        self.supervise_q = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(reduce_dim, classes, kernel_size=3, padding=1, bias=False)) for _ in range(4)]
        )

        if self.use_transformer_fusion:
            self.scale_pos_embed = nn.Parameter(torch.zeros(1, 4, reduce_dim))
            self.unified_proto_token = nn.Parameter(torch.zeros(1, 1, reduce_dim))
            proto_encoder_layer = nn.TransformerEncoderLayer(
                d_model=reduce_dim,
                nhead=8,
                dim_feedforward=reduce_dim * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.prototype_transformer = nn.TransformerEncoder(proto_encoder_layer, num_layers=2)
            self.prototype_norm = nn.LayerNorm(reduce_dim)
            nn.init.normal_(self.scale_pos_embed, std=0.02)
            nn.init.normal_(self.unified_proto_token, std=0.02)

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD(
            [
                {"params": model.down.parameters()},
                {"params": model.smooth.parameters()},
                {"params": model.REF.parameters()},
                {"params": model.supervise_s.parameters()},
                {"params": model.down_s.parameters()},
                {"params": model.DP.parameters()},
                {"params": model.fuse_up.parameters()},
                {"params": model.cls.parameters()},
                {"params": model.supervise_q.parameters()},
            ],
            lr=LR,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        if model.use_transformer_fusion:
            optimizer.add_param_group({"params": model.prototype_transformer.parameters()})
            optimizer.add_param_group({"params": model.prototype_norm.parameters()})
            optimizer.add_param_group({"params": [model.scale_pos_embed, model.unified_proto_token]})
        if model.use_cross_class_guidance:
            optimizer.add_param_group({"params": model.cross_class_guidance.parameters()})
        return optimizer

    @staticmethod
    def _masked_average_pool(feat, mask):
        mask_sum = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return (feat * mask).sum(dim=(2, 3), keepdim=True) / mask_sum

    @staticmethod
    def _token_to_map(token, ref_feat):
        return token.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_feat.shape[-2], ref_feat.shape[-1])

    @staticmethod
    def _global_token(feat):
        return feat.mean(dim=(2, 3))

    @staticmethod
    def _build_scale_prototypes(proto_token_list):
        return torch.stack(proto_token_list, dim=0).mean(dim=0)

    def _build_unified_prototypes(self, proto_token_list):
        multi_scale_proto = self._build_scale_prototypes(proto_token_list)
        if not self.use_transformer_fusion:
            return multi_scale_proto
        proto_tokens = multi_scale_proto + self.scale_pos_embed
        unified_token = self.unified_proto_token.expand(proto_tokens.shape[0], -1, -1)
        proto_tokens = torch.cat([unified_token, proto_tokens], dim=1)
        proto_tokens = self.prototype_transformer(proto_tokens)
        proto_tokens = self.prototype_norm(proto_tokens)
        unified_proto = proto_tokens[:, :1, :]
        scale_proto = proto_tokens[:, 1:, :]
        return scale_proto + unified_proto

    def forward(
        self,
        x,
        s_x=None,
        s_y=None,
        y=None,
    ):
        b, _, h, w = x.shape
        if s_x is None:
            s_x = torch.zeros(b, self.shot, 3, h, w, device=x.device, dtype=x.dtype)
        if s_y is None:
            s_y = torch.zeros(b, self.shot, h, w, device=x.device, dtype=x.dtype)

        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)

        query_feat_4 = self.down[0](query_feat_4)
        query_feat_3 = self.down[1](query_feat_3)
        query_feat_2 = self.down[2](query_feat_2)
        query_feat_1 = self.down[3](query_feat_1)

        query_4 = query_feat_4
        query_3 = self.smooth[0](query_feat_3)
        query_2 = self.smooth[1](query_feat_2)
        query_1 = self.smooth[2](query_feat_1)

        proto_token_list = []
        bg_token_list = []
        aux_loss_s = x.new_tensor(0.0)

        for i in range(self.shot):
            supp_gt = (s_y[:, i, :, :] == 1).float().unsqueeze(1)

            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                supp_feat_4 = self.layer4(supp_feat_3)

            supp_4 = self.down[0](supp_feat_4)
            supp_3 = self.smooth[0](self.down[1](supp_feat_3))
            supp_2 = self.smooth[1](self.down[2](supp_feat_2))
            supp_1 = self.smooth[2](self.down[3](supp_feat_1))

            supp_gt_4 = F.interpolate(supp_gt, size=supp_4.shape[-2:], mode="bilinear", align_corners=True)
            supp_gt_3 = F.interpolate(supp_gt, size=supp_3.shape[-2:], mode="bilinear", align_corners=True)
            supp_gt_2 = F.interpolate(supp_gt, size=supp_2.shape[-2:], mode="bilinear", align_corners=True)
            supp_gt_1 = F.interpolate(supp_gt, size=supp_1.shape[-2:], mode="bilinear", align_corners=True)

            supp_pro_4 = self.REF[0](supp_4, supp_gt_4)
            supp_pro_3 = self.REF[1](supp_3, supp_gt_3)
            supp_pro_2 = self.REF[2](supp_2, supp_gt_2)
            supp_pro_1 = self.REF[3](supp_1, supp_gt_1)

            scale_proto_token = torch.stack(
                [
                    self._global_token(supp_pro_4),
                    self._global_token(supp_pro_3),
                    self._global_token(supp_pro_2),
                    self._global_token(supp_pro_1),
                ],
                dim=1,
            )
            proto_token_list.append(scale_proto_token)
            bg_scale_token = torch.stack(
                [
                    self._masked_average_pool(supp_4, 1 - supp_gt_4).squeeze(-1).squeeze(-1),
                    self._masked_average_pool(supp_3, 1 - supp_gt_3).squeeze(-1).squeeze(-1),
                    self._masked_average_pool(supp_2, 1 - supp_gt_2).squeeze(-1).squeeze(-1),
                    self._masked_average_pool(supp_1, 1 - supp_gt_1).squeeze(-1).squeeze(-1),
                ],
                dim=1,
            )
            bg_token_list.append(bg_scale_token)

            if self.training:
                out_4_s = self.supervise_s[0](torch.cat([supp_4, supp_4, supp_pro_4], dim=1))
                out_3_s = self.supervise_s[1](torch.cat([supp_3, supp_3, supp_pro_3], dim=1))
                out_2_s = self.supervise_s[2](torch.cat([supp_2, supp_2, supp_pro_2], dim=1))
                out_1_s = self.supervise_s[3](torch.cat([supp_1, supp_1, supp_pro_1], dim=1))
                aux_loss_s = aux_loss_s + self.criterion(out_4_s, supp_gt_4.squeeze(1).long())
                aux_loss_s = aux_loss_s + self.criterion(out_3_s, supp_gt_3.squeeze(1).long())
                aux_loss_s = aux_loss_s + self.criterion(out_2_s, supp_gt_2.squeeze(1).long())
                aux_loss_s = aux_loss_s + self.criterion(out_1_s, supp_gt_1.squeeze(1).long())

        unified_scale_proto = self._build_unified_prototypes(proto_token_list)
        supp_pro_4 = self._token_to_map(unified_scale_proto[:, 0, :], query_4)
        supp_pro_3 = self._token_to_map(unified_scale_proto[:, 1, :], query_3)
        supp_pro_2 = self._token_to_map(unified_scale_proto[:, 2, :], query_2)
        supp_pro_1 = self._token_to_map(unified_scale_proto[:, 3, :], query_1)
        unified_bg_proto = self._build_scale_prototypes(bg_token_list)
        supp_bg_4 = self._token_to_map(unified_bg_proto[:, 0, :], query_4)
        supp_bg_3 = self._token_to_map(unified_bg_proto[:, 1, :], query_3)
        supp_bg_2 = self._token_to_map(unified_bg_proto[:, 2, :], query_2)
        supp_bg_1 = self._token_to_map(unified_bg_proto[:, 3, :], query_1)

        query_supp_4 = self.down_s[0](torch.cat([query_4, supp_pro_4], dim=1))
        query_supp_3 = self.down_s[1](torch.cat([query_3, supp_pro_3], dim=1))
        query_supp_2 = self.down_s[2](torch.cat([query_2, supp_pro_2], dim=1))
        query_supp_1 = self.down_s[3](torch.cat([query_1, supp_pro_1], dim=1))

        fuse_4 = self.DP[0](query_supp_4)
        if self.use_cross_class_guidance:
            fuse_4 = self.cross_class_guidance[0](fuse_4, query_4, supp_pro_4, supp_bg_4)
        fuse_3 = self.DP[1](_upsample_concat(fuse_4, query_supp_3, self.fuse_up[0]))
        if self.use_cross_class_guidance:
            fuse_3 = self.cross_class_guidance[1](fuse_3, query_3, supp_pro_3, supp_bg_3)
        fuse_2 = self.DP[2](_upsample_concat(fuse_3, query_supp_2, self.fuse_up[1]))
        if self.use_cross_class_guidance:
            fuse_2 = self.cross_class_guidance[2](fuse_2, query_2, supp_pro_2, supp_bg_2)
        fuse_1 = self.DP[3](_upsample_concat(fuse_2, query_supp_1, self.fuse_up[2]))
        if self.use_cross_class_guidance:
            fuse_1 = self.cross_class_guidance[3](fuse_1, query_1, supp_pro_1, supp_bg_1)

        out_4_q = self.supervise_q[0](F.interpolate(fuse_4, size=(h, w), mode="bilinear", align_corners=True))
        out_3_q = self.supervise_q[1](F.interpolate(fuse_3, size=(h, w), mode="bilinear", align_corners=True))
        out_2_q = self.supervise_q[2](F.interpolate(fuse_2, size=(h, w), mode="bilinear", align_corners=True))
        out_1_q = self.supervise_q[3](F.interpolate(fuse_1, size=(h, w), mode="bilinear", align_corners=True))

        out_4_q_0 = F.interpolate(out_4_q, size=query_supp_1.shape[-2:], mode="bilinear", align_corners=True)
        out_3_q_0 = F.interpolate(out_3_q, size=query_supp_1.shape[-2:], mode="bilinear", align_corners=True)
        out_2_q_0 = F.interpolate(out_2_q, size=query_supp_1.shape[-2:], mode="bilinear", align_corners=True)
        out_1_q_0 = F.interpolate(out_1_q, size=query_supp_1.shape[-2:], mode="bilinear", align_corners=True)

        fuse_0 = torch.cat([out_1_q_0, out_2_q_0, out_3_q_0, out_4_q_0, fuse_1], dim=1)
        query_pred_mask = self.cls(F.interpolate(fuse_0, size=(h, w), mode="bilinear", align_corners=True))

        query_pred_mask_save = torch.argmax(query_pred_mask[0].squeeze(0).permute(1, 2, 0), axis=-1)
        query_pred_mask_save[query_pred_mask_save != 0] = 255
        query_pred_mask_save[query_pred_mask_save == 0] = 0

        if self.training:
            main_loss = self.criterion(query_pred_mask, y.long())
            aux_loss_s = aux_loss_s / max(self.shot, 1)
            aux_loss_q = self.criterion(out_4_q, y.long())
            aux_loss_q = aux_loss_q + self.criterion(out_3_q, y.long())
            aux_loss_q = aux_loss_q + self.criterion(out_2_q, y.long())
            aux_loss_q = aux_loss_q + self.criterion(out_1_q, y.long())
            return query_pred_mask.max(1)[1], main_loss, aux_loss_s + aux_loss_q

        return query_pred_mask, query_pred_mask_save
