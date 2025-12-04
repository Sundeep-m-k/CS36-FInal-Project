# models/backbone_swin.py
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

import timm

from util.misc import NestedTensor
from .position_encoding import build_position_encoding
from .backbone import Joiner


class SwinBackbone(nn.Module):
    """
    Swin-T backbone that mimics the interface of BackboneBase:
    - forward takes NestedTensor or plain tensor
    - returns a dict[name] -> NestedTensor / Tensor
    - has .num_channels like the ResNet backbone
    """
    def __init__(self, train_backbone: bool, return_interm_layers: bool, d_model: int = 256):
        super().__init__()

        # Choose which feature levels to output
        if return_interm_layers:
            out_indices = (0, 1, 2, 3)
        else:
            out_indices = (3,)  # last stage only

        # timm Swin model
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
        )

        # Freeze if needed (following train_backbone flag)
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Channels of raw Swin features
        swin_channels = self.backbone.feature_info.channels()

        # Project each feature map to d_model channels
        self.proj = nn.ModuleList(
            [nn.Conv2d(c, d_model, kernel_size=1) for c in swin_channels]
        )

        # For consistency with ResNet backbone, this is the dim fed to the transformer
        self.num_channels = d_model

        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list):
        """
        Supports both NestedTensor and plain torch.Tensor, like BackboneBase.
        """
        is_nested = isinstance(tensor_list, NestedTensor)
        if is_nested:
            xs_in = tensor_list.tensors  # (B, C, H, W)
            mask_in = tensor_list.mask   # (B, H, W)
        else:
            xs_in = tensor_list
            mask_in = None

        feats = self.backbone(xs_in)  # list of feature maps from Swin
        out: Dict[str, NestedTensor] = OrderedDict()

        for i, x in enumerate(feats):
            # Project to d_model channels
            x_proj = self.proj[i](x)

            if is_nested:
                # Resize mask to this feature map size
                assert mask_in is not None
                m = F.interpolate(mask_in[None].float(), size=x_proj.shape[-2:]).to(torch.bool)[0]
                out[str(i)] = NestedTensor(x_proj, m)
            else:
                out[str(i)] = x_proj

        return out


def build_swin_backbone(args):
    """
    Build Swin backbone + position encoding + Joiner,
    similar to build_backbone in backbone.py
    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks

    # hidden_dim is the DETR transformer dimension (typically 256)
    d_model = getattr(args, "hidden_dim", 256)

    backbone = SwinBackbone(train_backbone, return_interm_layers, d_model=d_model)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
