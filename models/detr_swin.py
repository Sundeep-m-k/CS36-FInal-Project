# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model with Swin Transformer backbone
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .point_encoder import build_point_encoder
from .label_encoder import build_label_encoder
from .point_criterion import PointCriterion
from .position_encoding import build_position_encoding


# Simple wrapper for torchvision Swin to return features as list and position encodings
class TVSwinJoiner(torch.nn.Module):
    def __init__(self, tv_model, position_embedding):
        super().__init__()
        self.tv_model = tv_model
        self.position_embedding = position_embedding

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor with:
                - tensors: [B, 3, H, W]
                - mask:    [B, H, W] (True for padded pixels)

        Returns:
            out: list[NestedTensor]       # feature maps
            pos: list[Tensor]            # positional encodings
        """
        assert isinstance(tensor_list, NestedTensor)
        x = tensor_list.tensors  # [B, 3, H, W]

        # torchvision Swin exposes `features` module returning feature maps
        if hasattr(self.tv_model, "features"):
            feat = self.tv_model.features(x)
        elif hasattr(self.tv_model, "forward_features"):
            feat = self.tv_model.forward_features(x)
        else:
            # fallback to direct call (may include classifier) but we'll take intermediate
            feat = self.tv_model(x)

        # Ensure feat is a list of tensors
        if isinstance(feat, torch.Tensor):
            xs = [feat]
        elif isinstance(feat, (list, tuple)):
            xs = list(feat)
        else:
            # if dict, take values
            try:
                xs = list(feat.values())
            except Exception:
                raise RuntimeError("Unsupported Swin backbone output type")

        out = []
        pos = []
        m = tensor_list.mask
        assert m is not None

        for x in xs:
            # torchvision Swin is **channels-last**: [B, H, W, C]
            # DETR expects channels-first features: [B, C, H, W]
            if x.dim() == 4:
                # Heuristic: if second dim (would be C in NCHW) is much smaller than last dim,
                # we assume layout is NHWC and convert.
                # Example:
                #   NHWC: [B, 16, 16, 768] -> x.shape[1] = 16 < x.shape[-1] = 768  -> permute
                #   NCHW: [B, 768, 16, 16] -> x.shape[1] = 768 > x.shape[-1] = 16 -> keep
                if x.shape[1] < x.shape[-1]:
                    x = x.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]

            # Now x is [B, C, H, W]; resize mask to (H, W)
            mask = F.interpolate(
                m[None].float(), size=x.shape[-2:], mode="nearest"
            ).to(torch.bool)[0]

            nt = NestedTensor(x, mask)
            out.append(nt)
            pos.append(self.position_embedding(nt).to(x.dtype))

        return out, pos


class DETR(nn.Module):
    """ This is the DETR module with Swin backbone that performs object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        label_encoder=None,
        point_encoder=None,
    ):
        """Initializes the model.

        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, i.e., detection slots.
                         This is the maximal number of objects DETR can detect in a single image.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.label_encoder = label_encoder
        self.point_encoder = point_encoder
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor, points_supervision):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

           points_supervison:
               - "points"      , (N, 2)
               - "object_ids"  , (N,)
               - "labels"      , (N,)

           It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized box coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # features: list[NestedTensor], pos: list[positional_encodings]
        features, pos = self.backbone(samples)

        # Take last (highest-level) feature map
        src, mask = features[-1].decompose()
        assert mask is not None

        # Build query embeddings from points (per image)
        query_embed = self.point_encoder(
            points_supervision, self.backbone.position_embedding, self.label_encoder
        )
        bs = len(query_embed)

        # Transformer expects [B, C, H, W] for src
        hs = self.transformer(self.input_proj(src), mask, query_embed, pos[-1])

        depth = hs[0].size(0)

        outputs_class = []
        outputs_coord = []

        for idx in range(bs):
            cur_point_sup = points_supervision[idx]["points"]  # obj_num x 2

            # hs[idx]: depth x obj_num x hidden_dim
            outputs_class.append(self.class_embed(hs[idx]))  # depth x obj_num x (num_classes+1)

            o_coord = self.bbox_embed(hs[idx]).sigmoid() / 2
            o_coord[:, :, 0] = (-o_coord[:, :, 0] + cur_point_sup[None, :, 0]).clamp_(min=0.001)
            o_coord[:, :, 1] = (-o_coord[:, :, 1] + cur_point_sup[None, :, 1]).clamp_(min=0.001)
            o_coord[:, :, 2] = (o_coord[:, :, 2] + cur_point_sup[None, :, 0]).clamp_(max=0.999)
            o_coord[:, :, 3] = (o_coord[:, :, 3] + cur_point_sup[None, :, 1]).clamp_(max=0.999)
            o_coord = box_ops.box_xyxy_to_cxcywh(o_coord)
            outputs_coord.append(o_coord)  # depth x obj_num x 4

        outputs_class_depth = []
        outputs_coord_depth = []

        for dep_idx in range(depth):
            batched_cls = []
            batched_coord = []

            for idx in range(bs):
                batched_cls.append(outputs_class[idx][dep_idx])
                batched_coord.append(outputs_coord[idx][dep_idx])

            outputs_class_depth.append(batched_cls)
            outputs_coord_depth.append(batched_coord)

        gt_label = []
        for i in range(bs):
            gt_label.append(points_supervision[i]["labels"].unsqueeze(0))

        out = {"pred_boxes": outputs_coord_depth[-1], "gt_label": gt_label}

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_coord_depth)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_boxes": b} for b in outputs_coord[:-1]]  # except last


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each image of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_bbox = outputs["pred_boxes"]
        gt_label = outputs["gt_label"]

        assert target_sizes.shape[1] == 2
        bs = len(out_bbox)

        boxes = []

        for idx in range(bs):
            boxes.append(box_ops.box_cxcywh_to_xyxy(out_bbox[idx]))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        for idx in range(bs):
            boxes[idx] = boxes[idx] * scale_fct[idx][None]
        results = [{"boxes": (b,), "labels": (l,)} for b, l in zip(boxes, gt_label)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_swin_backbone(args):
    """Build Swin Transformer backbone with pretrained weights"""
    from torchvision.models.swin_transformer import swin_t, swin_s, swin_b
    from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights

    if args.backbone == "swin_tiny":
        backbone_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        num_channels = 768
    elif args.backbone == "swin_small":
        backbone_model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        num_channels = 768
    elif args.backbone == "swin_base":
        backbone_model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        num_channels = 1024
    else:
        raise ValueError(f"Unknown Swin backbone: {args.backbone}")

    # wrap torchvision model so it accepts NestedTensor and returns (features, pos)
    position_embedding = build_position_encoding(args)
    joiner = TVSwinJoiner(backbone_model, position_embedding)
    joiner.num_channels = num_channels
    return joiner


def build(args):
    num_classes = 3 if args.dataset_file != "coco" else 16  # CXR: 14+2 RSNA: 1+2

    if args.dataset_file == "cxr8":
        num_classes = 10  # 8+2
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    # Build Swin Transformer backbone with pretrained weights
    if args.backbone.startswith("swin_"):
        backbone = build_swin_backbone(args)
    else:
        backbone = build_backbone(args)

    transformer = build_transformer(args)

    label_encoder = build_label_encoder(args.hidden_dim, num_classes)

    point_encoder = build_point_encoder()

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        label_encoder=label_encoder,
        point_encoder=point_encoder,
    )

    matcher = build_matcher(args)
    weight_dict = {"loss_ce": 1, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef

    if args.cons_loss:
        weight_dict["loss_cons"] = args.cons_loss_coef
    if args.train_with_unlabel_imgs:
        weight_dict["loss_unlabelcons"] = args.unlabel_cons_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)

    losses = ["boxes"]
    if args.cons_loss:
        losses += ["consistency"]
    if args.masks:
        losses += ["masks"]

    criterion = PointCriterion(
        num_classes,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        args=args,
    )
    criterion.to(device)
    postprocessors = {"bbox": PostProcess()}

    return model, criterion, postprocessors
