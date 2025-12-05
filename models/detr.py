# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.

This file defines:
- TVJoiner: wrapper for torchvision backbones (Swin / ViT) to match DETR-style interface
- DETR: main detection model for PBC using point-supervision
- SetCriterion / PointCriterion: loss computation
- PostProcess: conversion to COCO-style outputs
- MLP: simple feed-forward head used to predict bounding boxes
- build(): factory that builds backbone + transformer + model + criterion
"""

from os import O_SYNC
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (
    NestedTensor, nested_tensor_from_tensor_list,
    accuracy, get_world_size, interpolate,
    is_dist_avail_and_initialized
)

from .backbone import build_backbone
from .matcher import build_matcher
# from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
#                            dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from .point_encoder import build_point_encoder
from .label_encoder import build_label_encoder
from .point_criterion import PointCriterion
from .position_encoding import build_position_encoding


# ----------------------------------------------------------------------
# Helper joiner for torchvision backbones (Swin / ViT)
# ----------------------------------------------------------------------
class TVJoiner(torch.nn.Module):
    """
    Wrapper to make torchvision backbones (e.g., Swin, ViT) compatible with
    the DETR/PBC pipeline.

    It:
    - accepts a NestedTensor (image + mask)
    - calls the backbone to get feature maps
    - builds a NestedTensor for each feature map
    - computes positional encodings for each map

    Returns:
        out: list of NestedTensor feature maps
        pos: list of positional encodings for each feature level
    """
    def __init__(self, backbone_module, position_embedding):
        super().__init__()
        self.backbone = backbone_module              # torchvision backbone
        self.position_embedding = position_embedding # positional encoding module
        # num_channels (output feature dim) is set externally on the instance

    def forward(self, tensor_list):
        """Supports NestedTensor and plain Tensor inputs. Returns (features_list, pos_list)"""
        # Case 1: standard NestedTensor (image + padding mask)
        if isinstance(tensor_list, NestedTensor):
            x = tensor_list.tensors

            # If model defines forward_features (e.g., Swin, ViT), use it.
            if hasattr(self.backbone, 'forward_features'):
                feat = self.backbone.forward_features(x)
            else:
                feat = self.backbone(x)

            # Normalize to a dict of features
            if isinstance(feat, torch.Tensor):
                xs = {'0': feat}
            elif isinstance(feat, dict):
                xs = feat
            else:
                # Fallback: try to iterate over a list of feature maps
                try:
                    xs = {str(i): t for i, t in enumerate(feat)}
                except Exception:
                    raise RuntimeError('Unsupported backbone output type')

            out = []
            pos = []

            # Build NestedTensor + positional embedding for each feature level
            for name, x in xs.items():
                m = tensor_list.mask
                assert m is not None

                # Resize mask to feature map spatial size
                mask = F.interpolate(
                    m[None].float(), size=x.shape[-2:]
                ).to(torch.bool)[0]

                nt = NestedTensor(x, mask)
                out.append(nt)
                pos.append(self.position_embedding(nt).to(x.dtype))

            return out, pos

        # Case 2: plain tensor (no NestedTensor/mask)
        else:
            if hasattr(self.backbone, 'forward_features'):
                feat = self.backbone.forward_features(tensor_list)
            else:
                feat = self.backbone(tensor_list)

            if isinstance(feat, torch.Tensor):
                return [feat], [None]
            elif isinstance(feat, dict):
                return list(feat.values()), [None] * len(feat)
            else:
                return list(feat), [None] * len(feat)


# ----------------------------------------------------------------------
# DETR model used in PBC
# ----------------------------------------------------------------------
class DETR(nn.Module):
    """ DETR module that performs object detection from point-supervised queries. """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        label_encoder=None,
        point_encoder=None
    ):
        """
        Args:
            backbone: feature extractor (ResNet / Swin / ViT) wrapped to return NestedTensor features
            transformer: DETR-style encoder-decoder transformer
            num_classes: number of foreground classes (without "no-object")
            num_queries: maximum number of objects per image (query slots)
            aux_loss: if True, also compute intermediate decoder losses
            label_encoder: encodes class labels into embedding space
            point_encoder: encodes point supervision into query embeddings
        """
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        # Classification head: predicts (num_classes + 1) with "no-object"
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        # Bounding box head: predicts 4 numbers (cx, cy, w, h) from hidden_dim
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Query embedding (used by standard DETR; in PBC, we override using point_encoder)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 1x1 conv to project backbone feature channels -> transformer hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.label_encoder = label_encoder
        self.point_encoder = point_encoder
        self.backbone = backbone
        self.aux_loss = aux_loss


    def forward(self, samples: NestedTensor, points_supervision):
        """
        Forward pass of PBC DETR.

        Args:
            samples:
                - NestedTensor with:
                    samples.tensors: [B, 3, H, W]
                    samples.mask: [B, H, W] (1 = padded region)
            points_supervision: list of dicts (length B) with keys:
                - "points":  (N_i, 2) normalized x,y coordinates per image
                - "object_ids": (N_i,) optional ids
                - "labels": (N_i,) class labels for each annotated point

        Returns:
            dict with keys:
                - "pred_boxes": list[Tensor] shaped [num_layers x N_i x 4]
                - "gt_label":  list[Tensor] shaped [1 x N_i] labels per image
                - "aux_outputs": (optional) intermediate decoder box predictions
        """

        # Convert list/tensor input into NestedTensor if needed
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # Extract features and positional encodings from backbone
        features, pos = self.backbone(samples)

        # Use the last (highest-level) feature map
        src, mask = features[-1].decompose()
        assert mask is not None

        # Encode point annotations into query embeddings (one set per image)
        query_embed = self.point_encoder(
            points_supervision,
            self.backbone.position_embedding,
            self.label_encoder
        )
        bs = len(query_embed)

        # Transformer encoder-decoder
        # hs is a list (length B) of tensors [num_layers, N_i, hidden_dim]
        hs = self.transformer(
            self.input_proj(src),  # project features to hidden_dim
            mask,
            query_embed,
            pos[-1]               # positional encoding for last feature map
        )

        depth = hs[0].size(0)  # number of decoder layers

        outputs_class = []
        outputs_coord = []

        # Process each image in the batch independently because N_i (points) can differ
        for idx in range(bs):
            cur_point_sup = points_supervision[idx]['points']  # [N_i, 2]

            # Class predictions: [num_layers, N_i, num_classes+1]
            outputs_class.append(self.class_embed(hs[idx]))

            # Box predictions: [num_layers, N_i, 4], values ~ [0,1] after sigmoid
            o_coord = self.bbox_embed(hs[idx]).sigmoid() / 2

            # Decode from relative offsets to final box coordinates anchored at points
            # (cx1, cy1, cx2, cy2) in [0,1], then convert to cxcywh
            o_coord[:, :, 0] = (-o_coord[:, :, 0] + cur_point_sup[None, :, 0]).clamp_(min=0.001)
            o_coord[:, :, 1] = (-o_coord[:, :, 1] + cur_point_sup[None, :, 1]).clamp_(min=0.001)
            o_coord[:, :, 2] = (o_coord[:, :, 2] + cur_point_sup[None, :, 0]).clamp_(max=0.999)
            o_coord[:, :, 3] = (o_coord[:, :, 3] + cur_point_sup[None, :, 1]).clamp_(max=0.999)

            # Convert (x1,y1,x2,y2) -> (cx,cy,w,h)
            o_coord = box_ops.box_xyxy_to_cxcywh(o_coord)
            outputs_coord.append(o_coord)  # [num_layers, N_i, 4]

        # Re-arrange outputs by decoder depth (layer-wise) and batch dimension
        outputs_class_depth = []
        outputs_coord_depth = []

        for dep_idx in range(depth):
            batched_cls = []
            batched_coord = []
            for idx in range(bs):
                batched_cls.append(outputs_class[idx][dep_idx])   # [N_i, C]
                batched_coord.append(outputs_coord[idx][dep_idx]) # [N_i, 4]
            outputs_class_depth.append(batched_cls)
            outputs_coord_depth.append(batched_coord)

        # Collect ground-truth labels per image
        gt_label = []
        for i in range(bs):
            gt_label.append(points_supervision[i]['labels'].unsqueeze(0))

        # Final output uses last decoder layer boxes + labels
        out = {
            'pred_boxes': outputs_coord_depth[-1],
            'gt_label': gt_label
        }

        # Add auxiliary outputs (intermediate decoder layers) if enabled
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_coord_depth)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        """
        Prepare auxiliary loss outputs (for all decoder layers except the last).
        torchscript does not support dicts with heterogeneous entries, so we store
        only 'pred_boxes' in each auxiliary dict.
        """
        return [{'pred_boxes': b} for b in outputs_coord[:-1]]


# ----------------------------------------------------------------------
# SetCriterion (not directly used here; PBC uses PointCriterion)
# ----------------------------------------------------------------------
class SetCriterion(nn.Module):
    """
    Standard DETR criterion (Hungarian matching + classification + bbox + GIoU losses).

    In this PBC project, we primarily use PointCriterion, but SetCriterion is kept
    for reference and potential full DETR training.
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        Args:
            num_classes: #foreground classes (no-object is class index num_classes)
            matcher: Hungarian matcher module
            weight_dict: weights for each loss component
            eos_coef: weight for "no-object" class in classification loss
            losses: list of active loss types (e.g., ['labels', 'boxes', 'cardinality'])
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

        """
        Example of weight_dict:
        {
            'loss_ce'  : 1, 'loss_bbox'  : 5, 'loss_giou'  : 2,
            'loss_ce_0': 1, 'loss_bbox_0': 5, 'loss_giou_0': 2,
            ...
        }
        """
        self.eos_coef = eos_coef
        self.losses = losses

        # Class weights for cross-entropy:
        # last index is "no-object", scaled by eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss (cross-entropy over predicted logits).

        targets: list of dicts with key "labels"
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # [B, Q, C+1]

        idx = self._get_src_permutation_idx(indices)
        # Gather GT labels for matched targets
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        # Initialize all queries as "no-object"
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )

        # Fill matched positions with actual labels
        target_classes[idx] = target_classes_o

        # Cross-entropy with class weights
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {'loss_ce': loss_ce}

        if log:
            # Classification error for logging (not a separate loss)
            losses['class_error'] = 100 - accuracy(
                src_logits[idx], target_classes_o
            )[0]
        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Cardinality error: absolute error in number of predicted non-empty boxes.
        Used only for logging; does not backpropagate gradients.
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device

        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )

        # Count number of predictions that are NOT "no-object"
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)

        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Bounding box regression loss:
        - L1 loss on (cx,cy,w,h)
        - GIoU loss on boxes converted to (x1,y1,x2,y2)

        targets must contain "boxes": [N_i, 4] in normalized cxcywh format.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)],
            dim=0
        )

        # L1 regression loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # GIoU loss
        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Placeholder for mask losses (not used in this project).
        Would normally contain focal and dice losses for segmentation.
        """
        pass


    def _get_src_permutation_idx(self, indices):
        """
        Permute predictions following matching indices.
        """
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    def _get_tgt_permutation_idx(self, indices):
        """
        Permute targets following matching indices.
        """
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """
        Dispatch to the correct loss function based on 'loss' string.
        """
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    def forward(self, outputs, targets):
        """
        Compute full loss dictionary.

        Args:
            outputs: dict from DETR model
            targets: list of dicts (one per image) with keys depending on loss types
        """
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != 'aux_outputs'
        }

        # Hungarian matching on last decoder layer
        indices = self.matcher(outputs_without_aux, targets)

        # Compute average number of target boxes across processes (for normalization)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device
        )

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(
            num_boxes / get_world_size(), min=1
        ).item()

        # Compute all requested loss components
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes)
            )

        # If auxiliary outputs exist, compute losses for intermediate layers too
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Skip mask losses for intermediate layers (too expensive)
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging only for last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    # Add layer suffix (_0, _1, ...)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


# ----------------------------------------------------------------------
# Post-processing: convert predictions to COCO-style dicts
# ----------------------------------------------------------------------
class PostProcess(nn.Module):
    """
    This module converts the model's output into the format expected by the COCO API.
    """

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        Args:
            outputs: dict from model forward() with keys:
                - 'pred_boxes': list of tensors (normalized cxcywh)
                - 'gt_label':  list of label tensors
            target_sizes: [B, 2] tensor of (height, width) for each original image

        Returns:
            List of dicts, one per image:
                {
                    'boxes': (tensor of xyxy boxes scaled to image size,),
                    'labels': (tensor of labels,)
                }
        """
        out_bbox = outputs['pred_boxes']
        gt_label = outputs['gt_label']

        assert target_sizes.shape[1] == 2
        bs = len(out_bbox)

        boxes = []

        # Convert cxcywh to xyxy
        for idx in range(bs):
            boxes.append(
                box_ops.box_cxcywh_to_xyxy(out_bbox[idx])
            )

        # Scale normalized boxes to absolute pixel coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack(
            [img_w, img_h, img_w, img_h], dim=1
        )

        for idx in range(bs):
            boxes[idx] = boxes[idx] * scale_fct[idx][None]

        # Package results in a list of dicts
        results = [
            {'boxes': (b,), 'labels': (l,)}
            for b, l in zip(boxes, gt_label)
        ]
        return results


# ----------------------------------------------------------------------
# Simple MLP used for bounding box regression head
# ----------------------------------------------------------------------
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (feed-forward network). """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Args:
            input_dim: input feature dimension
            hidden_dim: hidden layer size
            output_dim: output size
            num_layers: total number of layers
        """
        super().__init__()
        self.num_layers = num_layers

        h = [hidden_dim] * (num_layers - 1)
        # Build linear layers: input_dim -> hidden_dim -> ... -> output_dim
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        """
        Forward pass through the MLP with ReLU activations
        on all but the last layer.
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


    # ------------------------------------------------------------------
    # Swin backbone builder (wrapped in TVJoiner)
    # NOTE: This is defined inside MLP in your file due to indentation.
    # ------------------------------------------------------------------
    def build_swin_backbone(args):
        """Build Swin Transformer backbone (pretrained) and wrap with TVJoiner."""
        # Import here so that Swin is only required if actually used
        from torchvision.models.swin_transformer import swin_t, swin_s, swin_b
        from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights

        if args.backbone == 'swin_tiny':
            tv_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
            num_channels = 768
        elif args.backbone == 'swin_small':
            tv_model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
            num_channels = 768
        elif args.backbone == 'swin_base':
            tv_model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
            num_channels = 1024
        else:
            raise ValueError(f"Unknown Swin backbone: {args.backbone}")

        position_embedding = build_position_encoding(args)
        joiner = TVJoiner(tv_model, position_embedding)
        joiner.num_channels = num_channels
        return joiner


    # ------------------------------------------------------------------
    # ViT backbone builder (wrapped in TVJoiner)
    # NOTE: Also indented inside MLP in your current file.
    # ------------------------------------------------------------------
    def build_vit_backbone(args):
        """Build Vision Transformer backbone (pretrained) and wrap with TVJoiner."""
        from torchvision.models.vision_transformer import vit_b_16, vit_l_16
        from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights

        if args.backbone == 'vit_base':
            tv_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            num_channels = 768
        elif args.backbone == 'vit_large':
            tv_model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
            num_channels = 1024
        else:
            raise ValueError(f"Unknown ViT backbone: {args.backbone}")

        position_embedding = build_position_encoding(args)
        joiner = TVJoiner(tv_model, position_embedding)
        joiner.num_channels = num_channels
        return joiner


# ----------------------------------------------------------------------
# Factory: build model, criterion, and postprocessors
# ----------------------------------------------------------------------
def build(args):
    """
    Build the full PBC DETR model + criterion + postprocessor.

    Selects:
    - number of classes depending on dataset
    - appropriate backbone (ResNet/Swin/ViT)
    - transformer, label encoder, point encoder
    - point-based criterion and post-processing
    """

    # Default number of classes:
    # - CXR: 3 classes (1 foreground + 2 extra?)
    # - COCO: 16 classes (as in original DETR)
    num_classes = 3 if args.dataset_file != 'coco' else 16  # CXR: 14+2 RSNA: 1+2

    if args.dataset_file == 'cxr8':
        num_classes = 10  # 8 classes + 2
    if args.dataset_file == "coco_panoptic":
        num_classes = 250

    device = torch.device(args.device)

    # Choose backbone type:
    # - torchvision Swin / ViT via TVJoiner
    # - or custom ResNet-like backbone via build_backbone()
    if hasattr(args, 'backbone') and isinstance(args.backbone, str) and args.backbone.startswith('swin_'):
        backbone = MLP.build_swin_backbone(args)
    elif hasattr(args, 'backbone') and isinstance(args.backbone, str) and args.backbone.startswith('vit_'):
        backbone = MLP.build_vit_backbone(args)
    else:
        backbone = build_backbone(args)

    # Build encoder-decoder transformer
    transformer = build_transformer(args)

    # Build label encoder to embed class labels
    label_encoder = build_label_encoder(args.hidden_dim, num_classes)

    # Build point encoder (turns annotated points + labels into queries)
    point_encoder = build_point_encoder()

    # Main DETR-style PBC model
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        label_encoder=label_encoder,
        point_encoder=point_encoder
    )

    # Hungarian matcher
    matcher = build_matcher(args)

    # Loss weights
    weight_dict = {
        'loss_ce': 1,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef
    }

    # Consistency losses for PBC
    if args.cons_loss:
        weight_dict["loss_cons"] = args.cons_loss_coef
    if args.train_with_unlabel_imgs:
        weight_dict["loss_unlabelcons"] = args.unlabel_cons_loss_coef

    # Optional mask losses (not used in this point-based project)
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # If using auxiliary decoder losses, duplicate weights for each intermediate layer
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({
                k + f'_{i}': v for k, v in weight_dict.items()
            })
        weight_dict.update(aux_weight_dict)

    # Active loss components for PBC
    losses = ['boxes']
    if args.cons_loss:
        losses += ["consistency"]
    if args.masks:
        losses += ["masks"]

    # Point-based criterion (custom for PBC)
    criterion = PointCriterion(
        num_classes,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        args=args
    )
    criterion.to(device)

    # Only bbox post-processor needed (no segmentation)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
