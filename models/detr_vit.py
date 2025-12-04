# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model with Vision Transformer backbone
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (
    NestedTensor, nested_tensor_from_tensor_list,
    accuracy, get_world_size, interpolate,
    is_dist_avail_and_initialized
)

from models.position_encoding import build_position_encoding  # for pos enc

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .point_encoder import build_point_encoder
from .label_encoder import build_label_encoder
from .point_criterion import PointCriterion


class DETR(nn.Module):
    """ This is the DETR module with ViT backbone that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, label_encoder=None, point_encoder=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
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
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            points_supervison:
               - "points" , (N,2)
               - "object_ids" (N,)
               - "labels" (N,)

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        query_embed = self.point_encoder(
            points_supervision,
            self.backbone.position_embedding,
            self.label_encoder
        )
        bs = len(query_embed)

        hs = self.transformer(self.input_proj(src), mask, query_embed, pos[-1])

        depth = hs[0].size(0)

        outputs_class = []
        outputs_coord = []

        for idx in range(bs):
            cur_point_sup = points_supervision[idx]['points']  # obj_num x 2
            outputs_class.append(self.class_embed(hs[idx]))    # 6 x obj_num x cls_num

            o_coord = self.bbox_embed(hs[idx]).sigmoid() / 2
            o_coord[:, :, 0] = (-o_coord[:, :, 0] + cur_point_sup[None, :, 0]).clamp_(min=0.001)
            o_coord[:, :, 1] = (-o_coord[:, :, 1] + cur_point_sup[None, :, 1]).clamp_(min=0.001)
            o_coord[:, :, 2] = (o_coord[:, :, 2] + cur_point_sup[None, :, 0]).clamp_(max=0.999)
            o_coord[:, :, 3] = (o_coord[:, :, 3] + cur_point_sup[None, :, 1]).clamp_(max=0.999)
            o_coord = box_ops.box_xyxy_to_cxcywh(o_coord)
            outputs_coord.append(o_coord)  # 6 x obj_num x 4

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
            gt_label.append(points_supervision[i]['labels'].unsqueeze(0))

        out = {'pred_boxes': outputs_coord_depth[-1], 'gt_label': gt_label}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_coord_depth)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b} for b in outputs_coord[:-1]]  # except last ..


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_bbox = outputs['pred_boxes']
        gt_label = outputs['gt_label']

        assert target_sizes.shape[1] == 2
        bs = len(out_bbox)

        boxes = []

        for idx in range(bs):
            boxes.append(box_ops.box_cxcywh_to_xyxy(out_bbox[idx]))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        for idx in range(bs):
            boxes[idx] = boxes[idx] * scale_fct[idx][None]
        results = [{'boxes': (b,), 'labels': (l,)} for b, l in zip(boxes, gt_label)]

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
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


class ViTBackbone(nn.Module):
    """
    Wraps a torchvision VisionTransformer so that:
    - input: NestedTensor(images, mask)
    - output: (features, pos) like the CNN backbone
      where:
        features[-1] is a NestedTensor with [B, C, H_feat, W_feat]
        pos[-1] is positional encoding for that feature map
    """
    def __init__(self, vit_model: nn.Module, position_embedding: nn.Module, num_channels: int):
        super().__init__()
        self.vit = vit_model
        self.position_embedding = position_embedding
        self.num_channels = num_channels

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        x, mask = samples.decompose()  # x: [B, 3, H, W], mask: [B, H, W] or None
        assert isinstance(x, torch.Tensor)

        # --- Ensure input matches ViT's expected image_size (e.g., 224x224) ---
        target_size = self.vit.image_size  # typically 224 for vit_b_16 / vit_l_16
        B, C, H, W = x.shape
        if H != target_size or W != target_size:
            x = F.interpolate(
                x,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )  # [B, 3, target_size, target_size]

        # -------------------------------------------------------
        # Follow torchvision VisionTransformer forward logic:
        # 1) _process_input -> patch tokens [B, N_patches, C]
        # 2) prepend class token -> [B, 1+N_patches, C]
        # 3) feed through encoder (adds pos_embedding internally)
        # -------------------------------------------------------
        # Step 1: patch tokens
        x_patches = self.vit._process_input(x)        # [B, N_patches, C]

        # Step 2: prepend CLS token
        n = x_patches.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)  # [B, 1, C]
        x_tokens = torch.cat([batch_class_token, x_patches], dim=1)  # [B, 1+N, C]

        # Step 3: encoder (adds pos_embedding, dropout, transformer blocks, ln)
        x_tokens = self.vit.encoder(x_tokens)         # [B, 1+N, C]

        # Drop class token, keep patch tokens
        x_patches_out = x_tokens[:, 1:, :]            # [B, N_patches, C]
        B, N, C = x_patches_out.shape

        # Assume square grid of patches: N = H_feat * W_feat
        H_feat = W_feat = int(N ** 0.5)
        assert H_feat * W_feat == N, "ViT patches do not form a square feature map"

        # Reshape to [B, C, H_feat, W_feat]
        x_feat = (
            x_patches_out.view(B, H_feat, W_feat, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Downsample mask to feature resolution
        if mask is not None:
            # mask: [B, H_orig, W_orig] -> [B, H_feat, W_feat]
            mask_feat = F.interpolate(
                mask[None].float(),
                size=(H_feat, W_feat),
                mode="nearest"
            )[0].bool()
        else:
            mask_feat = None

        feat_nested = NestedTensor(x_feat, mask_feat)
        pos = self.position_embedding(feat_nested)  # positional encoding for this feature map

        return [feat_nested], [pos]


def build_vit_backbone(args):
    """Build Vision Transformer backbone with pretrained weights and DETR-style interface"""
    from torchvision.models.vision_transformer import vit_b_16, vit_l_16
    from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights

    if args.backbone == 'vit_base':
        vit_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        num_channels = 768
    elif args.backbone == 'vit_large':
        vit_model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        num_channels = 1024
    else:
        raise ValueError(f"Unknown ViT backbone: {args.backbone}")

    # Build positional encoding like the CNN backbone uses
    position_embedding = build_position_encoding(args)

    # Wrap into ViTBackbone so DETR can use it the same way as CNN backbones
    backbone = ViTBackbone(vit_model, position_embedding, num_channels)
    return backbone


def build(args):
    num_classes = 3 if args.dataset_file != 'coco' else 16  # CXR: 14+2 RSNA: 1+2

    if args.dataset_file == 'cxr8':
        num_classes = 10  # 8+2
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    # Build Vision Transformer backbone with pretrained weights
    if args.backbone.startswith('vit_'):
        backbone = build_vit_backbone(args)
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
        point_encoder=point_encoder
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

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
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['boxes']
    if args.cons_loss:
        losses += ["consistency"]
    if args.masks:
        losses += ["masks"]

    criterion = PointCriterion(
        num_classes,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        args=args
    )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
