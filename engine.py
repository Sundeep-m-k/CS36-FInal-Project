# ------------------------------------------------------------------------
# UP-DETR
# Copyright (c) Tencent, Inc.
# Modified from DETR (Facebook AI Research)
# ------------------------------------------------------------------------
"""
Training and evaluation logic used by main scripts
"""

# -----------------------------
# Imports
# -----------------------------
import math
import os
import sys
from typing import Iterable
import copy
import collections

import torch
import torchvision.transforms.functional as F
from numpy import save

import datasets.transforms as T
import util.misc as utils

from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


# -----------------------------
# Global storage class
# Used to store global variables
# -----------------------------
class globalV():
    def __init__(self):
        # dictionary for shared values
        self.global_v = collections.defaultdict(int)

    def set_value(self, key, value):
        self.global_v[key] = value

    def get_value(self, key):
        return self.global_v[key]


# Global object used across training
global_value = globalV()


# -----------------------------
# Helper function
# Select elements of list by index
# -----------------------------
def splitList_by_idx(List, idx):
    idx = set(idx)
    return [v for i, v in enumerate(List) if i in idx]


# -----------------------------
# Random erasing augmentation
# Used for unlabeled data
# -----------------------------
t_randomErasing = T.RandomErasing(
    times=20, p=1, scale=(0.005, 0.02), value=0
)


# ------------------------------------------------------------------------
# Training for one epoch
# ------------------------------------------------------------------------
def train_one_epoch(
        args,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float = 0):

    # Switch model and loss into training mode
    model.train()
    criterion.train()

    # -----------------------------
    # Setup metric logger
    # -----------------------------
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr',
        utils.SmoothedValue(window_size=1, fmt='{value:.6f}')
    )

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # -----------------------------
    # Training loop
    # -----------------------------
    for samples, points_supervision, targets, filenames, is_unlabels in \
        metric_logger.log_every(data_loader, print_freq, header):

        # Move data to GPU
        samples = samples.to(device)
        points_supervision = [
            {k: v.to(device) for k, v in t.items()}
            for t in points_supervision
        ]
        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        samples_tensors = samples.tensors

        # -----------------------------
        # Semi-supervised learning
        # Handle unlabeled images
        # -----------------------------
        if args.train_with_unlabel_imgs and sum(is_unlabels) >= 1:

            # Skip batches where all images are unlabeled
            if len(targets) == sum(is_unlabels):
                continue

            record_idx = []   # indices of unlabeled images
            flip_imgs = []   # horizontally flipped copies
            flip_points = [] # flipped supervision points

            # Generate flipped images & adjusted points
            for idx, (filename, sample, point_sup, is_unlabel) in enumerate(
                    zip(filenames, samples_tensors,
                        points_supervision, is_unlabels)):

                if is_unlabel:
                    record_idx.append(idx)

                    # Flip image horizontally
                    flip_img = torch.flip(sample, dims=[-1])

                    # Apply random erasing augmentation (RSNA)
                    if args.dataset_file == 'rsna':
                        flip_img = t_randomErasing(flip_img)

                    flip_imgs.append(flip_img.unsqueeze(0))

                    # Flip corresponding point annotations
                    point_sup_ = copy.deepcopy(point_sup)
                    point_sup_['points'][:, 0] = \
                        1 - point_sup_['points'][:, 0]

                    # Add small noise to points
                    eps = 0.05
                    relative = torch.Tensor(
                        point_sup_['points'].size(0), 2
                    ).uniform_(eps, eps).to(point_sup_['points'].device)

                    point_sup_['points'] += relative

                    flip_points.append(point_sup_)

            # Verify number of flipped images matches points
            assert len(flip_imgs) == len(flip_points)

            # Append augmented unlabeled samples to batch
            flip_imgs = torch.cat(flip_imgs, dim=0)
            samples = torch.cat([samples_tensors, flip_imgs], dim=0)

            points_supervision += flip_points

        # -----------------------------
        # Forward pass
        # -----------------------------
        outputs = model(samples, points_supervision)

        # -----------------------------
        # Separate labeled vs unlabeled
        # -----------------------------
        if args.train_with_unlabel_imgs and sum(is_unlabels) >= 1:

            outputs_for_ori     = {'pred_boxes': [], 'aux_outputs': []}
            outputs_for_unlabel = {'pred_boxes': [], 'aux_outputs': []}

            pred_boxes, aux_outputs = (
                outputs['pred_boxes'], outputs['aux_outputs']
            )

            N_extra = len(flip_imgs)
            ori_batch_num = len(targets)
            cur_batch_num = len(points_supervision)

            # Split predictions
            pred_boxes_ori, pred_boxes_unlabel_flip = (
                pred_boxes[:cur_batch_num - N_extra],
                pred_boxes[cur_batch_num - N_extra:]
            )

            # Separate labeled & unlabeled sets
            label_idx = list(set(record_idx) ^ set(range(ori_batch_num)))
            targets = splitList_by_idx(targets, label_idx)

            pred_boxes_ori, pred_boxes_unlabel = (
                splitList_by_idx(pred_boxes_ori, label_idx),
                splitList_by_idx(pred_boxes_ori, record_idx)
            )

            # Combine unlabeled predictions
            assert len(pred_boxes_unlabel) == len(pred_boxes_unlabel_flip)

            outputs_for_ori['pred_boxes'] += pred_boxes_ori
            outputs_for_unlabel['pred_boxes'].append(
                torch.cat(pred_boxes_unlabel + pred_boxes_unlabel_flip, dim=0)
            )

            # Handle auxiliary outputs
            for aux_output in aux_outputs:

                aux_output = aux_output['pred_boxes']

                aux_output_ori, aux_output_unlabel_flip = (
                    aux_output[:cur_batch_num - N_extra],
                    aux_output[cur_batch_num - N_extra:]
                )

                aux_output_ori, aux_output_unlabel = (
                    splitList_by_idx(aux_output_ori, label_idx),
                    splitList_by_idx(aux_output_ori, record_idx)
                )

                outputs_for_ori['aux_outputs'].append(
                    {'pred_boxes': aux_output_ori}
                )

                outputs_for_unlabel['aux_outputs'].append(
                    {'pred_boxes': [
                        torch.cat(
                            aux_output_unlabel +
                            aux_output_unlabel_flip,
                            dim=0
                        )
                    ]}
                )

            outputs = outputs_for_ori
            unlabel_outputs = outputs_for_unlabel

        # -----------------------------
        # Compute losses
        # -----------------------------
        loss_dict = criterion(outputs, targets)

        # Add unlabeled consistency loss
        if args.train_with_unlabel_imgs and sum(is_unlabels) >= 1:
            loss_dict_unlabel = criterion(
                unlabel_outputs,
                targets=None,
                specifiec_loss='cal_unlabel_consistency'
            )
            loss_dict.update(loss_dict_unlabel)

        # Apply loss weights
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k]
            for k in loss_dict.keys()
            if k in weight_dict
        )

        # -----------------------------
        # Reduce losses for logging
        # -----------------------------
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict and len(k.split('_')) == 2
        }

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # Check for invalid values
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # -----------------------------
        # Backpropagation step
        # -----------------------------
        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # Update metrics
        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # -----------------------------
    # Aggregate metrics
    # -----------------------------
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
    }


# ------------------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
        args,
        model,
        criterion,
        postprocessors,
        data_loader,
        base_ds,
        device,
        output_dir):

    # Switch into evaluation mode
    model.eval()
    criterion.eval()

    # Metric logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # COCO evaluators
    iou_types = tuple(
        k for k in ('segm', 'bbox') if k in postprocessors.keys()
    )
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    # Panoptic evaluator if needed
    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    # CSV output for saving pseudo labels
    if args.save_csv:
        import csv
        csv_write = csv.writer(open(args.save_csv, 'w'))

    # -----------------------------
    # Evaluation loop
    # -----------------------------
    for samples, points_supervision, targets, filename, _ in \
            metric_logger.log_every(data_loader, 10, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        points_supervision = [
            {k: v.to(device) for k, v in t.items()}
            for t in points_supervision
        ]

        outputs = model(samples, points_supervision)

        # Compute losses
        loss_dict = criterion(outputs, targets, 'test')

        weight_dict = criterion.weight_dict
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict and len(k.split('_')) == 2
        }

        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled
        )

        # -----------------------------
        # Convert predictions to boxes
        # -----------------------------
        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0
        )
        results = postprocessors['bbox'](
            outputs, orig_target_sizes
        )

        # For segmentation masks
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack(
                [t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](
                results, outputs,
                orig_target_sizes, target_sizes
            )

        # Map results to image IDs
        res = {
            target['image_id'].item(): output
            for target, output in zip(targets, results)
        }

        # -----------------------------
        # Update evaluators
        # -----------------------------
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:

            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes
            )

            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"

                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # -----------------------------
    # Aggregate evaluation results
    # -----------------------------
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # Final compute
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()

    # Collect statistics
    stats = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
    }

    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = \
                coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = \
                coco_evaluator.coco_eval['segm'].stats.tolist()

    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th']  = panoptic_res["Things"]
        stats['PQ_st']  = panoptic_res["Stuff"]

    return stats, coco_evaluator
