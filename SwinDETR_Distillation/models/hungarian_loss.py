""" Hungarian Loss for DETR Encoder-Decoder"""


import torch
import torch.nn.functional as F
from torch import nn, Tensor

from typing import Dict, List, Tuple

from .hungarian_matcher import HungarianMatcher
from utils.box_ops import box_cxcywh_to_xyxy, box_iou, giou


class HungarianLoss(nn.Module):
    def __init__(self, args):
        """
        args:
            - args.class_cost_weight, args.bbox_cost_weight, args.giou_cost_weight
            - args.num_class: number of object categories, omitting the special no-object category
            - args.eos_cost_weight: relative classification weight applied to the no-object category
        """
        super().__init__()

        self.matcher = HungarianMatcher(args.class_cost_weight, args.bbox_cost_weight, args.giou_cost_weight)
        self.num_class = args.num_class

        self.class_cost_weight = args.class_cost_weight
        self.bbox_cost_weight = args.bbox_cost_weight
        self.giou_cost_weight = args.giou_cost_weight

        empty_weight = torch.ones(args.num_class + 1)
        empty_weight[-1] = args.eos_cost_weight
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        ans = self._compute_loss(x, y)

        for i, aux in enumerate(x['aux']):
            ans.update({f'{k}_aux{i}': v for k, v in self._compute_loss(aux, y).items()})

        return ans

    def _compute_loss(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        Params:
            - x: a dictionary containing:
                'class': [batch_size, num_query, num_class + 1]
                'bbox': [batch_size, num_query, 4]
                'aux: a list of dictionaries containing the same keys as above (list size = num_decoder_layer - 1 (exclude last layer))
            - y: a list of dictionaries containing:
                'labels': a tensor of shape [num_objects] that stores the ground-truth classes of objects
                'boxes': a tensor of shape [num_objects, 4] that stores the ground-truth bounding boxes of objects represented as [cx, cy, w, h]

        Returns:
            a dictionary containing classification loss, bbox loss, and GIoU loss
        """
        ids = self.matcher(x, y)
        idx = self._get_permutation_idx(ids)

        # --- classification loss ---
        logits = x['class']

        target_class_o = torch.cat([t['labels'][J] for t, (_, J) in zip(y, ids)])
        target_class = torch.full(logits.shape[:2], self.num_class, dtype=torch.int64, device=logits.device)
        target_class[idx] = target_class_o

        classification_loss = F.cross_entropy(logits.transpose(1, 2), target_class, self.empty_weight)
        classification_loss *= self.class_cost_weight

        # --- bbox loss (ignore boxes that has no object) ---
        mask = target_class_o != self.num_class
        boxes = x['bbox'][idx][mask]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(y, ids)], dim=0)[mask]

        num_boxes = len(target_boxes) + 1e-6

        bbox_loss = F.l1_loss(boxes, target_boxes, reduction='none')
        bbox_loss = bbox_loss.sum() / num_boxes
        bbox_loss *= self.bbox_cost_weight

        # --- giou loss ---
        giou_loss = 1 - torch.diag(giou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target_boxes)))
        giou_loss = giou_loss.sum() / num_boxes
        giou_loss *= self.giou_cost_weight

        # --- compute statistics (mAP) ---
        with torch.no_grad():
            pred_class = F.softmax(logits[idx], -1).max(-1)[1]
            class_mask = (pred_class == target_class_o)[mask]
            iou = torch.diag(box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target_boxes))[0])

            ap = []
            for threshold in range(50, 100, 5):
                ap.append(((iou >= threshold / 100) * class_mask).sum() / num_boxes)

            ap = torch.mean(torch.stack(ap))

        return {'classification loss': classification_loss,
                'bbox loss': bbox_loss,
                'GIoU loss': giou_loss,
                'mAP': ap}

    @staticmethod
    def _get_permutation_idx(indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx