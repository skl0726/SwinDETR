""" Hungarian Matcher for DETR """


import torch
from torch import nn, Tensor

from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.box_ops import box_cxcywh_to_xyxy, giou


class HungarianMatcher(nn.Module):
    def __init__(self, class_cost, bbox_cost, giou_cost):
        """
        Params:
            - closs_cost: relative weight of the classification error in the matching cost
            - bbox_cost: relative weight of the L1 error of the bounding box coordinates in the matching cost
            - giou_cost: relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def forward(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        batch_size, num_query = x['class'].shape[:2]

        out_prob = x['class'].flatten(0, 1).softmax(-1) # [batch_size * num_query, num_classes]
        out_bbox = x['bbox'].flatten(0, 1) # [batch_size * num_query, 4]

        tgt_ids = torch.cat([t['labels'] for t in y])
        tgt_bbox = torch.cat([t['boxes'] for t in y])

        class_cost = -out_prob[:, tgt_ids]
        bbox_cost = torch.cdist(out_bbox, tgt_bbox, p=1) # L1 cost
        giou_cost = -giou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_query, -1).cpu()

        sizes = [len(t['boxes']) for t in y]
        ids = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in ids]