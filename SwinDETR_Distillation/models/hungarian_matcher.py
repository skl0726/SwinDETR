""" Hungarian Matcher for DETR Encoder-Decoder"""


import torch
import torch.nn as nn
from torch import Tensor

from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment

from utils.box_ops import box_cxcywh_to_xyxy, giou


class HungarianMatcher(nn.Module):
    def __init__(self, class_cost_weight, bbox_cost_weight, giou_cost_weight):
        """
        Params:
            - closs_cost_weight: relative weight of the classification error in the matching cost
            - bbox_cost_weight: relative weight of the L1 error of the bounding box coordinates in the matching cost
            - giou_cost_weight: relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()

        self.class_cost_weight = class_cost_weight
        self.bbox_cost_weight = bbox_cost_weight
        self.giou_cost_weight = giou_cost_weight

    @torch.no_grad()
    def forward(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        batch_size, num_query = x['class'].shape[:2]

        out_prob = x['class'].flatten(0, 1).softmax(-1) # [batch_size * num_query, num_classes + 1]
        out_bbox = x['bbox'].flatten(0, 1) # [batch_size * num_query, 4]

        tgt_ids = torch.cat([t['labels'] for t in y])
        tgt_bbox = torch.cat([t['boxes'] for t in y])

        # class prediction cost
        class_cost = -out_prob[:, tgt_ids]
        # bounding box coordinate prediction cost
        bbox_cost = torch.cdist(out_bbox, tgt_bbox, p=1) # L1 cost
        giou_cost = -giou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)) # GIoU cost

        cost_matrix = self.bbox_cost_weight * bbox_cost + self.class_cost_weight * class_cost + self.giou_cost_weight * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_query, -1).cpu() # [batch_size, num_query, # of GT target]

        sizes = [len(t['boxes']) for t in y]
        # 1) split the last dimension of cost matrix by sizes
        # 2) linear_sum_assignment: hunagarian algorithm function (scipy)
        #    -> returns pairs of indices between queries and ground truth that achieve the minimum cost in a one-to-one matching
        ids = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        # return the indices of the matched pairs (predicted, GT)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in ids]