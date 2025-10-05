""" Hard Distillation Loss using Hungarian Loss for DETR Encoder-Decoder """


import torch
import torch.nn as nn
from torch import Tensor

from typing import Dict, List, Union

from .hungarian_loss import HungarianLoss
from utils.box_ops import box_xyxy_to_cxcywh


class HardDistillationLoss(nn.Module):
    def __init__(self, teacher: nn.Module, args):
        """
        Hard distillation: 
         - teacher prediction: pseudo‐GT, apply HungarianLoss
        """
        super().__init__()

        self.teacher = teacher
        self.alpha = args.alpha
        self.criterion = HungarianLoss(args)
        self.num_class = args.num_class
        self.score_thresh = args.score_thresh

    def forward(
        self,
        x: torch.Tensor,
        outputs: Union[
            Dict[str, Tensor],  # detection output
            Dict[str, Tensor]   # distillation output
        ],
        labels: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        _, _, img_h, img_w = x.shape

        student_output_det, student_output_distill = outputs

        # 1) detection loss (student loss) on ground‐truth
        loss_det = self.criterion(student_output_det, labels)
        
        # 2) teacher forward -> create pseudo‐GT
        with torch.no_grad():
            teacher_outputs = self.teacher(x)  # List[Dict], len(List) = batch_size

        pseudo_targets = []
        for t_out in teacher_outputs:
            boxes  = t_out['boxes']   # [N_pred, 4]
            labels = t_out['labels']  # [N_pred]
            scores = t_out['scores']  # [N_pred]
            keep = scores >= self.score_thresh

            boxes_keep = boxes[keep]
            boxes_cxcywh = box_xyxy_to_cxcywh(boxes_keep) # Faster R-CNN Bbox output style: (x1, y1, x2, y2)
            # normalization
            boxes_cxcywh[:, [0, 2]] /= img_w
            boxes_cxcywh[:, [1, 3]] /= img_h

            pseudo_targets.append({
                'boxes':  boxes_cxcywh,
                'labels': labels[keep],
            })

        # 3) HungarianLoss on student distill outputs vs pseudo‐GT
        loss_dist = self.criterion(student_output_distill, pseudo_targets)

        # 4) combine detection & distillation losses
        loss = {}
        for k in loss_det:
            loss[k] = (1 - self.alpha) * loss_det[k] + self.alpha * loss_dist[k]

        return loss