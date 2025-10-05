""" utilities for bounding box manipulation and GIoU """


import torch


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2,
         (x2 - x1), (y2 - y1)]
    return torch.stack(b, dim=-1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def box_iou(boxes1, boxes2):
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # [N, M, 2]

    wh = (rb - lt).clamp(min=0) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

    union = area1[:, None] + area2 - inter
    
    iou = inter / union

    return iou, union


def giou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0) # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area