""" SwinDETR Model (with inference module) """


import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.quantization import quantize_dynamic

from typing import Dict, Union, List, Tuple

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.backbone import build_backbone
from models.transformer import Transformer
from utils.box_ops import box_cxcywh_to_xyxy


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

class SwinDETR(nn.Module):
    def __init__(self, args):
        """
        args:
            - (args.num_groups, args.growth_rate, args.num_blocks, args.swin_transformer)
            - args.hidden_dim
            - args.num_head
            - args.num_encoder_layer, args.num_decoder_layer
            - args.dim_feedforward
            - args.dropout
            - args.num_query
            - args.num_class
        """
        super().__init__()

        if args.swin_transformer == 'swin_t' or args.swin_transformer == 'swin_b':
            self.set_swin_transformer = args.swin_transformer
        else:
            self.set_swin_transformer = None            

        self.backbone = build_backbone(args)

        if self.set_swin_transformer is None:
            self.reshape = nn.Conv2d(self.backbone.backbone.out_channels, args.hidden_dim, kernel_size=1)
        
        self.transformer = Transformer(args.hidden_dim, args.num_head, args.num_encoder_layer, args.num_decoder_layer,
                                       args.dim_feedforward, args.dropout)
        
        self.query_embed = nn.Embedding(args.num_query, args.hidden_dim)
        self.class_embed = nn.Linear(args.hidden_dim, args.num_class + 1)
        self.bbox_embed = MLP(args.hidden_dim, args.hidden_dim, 4, 3)

    def forward(self, x) -> Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]:
        """
        Params:
            - x: a tensor of shape [batch_size, 3, image_height, image_width]

        Returns:
            a dictionary with the following elements:
                - class: the classification results for all queries with shape [batch_size, num_query, num_class + 1] (+1 stands for no object class)
                - bbox: the normalized bounding box for all queries with shape [batch_size, num_query, 4] (represented as [cx, cy, width, height])
        """
        features, (pos, mask) = self.backbone(x)
        
        if self.set_swin_transformer is None:
            features = self.reshape(features)

        out = self.transformer(features, mask, self.query_embed.weight, pos)

        output_class = self.class_embed(out)
        output_coord = self.bbox_embed(out).sigmoid()

        return {'class': output_class[-1],
                'bbox': output_coord[-1],
                'aux': [{'class': oc, 'bbox': ob} for oc, ob in zip(output_class[:-1], output_coord[:-1])]}
    

class SwinDETRWrapper(nn.Module):
    def __init__(self, model, post_process):
        super().__init__()
        self.swin_detr = model
        self.post_process = post_process

    def forward(self, x, img_size) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Params:
            - x: batch images of shape [batch_size, 3, args.target_height, args.target_width] where batch_size equals to 1
                 (if tensor with batch_size larger than 1 is passed in, only the first image prediction will be returned)
            - img_size: tensor of shape [batch_size, img_width, img_height]

        Returns:
            the first image prediction in the following order: scores, labels, boxes
        """
        out = self.swin_detr(x)
        out = self.post_process(out, img_size)[0]
        return out['scores'], out['labels'], out['boxes']
    

class PostProcess(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x, img_size) -> List[Dict[str, Tensor]]:
        logits, bboxes = x['class'], x['bbox']
        prob = F.softmax(logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        img_w, img_h = img_size.unbind(1)
        scale = torch.stack([img_w, img_h, img_w, img_h], 1).unsqueeze(1)
        boxes = box_cxcywh_to_xyxy(bboxes)
        boxes *= scale

        return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]


@torch.no_grad()
def build_inference_model(args, quantize=False):
    assert os.path.exists(args.weight), 'inference model should have pre-trained weight'
    device = torch.device(args.device)

    model = SwinDETR(args).to(device)
    model.load_state_dict(torch.load(args.weight, map_location=device))

    post_process = PostProcess.to(device)

    wrapper = SwinDETRWrapper(model, post_process).to(device)
    wrapper.eval()

    if quantize:
        wrapper = quantize_dynamic(wrapper, {nn.Linear})

    print('optimizing model for inference...')
    return torch.jit.trace(wrapper, (torch.rand(1, 3, args.target_height, args.target_width).to(device),
                                     torch.as_tensor([args.target_width, args.target_height]).unsqueeze(0).to(device)))