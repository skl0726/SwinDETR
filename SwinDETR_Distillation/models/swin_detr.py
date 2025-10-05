""" SwinDETR Model (with inference module) """


import torch
import torch.nn.functional as F
from torch import nn, Tensor

from typing import Dict, Union, List

from .backbone.swin_transformer import SwinTransformer
from .positional_encoding import PositionEmbeddingSine
from .encoder_decoder import TransformerEncoderDecoder


def backbone(args):
    backbone_model = SwinTransformer(
        embed_dim=args.swin_embed_dim,
        depths=args.swin_depths,
        num_heads=args.swin_num_heads,
        window_size=args.swin_window_size,
        drop_path_rate=args.swin_drop_path_rate,
    ).to(args.device)
    
    # load pretrained weights
    checkpoint = torch.load(args.swin_weights_path, map_location=args.device)
    backbone_model.load_state_dict(checkpoint['model'], strict=False)

    return backbone_model


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
            - args.swin_weights_path = f"swin_weights/swin_{swin_version}_patch4_window7_224_22k.pth",
            - args.device = device,
            - args.hidden_dim = 768,
            - args.num_head = 8,
            - args.num_encoder_layer = 6,
            - args.num_decoder_layer = 6,
            - args.dim_feedforward = 2048,
            - args.dropout = 0.1,
            - args.num_query = 100,
            - args.num_distill_query = 100,
            - args.num_class = 90
        """
        super().__init__()

        # Swin Transformer Backbone
        self.swin_transformer_backbone = backbone(args)
        for param in self.swin_transformer_backbone.parameters(): # freeze the backbone
            param.requires_grad = False

        # Dimension Reduction
        self.reduction = nn.Linear(args.swin_out_embed_dim, args.hidden_dim)

        # Position Embedding
        self.position_embedding = PositionEmbeddingSine(args.hidden_dim // 2)

        # DETR Transformer Encoder-Decoder
        self.transformer_encoder_decoder = TransformerEncoderDecoder(args.hidden_dim,
                                                                      args.num_head,
                                                                      args.num_encoder_layer,
                                                                      args.num_decoder_layer,
                                                                      args.dim_feedforward,
                                                                      args.dropout)
        
        # Query Embedding
        self.query_embed = nn.Embedding(args.num_query, args.hidden_dim)
        self.distill_query_embed = nn.Embedding(args.num_distill_query, args.hidden_dim)

        # Classification and Bounding Box Prediction
        self.class_embed = nn.Linear(args.hidden_dim, args.num_class + 1)
        self.bbox_embed = MLP(args.hidden_dim, args.hidden_dim, 4, 3)
        self.distill_class_embed = nn.Linear(args.hidden_dim, args.num_class + 1) 
        self.distill_bbox_embed = MLP(args.hidden_dim, args.hidden_dim, 4, 3)

        # Number of Queries
        self.num_query = args.num_query
        self.num_distill_query = args.num_distill_query

    def forward(self, x) -> Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]:
        """
        Params:
            - x: [batch_size, 3, image_height, image_width]

        Returns:   
            - detection: dictionary of detection prediction (class, bbox, aux)
            - distillation: dictionary of distillation prediction (class, bbox, aux)
        """
        # Backbone & Dimension Reduction
        forward_features = {}
        def hook_fn(module, input, output):
            forward_features['after_norm'] = output # forward features (except avgpool and flatten)
        hook = self.swin_transformer_backbone.norm.register_forward_hook(hook_fn)
        
        _ = self.swin_transformer_backbone(x)
        hook.remove()

        feat = forward_features['after_norm'] # [B, L, C]
        features = self.reduction(feat) # [B, L, C] -> [B, L, 256]
        features = features.permute(0, 2, 1).reshape( # [B, L, 256] -> [B, 256, H=7, W=7]
            features.shape[0],
            features.shape[2],
            int(features.shape[1]**0.5),
            int(features.shape[1]**0.5)
        )

        # Position Embedding
        (pos, mask) = self.position_embedding(features)

        # Encoder-Decoder
        out = self.transformer_encoder_decoder(
            features,
            mask,
            self.query_embed.weight,
            self.distill_query_embed.weight,
            pos
        )
        det_out = out[:, :, :self.num_query, :] # detection output
        distill_out = out[:, :, self.num_query:, :] # distillation output

        # Detection Heads
        out_cls_logits = self.class_embed(det_out) # [num_decoder_layer, batch_size, num_query, num_class + 1]
        out_bbox_preds = self.bbox_embed(det_out).sigmoid() # [num_decoder_layer, batch_size, num_query, 4]
        # Distillation Heads
        distill_out_cls_logits = self.distill_class_embed(distill_out) # [num_decoder_layer, batch_size, num_distill_query, num_class + 1]
        distill_out_bbox_preds = self.distill_bbox_embed(distill_out).sigmoid() # [num_decoder_layer, batch_size, num_distill_query, 4]

        detection = {
            'class': out_cls_logits[-1], # [batch_size, num_query, num_class + 1]
            'bbox': out_bbox_preds[-1],  # [batch_size, num_query, 4]
            'aux': [{'class': c, 'bbox': b} for c, b in zip(out_cls_logits[:-1], out_bbox_preds[:-1])]
        }

        distillation = {
            'class': distill_out_cls_logits[-1], # [batch_size, num_distill_query, num_class + 1]
            'bbox': distill_out_bbox_preds[-1],  # [batch_size, num_distill_query, 4]
            'aux': [{'class': c, 'bbox': b} for c, b in zip(distill_out_cls_logits[:-1], distill_out_bbox_preds[:-1])]
        }

        if self.training:
            outputs = detection, distillation
        else:
            outputs = detection
            # if self.num_distill_query == 0:
            #     return detection
            # outputs = {
            #     'class': (detection['class'] + distillation['class']) / 2, # [batch_size, num_query, num_class + 1]
            #     'bbox': (detection['bbox'] + distillation['bbox']) / 2,   # [batch_size, num_query, 4]
            #     'aux': [
            #         {
            #             'class': (detection['aux'][i]['class'] + distillation['aux'][i]['class']) / 2,
            #             'bbox': (detection['aux'][i]['bbox'] + distillation['aux'][i]['bbox']) / 2
            #         }
            #         for i in range(len(detection['aux']))
            #     ]
            # }

        return outputs
    




# ## Test!! (python3 -m models.swin_detr -> 명령어 터미널에서 칠 때 models 폴더 바깥에서 해야 함)
# if __name__ == "__main__":
#     from types import SimpleNamespace
#     import yaml
    

#     swin_version = "large" # "tiny", "small", "base", "large"
#     swin_config_path = f"./models/backbone/swin_configs/swin_{swin_version}_patch4_window7_224_22k.yaml"
#     swin_weights_path = f"./models/backbone/swin_weights/swin_{swin_version}_patch4_window7_224_22k.pth"
    
#     with open(f"./models/backbone/swin_configs/swin_{swin_version}_patch4_window7_224_22k.yaml", 'r') as f:
#         config = yaml.safe_load(f)
#     swin_config = config['MODEL']['SWIN']

#     args = SimpleNamespace(
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#         # Swin Transformer Parameters
#         swin_weights_path = swin_weights_path,
#         swin_embed_dim = swin_config['EMBED_DIM'],
#         swin_depths=swin_config['DEPTHS'],
#         swin_num_heads=swin_config['NUM_HEADS'],
#         swin_window_size=swin_config['WINDOW_SIZE'],
#         swin_drop_path_rate=config['MODEL'].get('DROP_PATH_RATE'),
#         swin_out_embed_dim = swin_config['EMBED_DIM'] * 8, # 768, 768, 1024, 1536
#         # DETR Parameters
#         hidden_dim = 256,
#         num_head = 8,
#         num_encoder_layer = 6,
#         num_decoder_layer = 6,
#         dim_feedforward = 2048,
#         dropout = 0.1,
#         num_query = 100,
#         num_distill_query = 100,
#         num_class = 80 # COCO dataset 90 + 1 no-object
#     )

#     input_tensor = torch.randn(1, 3, 224, 224)  # Dummy input tensor for testing

#     model = SwinDETR(args).to(args.device)

#     model.train()
#     # model.eval()
#     outputs = model(input_tensor)

#     # 결과 확인
#     if isinstance(outputs, tuple): # train
#         det, distill = outputs
#         print("Detection output shapes:")
#         print("  class:", det["class"].shape)   # [batch_size, num_query, num_class+1]
#         print("  bbox: ", det["bbox"].shape)    # [batch_size, num_query, 4]
#         print("Distillation output shapes:")
#         print("  class:", distill["class"].shape)
#         print("  bbox: ", distill["bbox"].shape)
#     else: # eval
#         print("Output shapes:")
#         print("  class:", outputs["class"].shape)
#         print("  bbox: ", outputs["bbox"].shape)

#     # 모델 구조 확인
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total parameters: {total_params}")
#     print(f"Trainable parameters: {trainable_params}")





# class SwinDETRWrapper(nn.Module):
#     def __init__(self, model, post_process):
#         super().__init__()
#         self.swin_detr = model
#         self.post_process = post_process

#     def forward(self, x, img_size) -> Tuple[Tensor, Tensor, Tensor]:
#         """
#         Params:
#             - x: batch images of shape [batch_size, 3, args.target_height, args.target_width] where batch_size equals to 1
#                  (if tensor with batch_size larger than 1 is passed in, only the first image prediction will be returned)
#             - img_size: tensor of shape [batch_size, img_width, img_height]

#         Returns:
#             the first image prediction in the following order: scores, labels, boxes
#         """
#         out = self.swin_detr(x)
#         out = self.post_process(out, img_size)[0]
#         return out['scores'], out['labels'], out['boxes']
    

# class PostProcess(nn.Module):
#     def __init__(self):
#         super().__init__()

#     @torch.no_grad()
#     def forward(self, x, img_size) -> List[Dict[str, Tensor]]:
#         logits, bboxes = x['class'], x['bbox']
#         prob = F.softmax(logits, -1)
#         scores, labels = prob[..., :-1].max(-1)

#         img_w, img_h = img_size.unbind(1)
#         scale = torch.stack([img_w, img_h, img_w, img_h], 1).unsqueeze(1)
#         boxes = box_cxcywh_to_xyxy(bboxes)
#         boxes *= scale

#         return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]


# @torch.no_grad()
# def build_inference_model(args, quantize=False):
#     assert os.path.exists(args.weight), 'inference model should have pre-trained weight'
#     device = torch.device(args.device)

#     model = SwinDETR(args).to(device)
#     model.load_state_dict(torch.load(args.weight, map_location=device))

#     post_process = PostProcess.to(device)

#     wrapper = SwinDETRWrapper(model, post_process).to(device)
#     wrapper.eval()

#     if quantize:
#         wrapper = quantize_dynamic(wrapper, {nn.Linear})

#     print('optimizing model for inference...')
#     return torch.jit.trace(wrapper, (torch.rand(1, 3, args.target_height, args.target_width).to(device),
#                                      torch.as_tensor([args.target_width, args.target_height]).unsqueeze(0).to(device)))