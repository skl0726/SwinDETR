""" Backbone Building """


from torch import nn, Tensor

from typing import Tuple

from models.positional_encoding import PositionEmbeddingSine
from models.swin_transformer import swin_t_for_detection, swin_b_for_detection
from models.densenet import DenseNet
            

class Joiner(nn.Module):
    def __init__(self, backbone, position_embedding: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding

    def forward(self, x) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        features = self.backbone(x)
        return features, self.position_embedding(features)
    

def build_backbone(args):
    position_embedding = PositionEmbeddingSine(args.hidden_dim // 2)
    if args.swin_transformer == 'swin_t':
        backbone = swin_t_for_detection() # swin transformer (swin_t) backbone
    elif args.swin_transformer == 'swin_b':
        backbone = swin_b_for_detection() # swin transformer (swin_b) backbone
    else:
        backbone = DenseNet(args.num_groups, args.growth_rate, args.num_blocks)
        
    return Joiner(backbone, position_embedding)