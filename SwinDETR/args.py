""" Args used in SwinDETR """


import torch

from argparse import ArgumentParser


def base_parser() -> ArgumentParser:
    parser = ArgumentParser('SwinDETR', add_help=False)

    # model parameters - backbone
    parser.add_argument('--num_groups', default=8, type=int)
    parser.add_argument('--growth_rate', default=32, type=int)
    parser.add_argument('--num_blocks', default=[6]*4, type=list)
    #parser.add_argument('--swin_transformer', default=True, type=bool)

    # model parameters - transformer (for DETR)
    #parser.add_argument('--hidden_dim', default=768, type=int) # dim of output of swin transformer (swin_t): 768 / swin_transformer (swin_b): 1024 / default (densenet): 512
    parser.add_argument('--num_head', default=8, type=int)
    parser.add_argument('--num_encoder_layer', default=6, type=int)
    parser.add_argument('--num_decoder_layer', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=.1, type=float)
    parser.add_argument('--num_query', default=30, type=int)
    parser.add_argument('--num_class', default=80, type=int)

    # dataset
    parser.add_argument('--target_height', default=224, type=int)
    parser.add_argument('--target_width', default=224, type=int)

    # loss
    parser.add_argument('--class_cost', default=2., type=float)
    parser.add_argument('--bbox_cost', default=5., type=float)
    parser.add_argument('--giou_cost', default=5., type=float)
    parser.add_argument('--eos_cost', default=.1, type=float)

    # miscellaneous
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    #parser.add_argument('--weight', default='./checkpoint/coco.pt', type=str)
    parser.add_argument('--seed', default=0, type=int)

    return parser