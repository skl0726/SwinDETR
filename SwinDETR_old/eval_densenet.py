""" Eval Model (DenseNet Based DETR) """


import torch
from torch.utils.data import DataLoader

import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
import time

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models import SwinDETR, SetCriterion
from args import base_parser
from datasets.dataset import COCODataset, collate_function
from utils.box_ops import box_iou


def evaluate(args):
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load data
    dataset = COCODataset(args.data_dir, args.ann_file, args.target_height, args.target_width, args.num_class, train=False)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=collate_function,
                            pin_memory=True,
                            num_workers=args.num_workers)

    # Load model
    model = SwinDETR(args).to(device)
    criterion = SetCriterion(args).to(device)

    # calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    # calculate only trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params}')

    # Load pretrained weights
    assert args.weight and os.path.exists(args.weight), "Model weight file must be specified and exist"
    model.load_state_dict(torch.load(args.weight, map_location=device))

    model.eval()
    criterion.eval()

    all_mAPs = []

    start_time = time.time()

    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = [{k: v.to(device) for k, v in t.items()} for t in y]

        with torch.no_grad():
            out = model(x)

        results = criterion(out, y)
        all_mAPs.append(results['mAP'].item())

    mAP = np.mean(all_mAPs)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'===== {args.swin_transformer} eval =====')
    print(f'mAP: {mAP:.4f}')
    print(f'evaluation time: {elapsed_time:.2f} seconds')


if __name__ == '__main__':
    parser = ArgumentParser('eval_densenet.py', parents=[base_parser()])

    parser.add_argument('--swin_transformer', default='densenet', type=str)
    parser.add_argument('--weight', default='./checkpoint/coco_densenet.pt', type=str)
    parser.add_argument('--hidden_dim', default=768, type=int)

    # evaluation config
    parser.add_argument('--batch_size', default=8, type=int)

    # dataset
    parser.add_argument('--data_dir', default='./coco/val2017', type=str)
    parser.add_argument('--ann_file', default='./coco/annotations/instances_val2017.json', type=str)

    # miscellaneous
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--task_name', default='coco_densenet_eval', type=str)
    parser.add_argument('--num_workers', default=8, type=int)

    evaluate(parser.parse_args())