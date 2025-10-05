""""" Train SwinDETR with Knowledge Distillation (Distillation Query) using DDP """


import os
import yaml
import argparse
import numpy as np
from argparse import ArgumentParser

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from models.swin_detr import SwinDETR
from models.hungarian_loss import HungarianLoss
from models.distillation_loss import HardDistillationLoss
from utils.misc import MetricsLogger, log_metrics
from datasets.dataset import COCODataset, collate_function


def get_args_parser():
    parser = ArgumentParser('SwinDETR', add_help=False)

    # Swin Transformer Parameters
    swin_version = "base" # # "tiny", "small", "base", "large"
    swin_config_path = f"./models/backbone/swin_configs/swin_{swin_version}_patch4_window7_224_22k.yaml"
    swin_weights_path = f"./models/backbone/swin_weights/swin_{swin_version}_patch4_window7_224_22k.pth"
    
    with open(swin_config_path, 'r') as f:
        config = yaml.safe_load(f)
    swin_config = config['MODEL']['SWIN']

    parser.add_argument('--swin_weights_path', default=swin_weights_path, type=str)
    parser.add_argument('--swin_embed_dim', default=swin_config['EMBED_DIM'], type=int)
    parser.add_argument('--swin_depths', default=swin_config['DEPTHS'], type=list)
    parser.add_argument('--swin_num_heads', default=swin_config['NUM_HEADS'], type=list)
    parser.add_argument('--swin_window_size', default=swin_config['WINDOW_SIZE'], type=int)
    parser.add_argument('--swin_drop_path_rate', default=config['MODEL'].get('DROP_PATH_RATE'), type=float)
    parser.add_argument('--swin_out_embed_dim', default=swin_config['EMBED_DIM'] * 8, type=int) # tiny -> 768, small -> 768, base -> 1024, large -> 1536

    # DETR Encoder-Decoder Parameters
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_head', default=8, type=int)
    parser.add_argument('--num_encoder_layer', default=6, type=int)
    parser.add_argument('--num_decoder_layer', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=.1, type=float)
    parser.add_argument('--num_query', default=100, type=int)
    parser.add_argument('--num_distill_query', default=100, type=int) # 100, 50, 25
    parser.add_argument('--num_class', default=90, type=int) # COCO Dataset 90 classes + 1 background

    # Loss (Hungarian Matcher, Hungarian Loss, Hard Distillation Loss using Hungarian Loss)
    parser.add_argument('--is_KD', default=1, type=int) # 1: True, 0: False
    parser.add_argument('--class_cost_weight', default=1., type=float)
    parser.add_argument('--bbox_cost_weight', default=5., type=float)
    parser.add_argument('--giou_cost_weight', default=2., type=float)
    parser.add_argument('--eos_cost_weight', default=.1, type=float)
    parser.add_argument('--alpha', default=.5, type=float)
    parser.add_argument('--score_thresh', default=.5, type=float)

    # Dataset
    parser.add_argument('--target_height', default=224, type=int)
    parser.add_argument('--target_width', default=224, type=int)
    parser.add_argument('--data_dir', default='./coco/train2017', type=str)
    parser.add_argument('--ann_file', default='./coco/annotations/instances_train2017.json', type=str)

    # Training Config
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=24, type=int) # default:100
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1500, type=int)
    parser.add_argument('--lr_drop', default=1000, type=int)
    parser.add_argument('--clip_max_norm', default=.1, type=float)

    # Miscellaneous
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--log_dir', default='./logs', type=str)

    parser.add_argument('--pth_dir', default='./checkpoint', type=str)
    parser.add_argument('--task_name', default='swindetr_hdistill100', type=str) # swindetr / hdistill, sdistill / 100, 50, 25

    return parser


def setup_distributed(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


def train_ddp(rank, world_size, args):
    # initialize distributed environment
    setup_distributed(rank, world_size)

    # set device
    device = torch.device(f'cuda:{rank}')

    # load dataset and sampler
    dataset = COCODataset(args.data_dir, args.ann_file, args.target_height, args.target_width, args.num_class)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            collate_fn=collate_function,
                            pin_memory=True,
                            num_workers=args.num_workers)

    # build models
    student_model = SwinDETR(args).to(device)
    if args.is_KD:
        teacher_model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
        teacher_model.eval()
        criterion = HardDistillationLoss(teacher=teacher_model, args=args).to(device)
    else:
        criterion = HungarianLoss(args).to(device)

    # wrap student with DDP
    student_model = DDP(student_model, device_ids=[rank])

    # optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, student_model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = StepLR(optimizer, args.lr_drop)

    # logging
    logger = MetricsLogger(folder=args.log_dir) if rank == 0 else None
    prev_best_loss = np.inf

    # resume
    weight_pth = f'{args.pth_dir}/{args.task_name}.pth'
    if os.path.exists(weight_pth):
        map_loc = {'cuda:%d' % 0: 'cuda:%d' % rank}
        # student_model.load_state_dict(torch.load(weight_pth, map_location=map_loc))
        student_model.module.load_state_dict(torch.load(weight_pth, map_location=map_loc))
        print(f'loading pre-trained weights from {weight_pth}')

    # training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        losses = []

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = [{k: v.to(device) for k, v in t.items()} for t in y]

            out = student_model(x)
            if args.is_KD:
                metrics = criterion(x, out, y)
            else:
                metrics = criterion(out[0], y)

            loss = sum(v for k, v in metrics.items() if 'loss' in k)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.clip_max_norm)
            optimizer.step()

            if rank == 0:
                print(f'Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] loss: {loss.item():.4f}')
                log_metrics({k: v for k,v in metrics.items() if 'aux' not in k})
                logger.step(metrics, epoch, batch_idx)

        lr_scheduler.step()
        avg_loss = np.mean(losses)

        if rank == 0:
            print(f'Epoch {epoch} avg_loss: {avg_loss:.4f}')
            logger.epoch_end(epoch)
            if avg_loss < prev_best_loss:
                print(f'[+] Loss improved ({prev_best_loss:.4f} -> {avg_loss:.4f}), saving model.')
                prev_best_loss = avg_loss
                os.makedirs(args.pth_dir, exist_ok=True)
                torch.save(student_model.module.state_dict(), f'{args.pth_dir}/{args.task_name}.pth')
                logger.add_scalar('Model', avg_loss, epoch)
            logger.flush()

    if rank == 0:
        logger.close()
    cleanup_distributed()


def main():
    parser = ArgumentParser('Train SwinDETR with Knowledge Distillation (Distillation Query)', parents=[get_args_parser()])
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()