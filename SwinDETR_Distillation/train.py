""" Train SwinDETR with Knowledge Distillation (Distillation Query) """


import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader

from torchvision.models.detection import fasterrcnn_resnet50_fpn

import numpy as np
from argparse import ArgumentParser

import os
import yaml

from models.swin_detr import SwinDETR
from models.hungarian_loss import HungarianLoss
from models.distillation_loss import HardDistillationLoss
from utils.misc import MetricsLogger, log_metrics
from datasets.dataset import COCODataset, collate_function


def get_args_parser():
    parser = ArgumentParser('SwinDETR', add_help=False)

    # Swin Transformer Parameters
    swin_version = "base" # "tiny", "small", "base", "large"
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


def main(args):
    # Load Datasets
    dataset = COCODataset(args.data_dir, args.ann_file, args.target_height, args.target_width, args.num_class)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=collate_function,
                            pin_memory=True,
                            num_workers=args.num_workers)
    batches = len(dataloader)
    
    # Load Model & Criterion
    student_model = SwinDETR(args).to(args.device)
    if args.is_KD:
        teacher_model = fasterrcnn_resnet50_fpn(pretrained=True).to(args.device)
        criterion = HardDistillationLoss(teacher=teacher_model, args=args).to(args.device)
    else:
        criterion = HungarianLoss(args).to(args.device)

    student_model.train()
    criterion.train()
    if args.is_KD:
        teacher_model.eval()

    # Resume Training
    weight_pth = f'{args.pth_dir}/{args.task_name}.pth'
    if os.path.exists(weight_pth):
        print(f'loading pre-trained weights from {weight_pth}')
        student_model.load_state_dict(torch.load(weight_pth, map_location=args.device))

    # Optimizer & Scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    lr_scheduler = StepLR(optimizer, args.lr_drop)
    prev_best_loss = np.inf

    # Logger
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = MetricsLogger(folder=args.log_dir)

    # Check Model Parameters
    if args.is_KD:
        teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"Total parameters (teacher): {teacher_total_params}")
    student_total_params = sum(p.numel() for p in student_model.parameters())
    student_trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"Total parameters (student): {student_total_params}")
    print(f"Trainable parameters (student): {student_trainable_params}")

    # Training Loop
    for epoch in range(args.epochs):
        losses = []

        for batch, (x, y) in enumerate(dataloader):
            x = x.to(args.device)
            y = [{k: v.to(args.device) for k, v in t.items()} for t in y]

            # Forward Pass
            out = student_model(x)
            if args.is_KD:
                metrics = criterion(x, out, y)  # type(metrics):
                                                # dict('classification loss', 'bbox loss', 'giou loss', 'mAP',
                                                #      'classification loss_aux0', 'bbox loss_aux0', 'giou loss_aux0', 'mAP_aux0',
                                                #       ...,
                                                #       'classification loss_aux4', 'bbox loss_aux4', 'giou loss_aux4', 'mAP_aux4')
            else:
                metrics = criterion(out[0], y)
            loss = sum(v for k, v in metrics.items() if 'loss' in k)
            losses.append(loss.cpu().item())

            # Print & Save Training Details
            print(f'Epoch {epoch} | {batch + 1} / {batches}')
            log_metrics({k: v for k, v in metrics.items() if 'aux' not in k})
            logger.step(metrics, epoch, batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.clip_max_norm)
            optimizer.step()

        # Learning Rate Scheduling
        lr_scheduler.step()
        
        # Log Epoch End
        logger.epoch_end(epoch)
        
        # Print Average Loss
        avg_loss = np.mean(losses)
        print(f'Epoch {epoch}, loss: {avg_loss:.8f}')

        # Save Best Model
        if avg_loss < prev_best_loss:
            print('[+] Loss improved from {:.8f} to {:.8f}, saving model...'.format(prev_best_loss, avg_loss))
            prev_best_loss = avg_loss
            
            if not os.path.exists(args.pth_dir):
                os.makedirs(args.pth_dir)
            try:
                state_dict = student_model.module.state_dict()
            except AttributeError:
                state_dict = student_model.state_dict()
            torch.save(state_dict, f'{args.pth_dir}/{args.task_name}.pth')

            logger.add_scalar('Model', avg_loss, epoch)

        logger.flush()

    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser('Train SwinDETR with Knowledge Distillation (Distillation Query)', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)