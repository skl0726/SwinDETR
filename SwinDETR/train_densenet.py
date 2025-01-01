""" Train Model (DenseNet Based DETR) """


import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader

import numpy as np
from argparse import ArgumentParser

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models import SwinDETR, SetCriterion
from args import base_parser
from datasets.dataset import COCODataset, collate_function
from utils.misc import MetricsLogger, save_arguments, log_metrics


import gc
gc.collect()
torch.cuda.empty_cache()


def main(args):
    print(args)
    save_arguments(args, args.task_name)

    #torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # load data
    dataset = COCODataset(args.data_dir, args.ann_file, args.target_height, args.target_width, args.num_class)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=collate_function,
                            pin_memory=True,
                            num_workers=args.num_workers)

    # load model
    model = SwinDETR(args).to(device)
    criterion = SetCriterion(args).to(device)

    # resume training
    if args.weight and os.path.exists(args.weight):
        print(f'loading pre-trained weights from {args.weight}')
        model.load_state_dict(torch.load(args.weight, map_location=device))

    # multi-GPU training
    if args.multi:
        model = torch.nn.DataParallel(model)

    # separate learning rate
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone
        }
    ]

    optimizer = AdamW(param_dicts, args.lr, weight_decay = args.weight_decay)
    lr_scheduler = StepLR(optimizer, args.lr_drop)
    prev_best_loss = np.inf
    batches = len(dataloader)
    logger = MetricsLogger()

    model.train()
    criterion.train()

    for epoch in range(args.epochs):
        losses = []
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = [{k: v.to(device) for k, v in t.items()} for t in y]

            out = model(x)

            metrics = criterion(out, y)

            loss = sum(v for k, v in metrics.items() if 'loss' in k)
            losses.append(loss.cpu().item())

            # print & save training details
            print(f'Epoch {epoch} | {batch + 1} / {batches}')
            log_metrics({k: v for k, v in metrics.items() if 'aux' not in k})
            logger.step(metrics, epoch, batch)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

        lr_scheduler.step()
        logger.epoch_end(epoch)
        avg_loss = np.mean(losses)
        print(f'Epoch {epoch}, loss: {avg_loss:.8f}')

        if avg_loss < prev_best_loss:
            print('[+] Loss improved from {:.8f} to {:.8f}, saving model...'.format(prev_best_loss, avg_loss))
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)

            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()

            torch.save(state_dict, f'{args.output_dir}/{args.task_name}.pt')
            prev_best_loss = avg_loss
            logger.add_scalar('Model', avg_loss, epoch)

        logger.flush()

    logger.close()

    
if __name__ == '__main__':
    parser = ArgumentParser('train_densenet.py', parents=[base_parser()])

    parser.add_argument('--swin_transformer', default='densenet', type=str) # densenet based model training
    parser.add_argument('--weight', default='./checkpoint/coco_densenet.pt', type=str)
    parser.add_argument('--hidden_dim', default=768, type=int)

    # training config
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int) # default: 1500
    parser.add_argument('--lr_drop', default=1000, type=int)
    parser.add_argument('--clip_max_norm', default=.1, type=float)

    # dataset
    parser.add_argument('--data_dir', default='./coco/train2017', type=str)
    parser.add_argument('--ann_file', default='./coco/annotations/instances_train2017.json', type=str)

    # miscellaneous
    parser.add_argument('--output_dir', default='./checkpoint', type=str)
    parser.add_argument('--task_name', default='coco_densenet', type=str) # densenet based model training
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--multi', default=True, type=bool)

    main(parser.parse_args())