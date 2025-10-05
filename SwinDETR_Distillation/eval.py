""" SwinDETR Evaluation """


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import argparse
import os
import yaml
import json
from tqdm import tqdm

from models.swin_detr import SwinDETR
from datasets.dataset import COCODataset, collate_function
from utils.box_ops import box_cxcywh_to_xyxy


def get_args_parser():
    parser = argparse.ArgumentParser('SwinDETR COCO Evaluation', add_help=False)

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

    # SwinDETR Weight Path
    parser.add_argument('--weights_path', default='./checkpoint/swindetr_hdistill100(epoch118).pth', type=str) ### swindetr / swindetr_hdistill

    # DETR Encoder-Decoder Parameters
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--num_encoder_layer', type=int, default=6)
    parser.add_argument('--num_decoder_layer', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_query', type=int, default=100)
    parser.add_argument('--num_distill_query', type=int, default=100) ### swindetr / swindetr_hdistill
    parser.add_argument('--num_class', type=int, default=90) # COCO Dataset 90 classes + 1 background

    # Evaluation settings
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--score_thresh', default=0.5, type=float,
                        help='score threshold to filter detections')

    # COCO dataset
    parser.add_argument('--val_data_dir', default='./coco/val2017', type=str,
                        help='path to COCO val2017 images')
    parser.add_argument('--val_ann_file', default='./coco/annotations/instances_val2017.json', type=str,
                        help='path to COCO annotation json for val')
    parser.add_argument('--target_height', default=224, type=int)
    parser.add_argument('--target_width', default=224, type=int)

    # Output
    parser.add_argument('--output_dir', type=str, default='./eval_AP',
                        help='directory to save results json')
    parser.add_argument('--output_name', type=str, default='results_swindetr.json') ### swindetr / swindetr_hdistill
    
    # Miscellaneous
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    
    return parser


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Build model
    model = SwinDETR(args).to(device)
    # Load trained weights
    ckpt = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # COCO ground truth
    coco_gt = COCO(args.val_ann_file)
    cat_ids = coco_gt.getCatIds()

    # Dataset and DataLoader
    dataset = COCODataset(args.val_data_dir, args.val_ann_file, args.target_height, args.target_width, args.num_class) # num_class / num_class+1 둘 결과 다름
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=collate_function)

    # Keep list of image IDs in order
    image_ids = dataset.ids

    results = []
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)

        logits = outputs['class']  # [B, Q, C+1]
        bboxes = outputs['bbox']   # [B, Q, 4]

        probs = F.softmax(logits, dim=-1)
        scores, labels = probs[..., :-1].max(-1)

        # Determine the subset of image_ids for this batch
        start = batch_idx * args.batch_size
        end = start + images.size(0)
        batch_image_ids = image_ids[start:end]

        for i in range(images.size(0)):
            img_id = batch_image_ids[i]
            # Retrieve original width and height from COCO
            img_info = coco_gt.imgs[img_id]
            orig_w, orig_h = img_info['width'], img_info['height']

            sc = scores[i]
            lb = labels[i]
            bb = bboxes[i]

            keep = sc > args.score_thresh
            sc = sc[keep]; lb = lb[keep]; bb = bb[keep]

            if sc.numel() == 0:
                continue

            # Convert boxes from cxcywh (normalized) to xywh (absolute)
            xyxy = box_cxcywh_to_xyxy(bb)
            xyxy[:, 0] *= orig_w; xyxy[:, 2] *= orig_w
            xyxy[:, 1] *= orig_h; xyxy[:, 3] *= orig_h
            xywh = torch.zeros_like(xyxy)
            xywh[:, 0] = xyxy[:, 0]
            xywh[:, 1] = xyxy[:, 1]
            xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
            xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]

            for score, label, box in zip(sc, lb, xywh):
                results.append({
                    'image_id': img_id,
                    'category_id': coco_gt.getCatIds()[label],
                    'bbox': [box[0].item(), box[1].item(), box[2].item(), box[3].item()],
                    'score': score.item()
                })

    # Save JSON and run COCOeval
    results_path = os.path.join(args.output_dir, args.output_name)
    with open(results_path, 'w') as f:
        json.dump(results, f)
    print(f"Saved results to {results_path}")

    coco_dt = coco_gt.loadRes(results_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    print(f"\nSummary Metrics:")
    print(f"mAP (IoU=0.50:0.95) = {stats[0]:.3f}")
    print(f"AP_small = {stats[3]:.3f}")
    print(f"AP_medium = {stats[4]:.3f}")
    print(f"AP_large = {stats[5]:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation for SwinDETR', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)