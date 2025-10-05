import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


COCO_ROOT = './coco'
IMG_DIR = os.path.join(COCO_ROOT, 'val2017')
ANN_FILE = os.path.join(COCO_ROOT, 'annotations', 'instances_val2017.json')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


class CocoDataset(CocoDetection):
    def __init__(self, img_dir, ann_file, transform=None):
        super().__init__(img_dir, ann_file, transforms=None)
        self.transform = transform

    def __getitem__(self, idx):
        img, anns = super(CocoDataset, self).__getitem__(idx)
        orig_w, orig_h = img.size
        img_id = self.ids[idx]

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'orig_size': (orig_w, orig_h)
        }

        if self.transform:
            img = self.transform(img)
        return img, target


def main():
    dataset = CocoDataset(IMG_DIR, ANN_FILE, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()

    coco_gt = COCO(ANN_FILE)

    results = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Inferencing", total=len(loader)):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for tgt, output in zip(targets, outputs):
                img_id = tgt['image_id'].item()
                orig_w, orig_h = tgt['orig_size']

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                scale_x = orig_w / 224.0
                scale_y = orig_h / 224.0

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    
                    x1_o = x1 * scale_x
                    y1_o = y1 * scale_y
                    w_o = (x2 - x1) * scale_x
                    h_o = (y2 - y1) * scale_y

                    results.append({
                        'image_id': img_id,
                        'category_id': int(label),
                        'bbox': [float(x1_o), float(y1_o), float(w_o), float(h_o)],
                        'score': float(score)
                    })

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    # coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
    # coco_eval.params.maxDets = [1, 10, 100]
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
    main()