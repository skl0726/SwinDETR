""" Inference - input: one image """


import torch
import torchvision.transforms as T

from argparse import ArgumentParser
from PIL import Image
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models import SwinDETR
from args import base_parser
from utils.box_ops import rescale_bboxes


def inference_model(args):
    model = SwinDETR(args)
    
    # Load pretrained weights
    assert args.weight and os.path.exists(args.weight), "Model weight file must be specified and exist"
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))

    model.eval()

    return model


def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.8+ confidence
    probas = outputs['class'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.8

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['bbox'][0, keep], im.size)
    #return probas[keep], bboxes_scaled
    
    # Keep only the boxes and probabilities of the desired classes
    filtered_probas = []
    filtered_boxes = []
    for i, p in enumerate(probas[keep]):
        cl = p.argmax()
        if cl < len(CLASSES) and CLASSES[cl] in ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"]:
            filtered_probas.append(p)
            filtered_boxes.append(bboxes_scaled[i])
    
    return filtered_probas, filtered_boxes


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes, COLORS * 100): # default: boxes.tolist()
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def inference(args):
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    im = Image.open(args.image_dir).convert("RGB")

    scores, boxes = detect(im, inference_model(args), transform)
        
    plot_results(im, scores, boxes)


if __name__ == '__main__':
    parser = ArgumentParser('main.py', parents=[base_parser()])

    parser.add_argument('--swin_transformer', default='swin_t', type=str)
    parser.add_argument('--weight', default='./checkpoint/coco_swin_t.pt', type=str)
    parser.add_argument('--hidden_dim', default=768, type=int)

    parser.add_argument('--image_dir', default='./images/11.jpg', type=str)

    # COCO 2017 classes
    CLASSES = [ "person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"]
    """
    CLASSES = [
        "person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign",
        "train", "airplane", "boat", "fire hydrant", "parking meter", "bench", "bird", "cat", 
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", 
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
        "toothbrush"
    ]
    """

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], 
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    torch.set_grad_enabled(False)
    inference(parser.parse_args())