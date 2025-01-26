# SwinDETR

## Abstract
This study proposes a novel Real-Time Object Detection model to enhance pedestrian safety at the three-way intersection in front of the Cheongam-ro CHANGE-UP GROUND. Cheongam-ro is a major road frequently used by both university members and external vehicles. Due to the high traffic volume and fast vehicle speeds, it creates a challenging environment for pedestrians to cross safely at the crosswalk. To address this issue, this study designed the 'SwinDETR' model, which uses the Swin Transformer as the backbone of DETR, to improve object detection performance in road environments.<br/>
The study identified the limitations of the original DETR, particularly its lack of performance in detecting small objects, attributing this to constraints in feature map resolution and the global processing nature of the Transformer. SwinDETR compensates for these limitations, demonstrating superior small object detection capabilities. Performance comparisons between SwinDETR and the conventional CNN-based DETR showed that SwinDETR outperformed DETR in evaluation time and inference tests using images of Cheongam-ro and campus roads, particularly for small object detection.<br/>
Based on the SwinDETR model, the study developed a Real-Time Object Detection system capable of measuring not only object classes and bounding boxes but also object speeds and distances through software. This model was further applied to analyze traffic conditions and implement a traffic information system. Two hours of traffic footage from Cheongam-ro were recorded and analyzed using Real-Time Object Detection to collect traffic data and test the traffic information system, successfully verifying its practicality.<br/><br/>
The main contributions of this study are as follows:
- Designing the 'SwinDETR' model to improve small object detection performance and introducing a novel Object Detection Architecture (Transformer-Transformer).
- Collecting cumulative traffic data based on Real-Time Object Detection, enabling effective alternatives for pedestrians to avoid road congestion and traffic accident risks based on time-specific traffic statistics.
- Implementing a traffic information system around the Changam-ro crosswalk, allowing pedestrians to directly assess road conditions:
    - Expansion of the system to major campus roads.
    - Provision of traffic accident prevention measures for not only vehicles but also small vehicles (motorcycles, scooters, bicycles, etc.).

## Model Overview

### Description
Hello

### Model Structure
<img src="">

### Inference Test
- Left: SwinDETR Image Inference / Right: DETR Image Inference
<br/>
<figure class="half">
    <a href="link"><img src="./Inference-Image/SwinDETR1.png" width="45%"></a>
    <a href="link"><img src="./Inference-Image/DETR1.png" width="45%"></a>
</figure>
<figure class="half">
    <a href="link"><img src="./Inference-Image/SwinDETR2.png" width="45%"></a>
    <a href="link"><img src="./Inference-Image/DETR2.png" width="45%"></a>
</figure>
<figure class="half">
    <a href="link"><img src="./Inference-Image/SwinDETR3.png" width="45%"></a>
    <a href="link"><img src="./Inference-Image/DETR3.png" width="45%"></a>
</figure>
<br/>
SwinDETR captures detailed local information (small objects) more effectively and performs well on detection than DETR.
Compared to DETR, SwinDETR is less likely to misjudge that there are objects in the background around small objects, and less likely to misjudge the class of small objects.

## Environment Setup

### Directory Structure
```
└── SwinDETR
    ├── args.py
    ├── coco
    │   ├── annotations
    │   │   ├── instances_train2017.json
    │   │   └── instances_val2017.json
    │   ├── train2017
    │   └── val2017
    ├── csv_analysis.py
    ├── datasets
    │   ├── __init__.py
    │   ├── dataset.py
    │   └── transforms.py
    ├── eval_densenet.py
    ├── eval_swin_t.py
    ├── inference_image.py
    ├── inference_video.py
    ├── inference_video_with_alert.py
    ├── models
    │   ├── __init__.py
    │   ├── backbone.py
    │   ├── criterion.py
    │   ├── densenet.py
    │   ├── matcher.py
    │   ├── positional_encoding.py
    │   ├── swin_detr.py
    │   ├── swin_transformer.py
    │   └── transformer.py
    ├── train_densenet.py
    ├── train_densenet.sh
    ├── train_swin_t.py
    ├── train_swin_t.sh
    └── utils
        ├── __init__.py
        ├── box_ops.py
        └── misc.py

9 directories, 28 files
```
- Train: train_swin_t.py / train_densenet.py<br/>
- Evaluation: eval_swin_t.py / eval_densenet.py<br/>
- Inference: inference_image.py / inference_video.py / inference_video_with_alert.py<br/>

### Needed Libraries
```
pip install torch torchvision opencv-python scipy einops matplotlib pycocotools tensorboard
```

### Dataset
In this study, the COCO 2017 Dataset was utilized to train the SwinDETR model for the Object Detection Task. The COCO 2017 Dataset occupies approximately 37GB of storage, including all images and annotations. It consists of over 330,000 images (Train Set: ~118,000, Validation Set: ~5,000, Test Set: ~41,000, Unlabeled Data: ~123,000) and includes a total of 80 classes, such as humans, animals, vehicles, and household appliances, along with approximately 1.5 million object annotations. These annotations provide detailed information for each object, such as bounding boxes.
- Link: [COCO 2017 Dataset](https://cocodataset.org/#home)<br/>
