#!/bin/bash

# in terminal, run this script to download the COCO 2017 dataset.
# brew install wget
# chmod +x download_coco_2017.sh
# ./download_coco_2017.sh

mkdir -p coco
cd coco

# train2017
wget http://images.cocodataset.org/zips/train2017.zip
unzip -q train2017.zip
rm train2017.zip

# val2017
wget http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip
rm val2017.zip

# annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip
rm annotations_trainval2017.zip

# rename annotations_trainval2017 to annotations
mv annotations_trainval2017 annotations

# REMOVE unwanted annotation files: captions and person_keypoints
rm annotations/captions_*.json
rm annotations/person_keypoints_*.json

echo "COCO dataset download and extraction completed!"