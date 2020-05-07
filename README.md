This is a basic segmentation script for segmenting people.

It is trained using the MS-COCO Dataset and the code for the Mask-RCNN is taken from https://github.com/matterport/Mask_RCNN

for training use: `python3 ./app.py train --dataset ./coco_dataset --download true`
for inference use `python3 ./app.py inference --dataset ./coco_dataset --image <path_to_image>`

