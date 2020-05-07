import os
import sys
import imgaug
import helpers

# Root directory of the project
ROOT_DIR = os.path.abspath("../Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib, utils


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

if __name__ == '__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=helpers.DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)
    print("Image: ", args.image)
    print("Video: ", args.video)

    # Configurations
    if args.command == "train":
        config = helpers.PeopleConfig()
    else:
        class InferenceConfig(helpers.PeopleConfig):
            BATCH_SIZE = 1
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
        model_path = COCO_WEIGHTS_PATH
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = helpers.PeopleDataset()
        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download, class_ids=[1])
        if args.year in '2014':
            dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download, class_ids=[1])
        dataset_train.prepare()

        # Validation dataset
        dataset_val = helpers.PeopleDataset()
        val_type = "val" if args.year in '2017' else "minival"
        dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download, class_ids=[1])
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=10,
                    layers='heads',
                    augmentation=augmentation)

    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
        model_path = model.find_last()
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

        if args.image:
            helpers.save_image(args.image, lambda image: model.detect([image], verbose=0))
        elif args.video:
            helpers.save_video(args.video, lambda image: model.detect([image], verbose=0)[0])
        else:
            print("You must provide either --image or --video commandline parameter when using inference mode")
