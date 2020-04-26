"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
from mrcnn.config import Config
from mrcnn import utils

import os
import sys
import math
import random
import numpy as np
import cv2
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN


sys.path.append(ROOT_DIR)  # To find local version of the library


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "building"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 2 building types

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1000
    IMAGE_MAX_DIM = 2000

    # Use smaller anchors because our image and objects are small


#     RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

# Reduce training ROIs per image because the images are small and have
# few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
#     TRAIN_ROIS_PER_IMAGE = 32

# Use a small epoch since the data is simple
#     STEPS_PER_EPOCH = 100

# use small validation steps since the epoch is small
#     VALIDATION_STEPS = 5


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("building", 1, "commercial")
        self.add_class("building", 2, "house")

        dataset_dir = os.path.join("..", "..", "datasets", "building", "train")

        for image in os.listdir(dataset_dir):
            self.add_image("building",
                           image_id=image,
                           path=os.path.join(dataset_dir, image, "image.png"))

    def load_mask(self, image_id):
        np.set_printoptions(threshold=99)
        """Generate instance masks for shapes of the given image ID.
        """
        masks_made = False
        masks = []
        class_ids = []
        mask_dir = os.path.join("..", "..", "datasets", "building", "train", self.image_info[image_id]['id'], "masks")
        classes = os.listdir(mask_dir)
        class_dirs = list(map(lambda x: os.path.join(mask_dir, x), classes))
        for class_dir, label in zip(class_dirs, classes):
            mask_instance_path = os.listdir(class_dir)
            mask_instance_path = list(map(lambda x: os.path.join(class_dir, x), mask_instance_path))
            for mask in mask_instance_path:
                img = skimage.io.imread(mask, as_gray=True)
                masks.append(img)
                class_ids.append(self.class_names.index(label))

        masks = np.array(masks).astype(np.bool)
        masks = np.moveaxis(masks, 0, -1)
        return masks, np.array(class_ids, dtype=np.int32)




building = ShapesDataset()
building.load_shapes()
building.prepare()
building.load_mask(0)
