import os
import cv2
import numpy as np
from glob import glob

class config:
    IMAGE_SIZE = 512
    BATCH_SIZE = 4
    NUM_CLASSES = 20
    DATA_DIR = "D:/DeepLabV3+/instance-level-human-parsing/instance-level_human_parsing/instance-level_human_parsing/Training"
    NUM_TRAIN_IMAGES = 1000
    NUM_VAL_IMAGES = 50
    train_images_paths = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[:NUM_TRAIN_IMAGES]
    train_masks_paths = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[:NUM_TRAIN_IMAGES]

    val_images_paths = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[
        NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES]
    val_masks_paths = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES]