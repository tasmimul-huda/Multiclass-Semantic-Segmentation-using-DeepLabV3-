import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import config
from data_loader import CIHPDataLoader

from model 
model = DeeplabV3Plus(config.train_images_paths,
                      config.train_masks_paths, 
                      config.val_images_paths,
                      config.val_masks_paths, 
                      config.IMAGE_SIZE, 
                      config.NUM_CLASSES)

# Build the model architecture
model.build_model()
# Compile the model
model.compile_model(learning_rate=0.0001)

history = model.train(epochs=2)