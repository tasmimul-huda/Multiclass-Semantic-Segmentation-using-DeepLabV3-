import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from config import config



class CIHPDataLoader:
    def __init__(self, image_paths, mask_paths, img_size=(512, 512), batch_size=32):
        self.img_size = img_size
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.num_samples = len(self.image_paths)

    def _parse_function(self, image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, self.img_size, method=tf.image.ResizeMethod.BILINEAR)
        image = tf.cast(image, tf.float32)
        image /= 255.0

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, self.img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.squeeze(mask)
        return image, mask

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))
        dataset = dataset.shuffle(buffer_size=self.num_samples, reshuffle_each_iteration=True)
        dataset = dataset.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset











    
# class DataLoader:
#     def __init__(self, BATCH_SIZE,IMAGE_SIZE ):
#         # self.image_list = image_list
#         # self.mask_list = mask_list
#         self.BATCH_SIZE = BATCH_SIZE
#         self.IMAGE_SIZE = IMAGE_SIZE
#         # self.data_generator(self.image_list, self.mask_list)
        
#     def read_image(self, image_path, mask=False):
#         image = tf.io.read_file(image_path)
#         if mask:
#             image = tf.image.decode_png(image, channels=1)
#             image.set_shape([None, None, 1])
#             image = tf.image.resize(images=image, size=[self.IMAGE_SIZE, self.IMAGE_SIZE])
#         else:
#             image = tf.image.decode_png(image, channels=3)
#             image.set_shape([None, None, 3])
#             image = tf.image.resize(images=image, size=[self.IMAGE_SIZE, self.IMAGE_SIZE])
#             image = tf.keras.applications.resnet50.preprocess_input(image)
#         return image

#     def load_data(self,image_list, mask_list):
#         image = self.read_image(image_list)
#         mask = self.read_image(mask_list, mask=True)
#         return image, mask

#     def data_generator(self, image_list, mask_list, shuffle=True):
#         print('len images: ', image_list)
#         print('len mask_list: ', mask_list)
#         dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
#         # dataset = dataset.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
#         # dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=True)
        
#         # if shuffle:
#         #     dataset = dataset.shuffle(buffer_size=len(self.image_list))
#         # # Prefetch the dataset for improved performance
#         # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
#         return dataset

# train_images_paths = sorted(glob(os.path.join(config.DATA_DIR, "Images/*")))[:config.NUM_TRAIN_IMAGES]
# train_masks_paths = sorted(glob(os.path.join(config.DATA_DIR, "Category_ids/*")))[:config.NUM_TRAIN_IMAGES]

# val_images_paths = sorted(glob(os.path.join(config.DATA_DIR, "Images/*")))[
#     config.NUM_TRAIN_IMAGES : config.NUM_VAL_IMAGES + config.NUM_TRAIN_IMAGES]
# val_masks_paths = sorted(glob(os.path.join(config.DATA_DIR, "Category_ids/*")))[
#     config.NUM_TRAIN_IMAGES : config.NUM_VAL_IMAGES + config.NUM_TRAIN_IMAGES]

# train_data_loader = CIHPDataLoader(train_images_paths,train_masks_paths)
# train_dataset = train_data_loader.get_dataset()


# val_data_loader = CIHPDataLoader(val_images_paths,val_masks_paths)
# val_dataset = val_data_loader.get_dataset()
    
# print("Train Dataset:", train_dataset)
# print("Val Dataset:", val_dataset)

# for i, (x,y) in enumerate(train_dataset):
#     print(x.shape, y.shape)
#     if i ==5:
#         break

# for i, (x,y) in enumerate(val_dataset):
#     print(x.shape, y.shape)
#     if i ==5:
#         break