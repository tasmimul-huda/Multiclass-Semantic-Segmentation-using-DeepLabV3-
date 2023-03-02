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


def convolution_block(block_input, num_filters=256, kernel_size=3,dilation_rate=1,padding="same",use_bias=False, ):
    x = layers.Conv2D(num_filters,kernel_size=kernel_size,dilation_rate=dilation_rate,padding="same",use_bias=use_bias,kernel_initializer=keras.initializers.HeNormal(),)(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

class DeeplabV3Plus():
    def __init__(self,train_images_paths,train_masks_paths, val_images_paths,val_masks_paths, image_size, num_classes):
        self.image_size = image_size
        self.num_classes = num_classes
        self.train_images_paths = train_images_paths
        self.train_masks_paths = train_masks_paths
        self.val_images_paths = val_images_paths
        self.val_masks_paths = val_masks_paths
        
        self.model = None
        
        self.train_data_loader = CIHPDataLoader(train_images_paths,train_masks_paths)
        self.val_data_loader = CIHPDataLoader(val_images_paths,val_masks_paths)
        
    def build_model(self):
        model_input = keras.Input(shape=(self.image_size, self.image_size, 3))
        resnet50 = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=model_input)
        x = resnet50.get_layer("conv4_block6_2_relu").output
        x = DilatedSpatialPyramidPooling(x)
        
        input_a = layers.UpSampling2D(
            size=(self.image_size // 4 // x.shape[1], self.image_size // 4 // x.shape[2]),
            interpolation="bilinear",)(x)
        
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = convolution_block(x)
        x = convolution_block(x)
        x = layers.UpSampling2D(
            size=(self.image_size // x.shape[1], self.image_size // x.shape[2]),
            interpolation="bilinear",)(x)
        
        model_output = layers.Conv2D(self.num_classes, kernel_size=(1, 1), padding="same")(x)
        self.model = keras.Model(inputs=model_input, outputs=model_output)
        # return keras.Model(inputs=model_input, outputs=model_output)
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=["accuracy"],
                          )
    def train(self, epochs=40):
        train_dataset = self.train_data_loader.get_dataset()
        val_dataset = self.val_data_loader.get_dataset()
        history =  self.model.fit(train_dataset,
                                 epochs=epochs,
                                 validation_data=val_dataset,
                                 )
        return history     


# model = DeeplabV3Plus(config.train_images_paths,
#                       config.train_masks_paths, 
#                       config.val_images_paths,
#                       config.val_masks_paths, 
#                       config.IMAGE_SIZE, 
#                       config.NUM_CLASSES)

# # Build the model architecture
# model.build_model()
# # Compile the model
# model.compile_model(learning_rate=0.0001)

# history = model.train(epochs=2)

# plt.plot(history.history["loss"])
# plt.title("Training Loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.show()

# plt.plot(history.history["accuracy"])
# plt.title("Training Accuracy")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.show()

# plt.plot(history.history["val_loss"])
# plt.title("Validation Loss")
# plt.ylabel("val_loss")
# plt.xlabel("epoch")
# plt.show()

# plt.plot(history.history["val_accuracy"])
# plt.title("Validation Accuracy")
# plt.ylabel("val_accuracy")
# plt.xlabel("epoch")
# plt.show()
