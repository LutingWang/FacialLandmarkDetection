#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:03:07 2020

@author: lutingwang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model

import visual

BATCH_SIZE = 50
STEPS_PER_EPOCH = 7500 / BATCH_SIZE
VALIDATION_STEPS = 2500 / BATCH_SIZE


def pretrain(model: Model, name: str):
    """Use a pretrained model to extract features.
    
    @param model: pretrained model acting as extractors
    @param name: dataset name either "train" or "test"
    """
    print("predicting on " + name)
    base_path = f'./dataset/{name}/'
    for batch_path in os.listdir(base_path):
        batch_path = base_path + batch_path + '/'
        images = np.zeros((BATCH_SIZE, 128, 128, 3), dtype=np.uint8)
        for i in range(BATCH_SIZE):
            with Image.open(batch_path + f'{i}.jpg') as image:
                images[i] = np.array(image)
        result = model.predict_on_batch(images)
        np.save(batch_path + 'resnet50.npy', result)


def data_generator(base_path: str):
    """Data generator for keras fitter.
    
    @param base_path: path to dataset
    """
    while True:
        for batch_path in os.listdir(base_path):
            batch_path = base_path + batch_path + '/'
            pts = np.load(batch_path + 'pts.npy')\
                .reshape((BATCH_SIZE, 196))
            _input = np.load(batch_path + 'resnet50.npy')
            yield _input, pts


train_generator = data_generator('./dataset/train/')
test_generator = data_generator('./dataset/test/')

if __name__ == '__main__':
    print("Extracting features with ResNet50")
    base_model = ResNet50(include_top=False, input_shape=(128, 128, 3))
    output = base_model.layers[38].output
    model = Model(inputs=base_model.input, outputs=output)
    pretrain(model, 'train')
    pretrain(model, 'test')
    print()
        
    print("Building CNN")
    model = Sequential()
    model.add(Conv2D(256, (1, 1), input_shape=(32, 32, 256), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(512, (2, 2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(196))
    model.compile('adam', loss='mse', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    print()
        
    history = model.fit_generator(
        train_generator, 
        steps_per_epoch=STEPS_PER_EPOCH, 
        validation_data=test_generator,
        validation_steps=VALIDATION_STEPS,
        epochs=4)
    
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Visualize results
    visual.train_result(model)
    visual.test_result(model)
    