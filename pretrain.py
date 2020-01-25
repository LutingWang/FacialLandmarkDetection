# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:48:50 2020

@author: ThinkPad
"""

import os
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
from PIL import Image

batch_size = 50

def resnet50(model, name):
    print("predicting on " + name)
    base_path = f'./dataset/{name}/'
    for batch_path in os.listdir(base_path):
        print("batch " + batch_path)
        batch_path = base_path + batch_path + '/'
        images = np.zeros((batch_size, 128, 128, 3), dtype=np.uint8)
        for i in range(batch_size):
            with Image.open(batch_path + f'{i}.jpg') as image:
                images[i] = np.array(image)
        result = model.predict_on_batch(images)
        np.save(batch_path + 'resnet50.npy', result)

if __name__ == '__main__':
    base_model = ResNet50(include_top=False, input_shape=(128, 128, 3))
    output = base_model.layers[38].output
    model = Model(inputs=base_model.input, outputs=output)
    resnet50(model, 'train')
    resnet50(model, 'test')