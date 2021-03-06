#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:11:31 2020

@author: lutingwang
"""
import numpy as np
from PIL import Image, ImageDraw
import functools


def _preview(image: Image, 
             pts: '98-by-2 matrix', 
             r=1, 
             color=(255, 0, 0)):
    """Draw landmark points on image."""
    draw = ImageDraw.Draw(image)
    for x, y in pts:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def _result(name: str, model):
    """Visualize model output on dataset specified by name."""
    path = f'./dataset/{name}/batch_0/'
    _input = np.load(path + 'resnet50.npy')
    pts = model.predict(_input)
    for i in range(50):
        with Image.open(path + f'{i}.jpg') as image:
            _preview(image, pts[i].reshape((98, 2)))
            image.save(f'./visualization/{name}/{i}.jpg')


train_result = functools.partial(_result, "train")
test_result = functools.partial(_result, "test")

if __name__ == '__main__':
    pass