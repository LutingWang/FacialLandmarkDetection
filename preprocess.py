#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:00:46 2020

@author: lutingwang
"""

import random
import os
import math
import numpy as np
from PIL import Image, ImageStat, ImageEnhance
import visual


def _crop(image, rect):
    x_min, y_min, x_max, y_max = rect
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    side = max(x_center - x_min, y_center - y_min)
    side *= _crop.scale
    rect = (x_center - side, y_center - side, 
            x_center + side, y_center + side)
    image = image.crop(rect)
    return image, rect

def _resize(image, pts):
    pts = pts / image.size * _resize.target_size
    image = image.resize(_resize.target_size, Image.ANTIALIAS)
    return image, pts

def _fliplr(image, pts):
    if random.random() >= 0.5:
        pts[:, 0] = _fliplr.width - pts[:, 0]
        pts = pts[_fliplr.perm]
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image, pts

def _relight(image):
    r, g, b = ImageStat.Stat(image).mean
    brightness = math.sqrt(0.241 * r ** 2 + 0.691 * g ** 2 + 0.068 * b ** 2)
    image = ImageEnhance.Brightness(image).enhance(128 / brightness)
    return image

def preprocess(dataset, name):
    print(f"start processing {name}")
    image_dir = './WFLW/WFLW_images/'
    target_base = f'./dataset/{name}/'
    pts_set = []
    batch = 0
    for data in dataset:
        if not pts_set:
            print(f"batch {batch}")
            target_dir = target_base + f'batch_{batch}/'
            os.mkdir(target_dir)
        data = data.split(' ')
        pts = np.array(data[:196], dtype=np.float32).reshape((98, 2))
        rect = [int(x) for x in data[196:200]]
        image_path = data[-1][:-1]
        
        with Image.open(image_dir + image_path) as image:
            img, rect = _crop(image, rect)
        pts -= rect[:2]
        img, pts = _resize(img, pts)
        img, pts = _fliplr(img, pts)
        img = _relight(img)
        
        img.save(target_dir + f'{len(pts_set)}.jpg')
        pts_set.append(np.array(pts))
        if len(pts_set) == preprocess.batch_size:
            np.save(target_dir + 'pts.npy', pts_set)
            pts_set = []
            batch += 1
            

if __name__ == '__main__':
    annotation_dir = './WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/'
    train_file = 'list_98pt_rect_attr_train.txt'
    test_file = 'list_98pt_rect_attr_test.txt'
    
    _crop.scale = 1.25
    _resize.target_size = (128, 128)
    _fliplr.width = 128
    _fliplr.perm = np.load('fliplr_perm.npy')
    preprocess.batch_size = 50
    
    with open(annotation_dir + train_file, 'r') as dataset:
        preprocess(dataset, 'train')
    with open(annotation_dir + test_file, 'r') as dataset:
        preprocess(dataset, 'test')
