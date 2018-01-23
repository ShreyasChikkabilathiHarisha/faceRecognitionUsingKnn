#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:16:44 2017

@author: risabh
"""

from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import os

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print("stage1")
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print(2)
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print(3)
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print(4)
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print(5)
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name = 'features'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

#if __name__ == "__main__":
    # Test pretrained model
model = VGG_16('/home/risabh/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
#model = Model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5")
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
im[:,:,0] -= 103.939
im[:,:,1] -= 116.779
im[:,:,2] -= 123.68
im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('features').output)
intermediate_output = intermediate_layer_model.predict(im)
filePath = './MsCelebV1-Faces-Aligned.Samples/'
folderDir = os.listdir(filePath)
for folder in folderDir:
    fileP = filePath+str(folder)+'/'
    fileDir = os.listdir(fileP)
    outFile = folder+'.csv'
    output_file = open(outFile,"wb")
    for f in fileDir:
        im = cv2.resize(cv2.imread(fileP+f), (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        intermediate_output = intermediate_layer_model.predict(im)
        output_file.write(str.encode(folder+','))
        output_file.write(str.encode(f+','))
        np.savetxt( output_file, intermediate_output, fmt='%0.5f', delimiter=',')
    print(outFile)
    output_file.close()

#out = model.predict(im)
