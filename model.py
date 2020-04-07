import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Cropping2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
import csv
import os

from sklearn.model_selection import train_test_split

from random import shuffle

import numpy as np
import sklearn
import cv2


def generator(samples, batch_size=1):
    num_samples = len(samples)
    angle_correction = 0.2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/'+batch_sample[0].strip()
                center_image = cv2.imread(name)
                #center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
                center_angle = float(batch_sample[3].strip())
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)

                #left data
                name_left = './data/'+batch_sample[1].strip()
                left_image = cv2.imread(name_left)
                #left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2YUV)
                left_angle = float(batch_sample[3].strip())+angle_correction
                images.append(left_image)
                angles.append(left_angle)
                images.append(np.fliplr(left_image))
                angles.append(-left_angle)

                #right data
                name_right = './data/'+batch_sample[2].strip()
                right_image = cv2.imread(name_right)
                #right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2YUV)
                right_angle = float(batch_sample[3].strip())-angle_correction
                images.append(right_image)
                angles.append(right_angle)
                #augmentation

                images.append(np.fliplr(right_image))
                angles.append(-right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)


def cloning_model(ch, row, col, lr=0.001):

    shape = (160, 320, 3)
    print(shape)
    model = tf.keras.Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255.0 - 0.5))

    model.add(Conv2D(16,5,strides=(2, 2),activation="relu"))
    model.add(Dropout(0.3))
    model.add(Conv2D(32,5,strides=(2, 2),activation="relu"))
    model.add(Dropout(0.3))
    model.add(Conv2D(64,3,activation="relu"))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(Dense(16))
    model.add(Dense(1))

    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer="adam")
    return model
