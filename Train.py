import os
import random
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import preprocessing

from keras.utils import to_categorical


# get the reference to the webcam
width  = 100
height = 100

##############

def load_images(base_path):
    images = []
    path = os.path.join(base_path, '*.jpg')
    for image_path in glob(path):
        image = preprocessing.image.load_img(image_path,
                                             target_size=(width, height))
        x = preprocessing.image.img_to_array(image)

        images.append(x)
    return images

###############

a = load_images('./data/a')
b = load_images('./data/b')
c = load_images('./data/c')
d = load_images('./data/d')
e = load_images('./data/e')
f = load_images('./data/f')

###################

# convert into numpy array
a = np.array(a)
b = np.array(b)
c = np.array(c)
d = np.array(d)
e = np.array(e)
f = np.array(f)


X = np.concatenate((a,b,c,d,e,f), axis=0)
print(X.shape)
##############

# normalization
X = X / 255.

###################


ya = [0 for item in enumerate(a)]
yb = [1 for item in enumerate(b)]
yc = [2 for item in enumerate(c)]
yd = [3 for item in enumerate(d)]
ye = [4 for item in enumerate(e)]
yf = [5 for item in enumerate(f)]

y = np.concatenate((ya,yb,yc,yd,ye,yf), axis=0)

y = to_categorical(y, num_classes=6)
print(y.shape)


#####################
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

# default parameters
conv_1 = 16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_drop = 0.2
dense_2_n = 512
dense_2_drop = 0.2
lr = 0.001

epochs = 500
batch_size = 32
color_channels = 3

print('aaa')

def build_model(conv_1_drop=conv_1_drop, conv_2_drop=conv_2_drop,
                dense_1_n=dense_1_n, dense_1_drop=dense_1_drop,
                dense_2_n=dense_2_n, dense_2_drop=dense_2_drop,
                lr=lr):
    model = Sequential()

    model.add(Convolution2D(conv_1, (5, 5),
                            input_shape=(width, height, color_channels),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_1_drop))

    model.add(Convolution2D(conv_2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_2_drop))
        
    model.add(Flatten())
        
    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_drop))

    model.add(Dense(dense_2_n, activation='relu'))
    model.add(Dropout(dense_2_drop))

    model.add(Dense(6, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])

    return model

#######################

import numpy as np

# model with base parameters
model = build_model()
print('a2')

model.summary()

print('a3')

#################

epochs = 50
##################

print('a4')

#model.fit(X, y, epochs=3)

model.fit(X, y, epochs=3)


model.summary()



print('a5')

model.save('HAR.h5')


print("Model_Ready")



