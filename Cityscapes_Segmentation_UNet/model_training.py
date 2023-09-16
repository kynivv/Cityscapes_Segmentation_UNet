import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import os

from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from glob import glob
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
import warnings

warnings.filterwarnings('ignore')


## Data Extraction
#with ZipFile('cityscapes.zip') as dataset:
#    dataset.extractall('data')


# Hyperparameters & Constants
BATCH_SIZE = 5
EPOCHS = 5
IMG_WIDTH = 256
IMG_HEIGHT = 96
IMG_SHAPE = (IMG_WIDTH, IMG_WIDTH, 3)
SPLIT = 0.20


# Data Preprocessing
data_path = 'data'

classes = os.listdir(f'{data_path}/train/')

X = []
Y = []

to_preprocess = {'train', 'val'}

for i, name in enumerate(classes):
    for dir in to_preprocess:
        images = glob(f'{data_path}/{dir}/{name}/*.png')
        for image in images:
            img = cv2.imread(image)
            if name == 'img':
                X.append(img)

            elif name == 'label':
                Y.append(img)

X = np.asarray(X)
Y = np.asarray(Y)

X = X.astype('float32')
Y = Y.astype('float32')


# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    shuffle= True,
                                                    random_state= 24,
                                                    test_size= SPLIT
                                                    )

print(X_train.shape,
      X_test.shape,
      Y_train.shape,
      Y_test.shape,
      )


# U-Net Model
model = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same',),
    layers.Dropout(0.1),

    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.2),

    layers.Conv2D(256, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding= 'same'),
    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Conv2DTranspose(64, (2, 2), strides= (2, 2), padding= 'same'),
    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.1),

    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Conv2D(3, (1, 1), activation='sigmoid')
])

model.compile(optimizer= 'adam',
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'],
              )


# Callbacks
checkpoint = ModelCheckpoint('output/model.h5',
                             monitor= 'val_loss',
                             verbose= 1,
                             save_best_only= True,
                             save_weights_only= False,
                             )


# Model Training
model.fit(X_train, Y_train,
          batch_size= BATCH_SIZE,
          epochs= EPOCHS,
          verbose= 1,
          validation_data= (X_test, Y_test),
          callbacks= checkpoint
          )
 
                

