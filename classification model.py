# This assignment done by :
# Tala Hamdan
# Raghad Banat
# Sadeel ALhaliq
# Raghad Ramadan


import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import *
from tensorflow import keras
from keras import Sequential
import tensorflow as tf
from keras.applications import MobileNet




train_datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1.0/255.0,
    preprocessing_function=None,
    data_format=None,
    dtype=None,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory('data/',target_size=(256, 256),
                                                    batch_size=128,
                                                    class_mode='categorical',
                                                    interpolation="lanczos",
                                                    subset="training")
test_generator = train_datagen.flow_from_directory('data/',target_size=(256, 256),
                                                    batch_size=128,
                                                    class_mode='categorical',
                                                    interpolation="lanczos",
                                                    subset="validation")



pre_model =MobileNet (input_shape=(256,256, 3),
                   include_top=False,
                   weights='imagenet',
                   pooling='avg')
pre_model.trainable = False
inputs = pre_model.input
x = Dense(64, activation='relu')(pre_model.output)
x = Dense(64, activation='relu')(x)
outputs = Dense(4, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss = 'categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
my_callbacks  = [EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              mode='auto')]
history = model.fit(train_generator, validation_data=test_generator, epochs=48, callbacks=my_callbacks)
# Plotting Accuracy, val_accuracy, loss, val_loss
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()
for i, met in model(['accuracy', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['Train', 'Validation'])
plt.show()

# Predict Data Test
pred = model.predict(test_generator)
pred = np.argmax(pred, axis=1)
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
pred = [labels[k] for k in pred]

print('\033[01m              Classification_report \033[0m')

print('\033[01m              Results \033[0m')
# Results
results = model.evaluate(test_generator, verbose=0)
print("    Test Loss:\033[31m \033[01m {:.5f} \033[30m \033[0m".format(results[0]))
print("Test Accuracy:\033[32m \033[01m {:.2f}% \033[30m \033[0m".format(results[1] * 100))