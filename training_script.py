# Implementing a character classifier
import tensorflow as tf
import pandas as pd
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

h5_filename = sys.argv[1]

train_dir = '.'

train_datagen = ImageDataGenerator(#rescale=1./255,
    data_format='channels_first',
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
   train_dir,
   target_size=(28, 28),
   color_mode='grayscale',
   batch_size=20,
   shuffle=True,
   classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '(', ')', 'div'],
   class_mode='categorical',
   subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_dir, # Same directory as training data
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=20,
    shuffle=True,
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '(', ')', 'div'],
    class_mode='categorical',
    subset='validation')

# Model
import keras
keras.backend.set_image_data_format('channels_first')

### MODEL TRAINING BLOCK START
# Three steps to create a CNN
# 1. Convolution
# 2. Activation
# 3. Pooling
# Repeat Steps 1,2,3 for adding more hidden layers
nb_filters_1 = 64
nb_conv_init = 5
# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
model = Sequential()

model.add(Conv2D(nb_filters_1, (nb_conv_init, nb_conv_init), input_shape=(1, 28, 28)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters_1, (nb_conv_init, nb_conv_init)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(nb_filters_1,(nb_conv_init, nb_conv_init)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters_1, (nb_conv_init, nb_conv_init)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(16))

model.add(Activation('softmax'))

# Model
import keras
keras.backend.set_image_data_format('channels_first')

### MODEL TRAINING BLOCK START
# Three steps to create a CNN
# 1. Convolution
# 2. Activation
# 3. Pooling
# Repeat Steps 1,2,3 for adding more hidden layers
nb_filters_1 = 64
nb_conv_init = 5
# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
model = Sequential()

model.add(Conv2D(nb_filters_1, (nb_conv_init, nb_conv_init), input_shape=(1, 28, 28)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters_1, (nb_conv_init, nb_conv_init)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(nb_filters_1,(nb_conv_init, nb_conv_init)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters_1, (nb_conv_init, nb_conv_init)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(16))

model.add(Activation('softmax'))

from keras import optimizers
ada = keras.optimizers.Adadelta(learning_rate=1, rho=0.95)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.SGD(lr_schedule)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=1000,
                    validation_steps=1000,
                    epochs=25)

model.save(h5_filename)
