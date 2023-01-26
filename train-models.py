import numpy as np

from PIL import Image
from io import BytesIO
from urllib import request

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.xception import Xception

import matplotlib.pyplot as plt
from IPython.display import display

img_size = 128

start_path = 'dices'
train_path = f"{start_path}/train/"
val_path = f"{start_path}/valid/"


def download_image(url):
    
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    return img


def preprocess(X, rescale=255):
    
    if rescale:
        X = X*1./rescale
        return X
    return X


def preprocess_xception(X):
    
    X /= 127.5
    X -= 1.
    
    return X

def get_model(img_size, learning_rate, droprate=0):
    
    inputs = keras.Input(shape=(img_size, img_size, 3))

    conv_1 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same')(inputs)
    conv_1_batched = keras.layers.BatchNormalization()(conv_1)
    conv_1_act = keras.layers.Activation(activation='relu') (conv_1_batched)
    drop_1 = keras.layers.Dropout(droprate)(conv_1_act)

    pooling_1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2))(drop_1)
   
    conv_2 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same')(pooling_1)
    conv_2_batched = keras.layers.BatchNormalization()(conv_2)
    conv_2_act = keras.layers.Activation(activation='relu') (conv_2_batched)
    drop_2 = keras.layers.Dropout(droprate)(conv_2_act)  
    
    pooling_2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2))(drop_2)

    conv_3 = keras.layers.Conv2D(filters=32, kernel_size=(1,1), padding='same')(pooling_2)
    conv_3_batched = keras.layers.BatchNormalization()(conv_3)
    conv_3_act = keras.layers.Activation(activation='relu') (conv_3_batched)
    
    drop_3 = keras.layers.Dropout(droprate)(conv_3_act)

    flatten = keras.layers.Flatten()(drop_3)

    inner_1 = keras.layers.Dense(72, activation='relu')(flatten)

    outputs = keras.layers.Dense(units=7, activation='softmax')(inner_1)

    model = keras.Model(inputs, outputs)   


    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    loss = keras.losses.CategoricalCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
#     display(model.summary())
    
    return model


def transfer_detection(img_size, inner_size, learning_rate, droprate=0.5):

    base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base_model.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))

    base = keras.Model(inputs=base_model.inputs, outputs=base_model.output)(inputs)

    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    drop = keras.layers.Dropout(droprate)(vectors)

    inner = keras.layers.Dense(inner_size, activation='relu')(drop)

    outputs = keras.layers.Dense(7, activation='softmax')(inner)
    
    model = keras.Model(inputs, outputs)
    
        
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    return model

###

best_lr = 0.0001
best_dr = 0.3
selected_n_epochs = 1

train_gen = ImageDataGenerator(preprocessing_function=preprocess, #rescale=1./255)
    shear_range=0.2,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_dataset = train_gen.flow_from_directory(directory=train_path,
                                              target_size=(img_size, img_size),
#                                               classes=['dicesback'],
                                              class_mode='categorical',
                                              shuffle=True,
                                              batch_size=64)

valid_gen = ImageDataGenerator(preprocessing_function=preprocess) #rescale=1./255)

valid_dataset = valid_gen.flow_from_directory(directory=val_path,
                                              target_size=(img_size, img_size),
#                                               classes=['dicesback'],
                                              class_mode='categorical',
                                              shuffle=True,
                                              batch_size=64)

classes = list(train_dataset.class_indices.keys())

detection_model = get_model(img_size, learning_rate=best_lr, droprate=best_dr)

checkpoint = keras.callbacks.ModelCheckpoint("trained-models/dice-detection-model-std-lanc-dr03-{val_accuracy:.3f}.h5",
                                            save_best_only=True, 
                                            monitor="val_accuracy",
                                            mode="max",
)

history_fin = detection_model.fit(train_dataset,
                                  epochs=selected_n_epochs,
                                  validation_data=valid_dataset,
                                 callbacks=[checkpoint])

###

best_lr = 0.0001
best_dr = 0.75
selected_n_epochs = 1

train_gen = ImageDataGenerator(preprocessing_function=preprocess_xception,
    shear_range=0.2,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_dataset = train_gen.flow_from_directory(directory=train_path,
                                              target_size=(img_size, img_size),
#                                               classes=['dicesback'],
                                              class_mode='categorical',
                                              shuffle=True,
                                              batch_size=64)

valid_gen = ImageDataGenerator(preprocessing_function=preprocess_xception)
valid_dataset = valid_gen.flow_from_directory(directory=val_path,
                                              target_size=(img_size, img_size),
#                                               classes=['dicesback'],
                                              class_mode='categorical',
                                              shuffle=True,
                                              batch_size=64)

classes = list(train_dataset.class_indices.keys())

classification_model = transfer_detection(img_size=128, inner_size=128, learning_rate=best_lr, droprate=best_dr)

checkpoint = keras.callbacks.ModelCheckpoint("trained-models/xception-classifier-prepr-lancoz-dr075-{val_accuracy:.3f}.h5",
                                            save_best_only=True, 
                                            monitor="val_accuracy",
                                            mode="max",
)

history_fin = classification_model.fit(train_dataset,
                                  epochs=selected_n_epochs,
                                  validation_data=valid_dataset,
                                 callbacks=[checkpoint])


