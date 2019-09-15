import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from dataset_tools import getDataset
from config import filesPerGenre, sliceSize, validationRatio, testRatio, nbEpoch

import warnings
warnings.filterwarnings('ignore')


def get_data():
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    train_X, train_y, validation_X, validation_y = getDataset(filesPerGenre, genres, sliceSize, validationRatio,                                                      testRatio, mode="train")
    test_X, test_y = getDataset(filesPerGenre, genres, sliceSize, validationRatio, testRatio, mode="test")
    return train_X, train_y, validation_X, validation_y, test_X, test_y


def get_model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(64, (2, 2), activation=keras.activations.elu, input_shape=(128, 128, 1)),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Conv2D(128, (2, 2), activation=keras.activations.elu),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Conv2D(256, (2, 2), activation=keras.activations.elu),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Conv2D(512, (2, 2), activation=keras.activations.elu),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation=keras.activations.elu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ])

    return model


def main():
    train_X, train_y, validation_X, validation_y, test_X, test_y = get_data()
    model = get_model()
    model.compile(optimizer=keras.optimizers.RMSprop(0.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    model.fit(train_X, train_y, epochs=nbEpoch, callbacks=[early_stopping], shuffle=True,
                validation_data=(validation_X, validation_y))
    model.evaluate(test_X, test_y)


if __name__ == '__main__':
    main()
