from tensorflow import keras
import os
from dataset_tools import getImageData
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def get_data():
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(genres)
    one_hot_encoded = keras.utils.to_categorical(integer_encoded)

    X = [[[] for _ in range(11)] for _ in range(1000)]
    Y = [[] for _ in range(1000)]
    for (gind, genre) in enumerate(genres):
        for f in os.listdir(f'./data/slices/{genre}'):
            ind = gind * 100 + int(f.split('_')[1]) - 1
            slice = int(f.split('_')[2].split('.')[0])
            if slice >= 11:
                continue
            X[ind][slice] = getImageData(f'./data/slices/{genre}/{f}', 128)

            genre_ind = genres.index(genre)
            Y[ind] = one_hot_encoded[genre_ind]
        print(f"{genre} completed")

    X = np.asarray(X)
    Y = np.asarray(Y)
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1, shuffle=True)
    return x_train, x_valid, y_train, y_valid


def get_model():
    model = keras.models.Sequential([
        keras.layers.Input(shape=(11, 128, 128, 1)),
        keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), data_format='channels_last',
                                recurrent_activation=keras.activations.hard_sigmoid,
                                activation=keras.activations.tanh, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2), padding='same', data_format='channels_last'),

        # keras.layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), data_format='channels_last',
        #                         recurrent_activation=keras.activations.hard_sigmoid,
        #                         activation=keras.activations.tanh, padding='same'),
        # keras.layers.BatchNormalization(),
        # keras.layers.MaxPooling2D((2, 2), padding='same', data_format='channels_last'),

        keras.layers.Flatten(),

        keras.layers.Dense(1024, activation=keras.activations.relu),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(10, activation=keras.activations.softmax),
    ])
    return model


def main():
    x_train, x_valid, y_train, y_valid = get_data()
    model = get_model()
    model.compile(optimizer=keras.optimizers.RMSprop(0.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    model.fit(x_train, y_train, epochs=100, shuffle=True, callbacks=[early_stopping], validation_split=0.1)
    model.evaluate(x_valid, y_valid)


if __name__ == '__main__':
    main()
