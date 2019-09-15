import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')


def get_data():
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    X = []
    Y = []

    for g in genres:
        for filename in os.listdir(f'./data/img_data_resized/{g}'):
            x = np.asarray(Image.open(f'./data/img_data_resized/{g}/{filename}'))
            X.append(x)
            Y.append(g)
        print(f'{g} completed')

    X = np.asarray(X)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    Y = onehot_encoded
    print(X.shape)
    print(Y.shape)

    return train_test_split(X, Y, test_size=0.1, random_state=4)


def get_model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu, input_shape=(500, 500, 4)),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Conv2D(128, (3, 3), activation=keras.activations.relu),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ])

    return model


def main():
    X_train, X_test, y_train, y_test = get_data()
    model = get_model()
    model.compile(optimizer=keras.optimizers.RMSprop(0.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    model.fit(X_train, y_train, epochs=20, callbacks=[early_stopping])
    model.evaluate(X_test, y_test)


if __name__ == '__main__':
    main()
