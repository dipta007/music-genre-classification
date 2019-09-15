import pandas as pd
import numpy as np
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


def get_data():
    data = pd.read_csv('./data/features.csv')
    data = data.drop(['filename'],axis=1)
    genre_list = data.iloc[:, -1]

    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)

    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

    return train_test_split(X, y, test_size=0.2, random_state=4)


def get_model():
    X_train, X_test, y_train, y_test = get_data()
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def main():
    X_train, X_test, y_train, y_test = get_data()
    x_val = X_train[:200]
    partial_x_train = X_train[200:]

    y_val = y_train[:200]
    partial_y_train = y_train[200:]

    model = get_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(partial_x_train,
              partial_y_train,
              epochs=30,
              batch_size=512,
              validation_data=(x_val, y_val))
    results = model.evaluate(X_test, y_test)

    print(results)


if __name__ == '__main__':
    main()
