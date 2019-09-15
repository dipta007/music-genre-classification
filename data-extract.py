import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import csv
from PIL import Image
from spectogram import createSpectrogramsFromAudio, createSlicesFromSpectrograms

import warnings
warnings.filterwarnings('ignore')


def get_img_data():
    cmap = plt.get_cmap('inferno')

    plt.figure(figsize=(10, 10))
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for g in genres:
        pathlib.Path(f'./data/img_data/{g}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'./data/genres/{g}'):
            songname = f'./data/genres/{g}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=5)
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
            plt.axis('off')
            plt.savefig(f'./data/img_data/{g}/{filename[:-3].replace(".", "")}.png')
            plt.clf()

        print(f'{g} completed')


def get_features():
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open('./data/features.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for g in genres:
        for filename in os.listdir(f'./data/genres/{g}'):
            songname = f'./data/genres/{g}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            file = open('./data/features.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

        print(f'{g} completed')


def resize_img():
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for g in genres:
        pathlib.Path(f'./data/img_data_resized/{g}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'./data/img_data/{g}'):
            songname = f'./data/img_data/{g}/{filename}'
            im = Image.open(songname)
            im = im.resize((500, 500), Image.ANTIALIAS)  # best down-sizing filter
            im.save(f'./data/img_data_resized/{g}/{filename}')

        print(f'{g} completed')


def convert_to_mp3():
    print("Starting to convert to mp3")
    pathlib.Path(f'./data/raw').mkdir(parents=True, exist_ok=True)
    for f in os.listdir('./data/genres'):
        if not os.path.isfile(f):
            for mp3 in os.listdir(os.path.join('./data/genres', f)):
                os.system(f"cp ./data/genres/{f}/{mp3} ./data/raw")

    files = os.listdir('./data/raw/')
    os.chdir('./data/raw/')

    for (ind, file) in enumerate(files):
        if file.endswith('.au'):
            os.system("sox " + str(file) + " " + str(file[:-3]) + ".mp3")
        if ind % 10 == 0:
            print(f"Completed {ind//10}%")

    os.system("rm *.au")


def create_slices():
    print("Creating spectrograms...")
    createSpectrogramsFromAudio()
    print("Spectrograms created!")

    print("Creating slices...")
    createSlicesFromSpectrograms()
    print("Slices created!")


if __name__ == '__main__':
    # get_img_data()
    # get_features()
    # resize_img()
    # convert_to_mp3()
    create_slices()
