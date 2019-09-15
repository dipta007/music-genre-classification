from subprocess import Popen, PIPE, STDOUT
import os
import eyed3
from config import rawDataPath
from PIL import Image
from config import pixelPerSecond, spectrogramsPath, slicesPath

desiredSize = 128
currentPath = os.path.dirname(os.path.realpath(__file__))
eyed3.log.setLevel("ERROR")


def isMono(filename):
    audiofile = eyed3.load(filename)
    if audiofile is None:
        return True
    return audiofile.info.mode == 'Mono'


def getGenre(filename):
    return filename.split('/')[-1][0: -10]


def createSlicesFromSpectrograms():
    for filename in os.listdir(spectrogramsPath):
        if filename.endswith(".png"):
            sliceSpectrogram(filename, desiredSize)


# TODO Improvement - Make sure we don't miss the end of the song
def sliceSpectrogram(filename, desiredSize):
    genre = filename.split("_")[0]  # Ex. Dubstep_19.png

    # Load the full spectrogram
    img = Image.open(spectrogramsPath + filename)

    # Compute approximate number of 128x128 samples
    width, height = img.size
    nbSamples = int(width / desiredSize)
    width - desiredSize

    # Create path if not existing
    slicePath = slicesPath + "{}/".format(genre);
    if not os.path.exists(os.path.dirname(slicePath)):
        os.makedirs(os.path.dirname(slicePath))

    # For each sample
    for i in range(nbSamples):
        print("Creating slice: ", (i + 1), "/", nbSamples, "for", filename)
        # Extract and save 128x128 sample
        startPixel = i * desiredSize
        imgTmp = img.crop((startPixel, 1, startPixel + desiredSize, desiredSize + 1))
        imgTmp.save(slicesPath + "{}/{}_{}.png".format(genre, filename[:-4], i))


def createSpectrogram(filename, newFilename):
    if isMono(rawDataPath + filename):
        command = "cp '{}' '/tmp/{}.mp3'".format(rawDataPath + filename, newFilename)
    else:
        command = "sox '{}' '/tmp/{}.mp3' remix 1,2".format(rawDataPath + filename, newFilename)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
    output, errors = p.communicate()
    if errors:
        print(errors)

    # Create spectrogram
    filename.replace(".mp3", "")
    command = "sox '/tmp/{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(newFilename, pixelPerSecond,
                                                                                       spectrogramsPath + newFilename)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
    output, errors = p.communicate()
    print("*********", errors, output)
    if errors:
        print(errors)

    # Remove tmp mono track
    os.remove("/tmp/{}.mp3".format(newFilename))


# Creates .png whole spectrograms from mp3 files
def createSpectrogramsFromAudio():
    genresID = dict()
    files = os.listdir(rawDataPath)
    files = [file for file in files if file.endswith(".mp3")]
    nbFiles = len(files)

    if not os.path.exists(os.path.dirname(spectrogramsPath)):
        os.makedirs(os.path.dirname(spectrogramsPath))

    for index, filename in enumerate(files):
        print(f"Creating spectrogram for file {index + 1}/{nbFiles}...")
        fileGenre = getGenre(rawDataPath + filename)
        genresID[fileGenre] = genresID[fileGenre] + 1 if fileGenre in genresID else 1
        fileID = genresID[fileGenre]
        newFilename = fileGenre + "_" + str(fileID)
        createSpectrogram(filename, newFilename)
