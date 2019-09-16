#Define paths for files
spectrogramsPath = "data/spectrograms/"
slicesPath = "data/slices/"
datasetPath = "data/dataset/"
rawDataPath = "data/raw/"

#Spectrogram resolution
pixelPerSecond = 50

#Slice parameters
sliceSize = 128

#Dataset parameters
filesPerGenre = 100
validationRatio = 0.1
testRatio = 0.1

#Model parameters
batchSize = 1024
learningRate = 0.001
nbEpoch = 20