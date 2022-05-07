# main file of ad-lstmvae

# import libraries
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

import utility as ut
import lstmvae as ad


import yaml


with open('./input.yaml') as f:
    
    hyperparameterList = yaml.safe_load(f)

# file directory
dataDir = hyperparameterList['dataDir']
outDir = hyperparameterList['outDir']

fileName = hyperparameterList['fileName'] # load data name (this is an example of anomaly (fire) happens periodically)


modelName = hyperparameterList['modelName'] # model file name for save/load
calibrationSize = hyperparameterList['calibrationSize'] # data size for model calibration

# hyperparameters


inputSize = hyperparameterList['inputSize'] # LSTM encoder input size
hiddenSize = hyperparameterList['hiddenSize'] # variational hidden layer dimensionality
numLayerEn = hyperparameterList['numLayerEn'] # deep LSTM encoder depth
numLayerDe = hyperparameterList['numLayerDe'] # deep LSTM decoder depth
predNorm = hyperparameterList['predNorm'] # normalization for reconstruction: suggested 0
isNorm = hyperparameterList['isNorm'] # input normalization: suggested 1 for LSTM encoder

epochs = hyperparameterList['epochs'] # training epoch number

predLatent = hyperparameterList['predLatent'] # reconstruction latency: suggested 0
threshold = hyperparameterList['threshold'] # anomaly score threshold: 90% 1.65^2; 95% 1.96^2; 99% 2.58^2  
isReload = hyperparameterList['isReload'] # relaod model?
isSave = hyperparameterList['isSave'] # save model?

dataFileName = fileName

# load data
    
dataValue, dataTime, dataLabel = ut.importData(dataDir, dataFileName)

# take samples for calibration
calibrationX = dataValue[0 : calibrationSize]

# model initialization
anomalyDetector = ad.ad_lstmvae(inputSize = inputSize, hiddenSize = hiddenSize, numLayerEn = numLayerEn, numLayerDe = numLayerDe, trainingEp = epochs, isVariational = 0, isNormal = isNorm, isKalman = 0, GaussianNorm = 0, calX = calibrationX, trainingRat = 0.8, lr = 1e-2, l2_reg = 0.0, predLatent = predLatent, predNorm = predNorm, isReload = isReload)
#sigma0: initialization for corruption optimization
#inputSize: encoder LSTM input size
#hiddenSize: valriational hidden layer dimension
#numLayerEn: depth of encoder
#numLayerDe: depth of decoder
#trainingEp: calibration epoch number
#isVariational: variation autoencoder: 0 is more robust to training (default); 1 is more robust to input noise
#isNormal: input normalization; suggested 1
#isKalman: using Kalman filter for fire score; suggested 1
#GaussianNorm: Gaussian or min-max normalization; suggested 0
#calX: calibration data
#trainingRat: split calibration to training and testing (only useful if isAdptive 1); suggested 0.8
#lr: learning rate, suggested 1e-4 to 1e-1 (need adjustment in deployment)
#l2_reg: l2 regularization for overfitting; suggest 0
#predLatent: prediction lantency; suggest 0
#predNorm: reconstruction normalization; suggested 0
#isReload: reload model


# model calibration or reload model

if isReload:
    anomalyDetector.loadModel(modelName)
else:
    anomalyDetector.mainCal()

# reconstruct data and anomaly score calculation
smoothedAnomalyScore = anomalyDetector.predict(dataValue)

# anomaly probability calculation
anomalyProbability = ut.score2Probability(smoothedAnomalyScore, threshold)

anomalyProbability[np.isnan(anomalyProbability)] = 0

# output probabilities
outFileName = outDir + 'LSTMVAE_' + dataFileName

probSerios = pd.Series(anomalyProbability, name = 'anomaly_probability')

dataValueSerios = pd.Series(dataValue, name = 'value')
frame = [dataTime, dataValueSerios, probSerios]

outSeries = pd.concat(frame, axis = 1)

outSeries.to_csv(outFileName, index = False)

# save model

if isSave:
    
    anomalyDetector.saveModel(outDir + modelName)









