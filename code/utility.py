## utility functions

# import libraries
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

# sensor signal data inport
def importData(dataDir, dataFileName):
    
    # dataDir: data directory
    # dataFileName: file name of the sensor signal under the directory
    
    dataName = dataDir + dataFileName

    data = pd.read_csv(dataName, keep_default_na = False)
    dataValue = data['value'].to_numpy()
    dataTime = data['timestamp']
    dataLabel = data['label']
    
    # return dataValue: signal data
    # dataTime: time sequence of the data value
    # dataLabel: ground truth of the signal: 1: anomaly (fire), 0: normal (non-fire); (used in the method validation)
    return dataValue, dataTime, dataLabel

# Reshape np array to LSTM input shape
def LSTMInput(inputSize, X):
    # inputSize: dimensionality of the LSTM encoder input
    # X: sensor signal
    
    n = X.size
    LSTMX = np.zeros((n + 1 - inputSize, inputSize))
    for i in range(n + 1 - inputSize):
        LSTMX[i, :] = X[i : i + inputSize]
    
    # return LSTMX: reshaped sensor signal for encoder input
    return LSTMX

# fire score calculation (squared z-score)
def calFireScor(mean, std, error):
    # normal condition is modeled by normal dist

    fireScore = ((error - mean) / std) ** 2
    
    return fireScore

# reverse LSTM input reshape
def backTimeSeq(X, inputSize):
    # X: in shape (-1, 1, inputSize)
    # inputSize: dimensionality of the encoder input
    seqX = np.zeros((X.shape[0] + inputSize - 1,))
    seqX[0 : inputSize] = X[0, 0, :] # head
    
    seqX[seqX.size - inputSize  : seqX.size] = X[-1, 0, :] # tail
    
    for i in range(X.shape[0] - 1 - inputSize):
        for j in range(inputSize):
            seqX[i + inputSize] += X[i + j, 0, inputSize - 1 - j] / inputSize

    # return seqX: in shape (-1, 1, 1)
    return seqX

# fire alarm
def alarm(X, crit):
    # X: smoothed fire score sequency
    # crit: alarm threshold
    
    ind = 0
    n = X.size
    m = crit.size
    alarmTime = np.zeros(crit.shape)
    
    for i in range(m):
        ind = 0
        while True:
            if ind >= n:
                return np.zeros(crit.shape)
            
            if X[ind] > crit[i]:
                alarmTime[i] = ind
                break;
            
            ind += 1
            
    # return alarmTime: time for fire alarm
    return alarmTime

# Kalman filter steps        
def KFSignal(measurement, transition_covariance, observation_covariance, initial_state_mean, initial_state_covariance, filterTime):
    
    # kalman filter initialization
    kf = KalmanFilter(initial_state_mean = initial_state_mean, n_dim_obs=1, transition_covariance = transition_covariance, initial_state_covariance = initial_state_covariance, observation_covariance = observation_covariance)


    kf = kf.em(measurement, n_iter=3)
    
    # Kalman filtering
    for i in range(filterTime):
    
        (measurement, smoothed_state_covariances) = kf.smooth(measurement)

    (filteredSeq, smoothed_state_covariances) = kf.smooth(measurement)
    
    # return filteredSeq: smoothed fire score
    return filteredSeq

# fire score to fire probability
def score2Probability(score, threshold):
    # a sigmoid function like fire probability function
    scoreHat = score - threshold
    probability = np.exp(scoreHat) / (1 + np.exp(scoreHat))
    return probability


    
    
