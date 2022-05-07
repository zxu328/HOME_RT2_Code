# class for LSTM

# suggested reference list:
# LSTM: Understanding LSTM Networks: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
# Auto-encoder: https://en.wikipedia.org/wiki/Autoencoder#:~:text=An%20autoencoder%20is%20a%20type,to%20ignore%20signal%20%E2%80%9Cnoise%E2%80%9D.
# Variational autoencoder: Doersch, C., 2016. Tutorial on variational autoencoders. arXiv preprint arXiv:1606.05908.
# Denoising autoencoder: Denoising autoencoders with Keras, TensorFlow, and Deep Learning: https://www.pyimagesearch.com/2020/02/24/denoising-autoencoders-with-keras-tensorflow-and-deep-learning/
# Kalman filter: Hargrave, P.J., 1989, February. A tutorial introduction to Kalman filtering. In IEE colloquium on Kalman filters: introduction, applications and future developments (pp. 1-1). IET.
# Kalman filter package: https://pykalman.github.io/
# ADAM optimizer: Gentle Introduction to the Adam Optimization Algorithm for Deep Learning; https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/#:~:text=Adam%20is%20a%20replacement%20optimization,sparse%20gradients%20on%20noisy%20problems.


# import pytorch, numpy, and Bayesian optimization libraries
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pykalman import KalmanFilter


import matplotlib.pyplot as plt

import utility as ut
import LSTMModel as LM



class ad_lstmvae:
    # model and vaviable initialization
    def __init__(self, inputSize, hiddenSize, numLayerEn, numLayerDe, trainingEp, isVariational, isNormal, isKalman, GaussianNorm, calX, trainingRat, lr, l2_reg, predLatent, predNorm, isReload):
        

        #inputSize: encoder LSTM input size
        #hiddenSize: valriational hidden layer dimension
        #numLayerEn: depth of encoder
        #numLayerDe: depth of decoder
        #trainingEp: calibration epoch number
        #isVariational: variation autoencoder: 0 is more robust to training; 1 is more robust to input noise
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
                
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayerEn = numLayerEn
        self.numLayerDe = numLayerDe
        self.trainingEp = trainingEp
        self.isVariational = isVariational
        self.isNorma = isNormal
        self.GaussianNorm = GaussianNorm
        self.calX = calX
        self.trainingRat = trainingRat
        if isVariational:
            self.model = LM.LSTMVAE(inputSize, hiddenSize, numLayerEn, numLayerDe)
            
        else:
            self.model = LM.LSTMAE(inputSize, hiddenSize, numLayerEn, numLayerDe)
            
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr, weight_decay = l2_reg)
        
        self.xTrain = None
        self.xTest = None
        self.meanError = None
        self.stdError = None
        self.isKalman = isKalman
        self.predLatent = predLatent
        self.predNorm = predNorm
        self.isReload = isReload
        
    # training step
    def trainingStep(self):
        
        # pytorch NN to training mode
        self.model.train()
        
        # convert training data to correct format and shape        
        xTrain = torch.from_numpy(self.xTrain).float()    
        xTrain = xTrain.view(-1, 1, self.inputSize)
        
        # add corruption
        
        xTrainUse = xTrain
            
        # define loss function    
        loss_fn1 = torch.nn.MSELoss() 
        
        # training iterations
        for t in range(self.trainingEp):
            # variational autoencoder training
            if self.isVariational:
                xPredict, mu, logVar = self.model(xTrainUse)
                loss2 = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
                loss1 = loss_fn1(xPredict, xTrain)
                loss = loss1 + loss2
            # autoencoder training
            else:
                xPredict = self.model(xTrainUse)
                loss = loss_fn1(xPredict, xTrain)
            
            if (t % 100 == 0):
                print('iterationK', t, 'loss:', loss.item())
                
            # backpropogation by optimizer (default: ADAM)
            self.model.zero_grad()
        
            loss.backward()
        
            self.optimizer.step()
            
        
            
    # testing step (testing to optimize corruption)
    def testingStep(self):
        # transfer testing data to correct format and shape
        corroptXTestOrg = torch.from_numpy(self.xTest).float()  
        corroptXTest = corroptXTest.view(-1, 1, self.inputSize)
        compareXTest = corroptXTestOrg.view(-1, 1, self.inputSize)    
        
        # pytorch NN to testing mode 
        self.model.eval()       
        if self.isVariational:
            testXPred, _, _ = self.model(corroptXTest)
        else:
            testXPred = self.model(corroptXTest)          
        N = self.xTest.size 
        
        # calculate testing RMSE for corruption optimization         
        testError = testXPred - compareXTest  
        errorNumpy = testError.detach().numpy()      
        testRMSE = np.sqrt(np.mean(errorNumpy ** 2))          
        return testRMSE
    

        
    # data reconstruction for fire detection
    def predict(self, xPre):
       
            
        # input data normalization
        if self.isNorma:
            
            # Gaussian normalization
            if self.GaussianNorm:
                meanX = np.mean(xPre)
                stdX = np.std(xPre)
                xPre = (xPre - meanX) / stdX
                
            # min-max optimization
            else:
                minX = np.min(xPre)
                maxX = np.max(xPre)
                if minX == maxX:
                    xPre = xPre - minX
                else:
                    xPre = (xPre - minX) / (maxX - minX)
                    
        # input data to compare to calculate fire score
        xPredCompare = xPre.copy()
        
        # transfer to LSTM input shape
        xPre = ut.LSTMInput(self.inputSize, xPre)
        
        # input data to pytorch tensor
        xPreUse = torch.from_numpy(xPre).float()
        xPreUse = xPreUse.view(-1, 1, self.inputSize)
        
        # pytorch NN to testing mode
        self.model.eval()
        
        # signal reconstruction
        if self.isVariational:
            yPred, _, _ = self.model(xPreUse)
        else:
            yPred = self.model(xPreUse)
        if self.isNorma:
            if self.GaussianNorm != 1:
                
                # output normalizatoin
                if self.predNorm:
                    yPredMax = torch.max(yPred)
                    yPreMin = torch.min(yPred)
                    yPred = (yPred - yPreMin) / (yPredMax - yPreMin)
        
        # back to numpy
        yPred = yPred.detach().numpy()
        yPred = ut.backTimeSeq(yPred, self.inputSize)
        
        # add lantency (optional)
        yPred = yPred[0 : yPred.size - self.predLatent]
        
        # comparing input and reconstruction for fire score
        xPredCompare = xPredCompare[self.predLatent : xPredCompare.size]   
        predictError = yPred - xPredCompare
        
        if self.isReload:
            self.stdError = np.std(predictError)
            self.meanError = np.mean(predictError)
    
        anomalyScore = ut.calFireScor(self.meanError, self.stdError, predictError)
        anomalyScoreSeq = anomalyScore
      
    
        # Kalman filter 
        
        transition_covariance = 1e8
        observation_covariance = 1e8
        initial_state_mean = 0
        initial_state_covariance = 1e8
        filterTime = 1

        if self.isKalman:
            smoothAnomalyScoreSeq = ut.KFSignal(anomalyScoreSeq, transition_covariance, observation_covariance, initial_state_mean, initial_state_covariance, filterTime)
        else:
            smoothAnomalyScoreSeq = anomalyScoreSeq
            
        
        return smoothAnomalyScoreSeq
    
        


    
    # data normalization
    def inputNorm(self):
        if self.GaussianNorm:
            meanX = np.mean(self.calX)
            stdX = np.std(self.calX)
            self.calX = (self.calX - meanX) / stdX
        else:
            minX = np.min(self.calX)
            maxX = np.max(self.calX)
            if minX == maxX:
                self.calX = self.calX - minX
            else:
                self.calX = (self.calX - minX) / (maxX - minX)
    
    # model calibration
    def mainCal(self):
        # calibration data reshape to correct shape
        self.calX = ut.LSTMInput(self.inputSize, self.calX)

        if self.isNorma:
            self.inputNorm()
            
        
        self.xTrain = self.calX.copy()
        self.trainingStep()
         
        xUse = self.xTrain.copy()
        xUse = torch.from_numpy(xUse).float()
        

            
        
    
        xUse = xUse.view(-1, 1, self.inputSize)
        xCompare = torch.from_numpy(self.xTrain).float()
        xCompare = xCompare.view(-1, 1, self.inputSize)
            
        if self.isVariational:
    
            xPredict, _, _ = self.model(xUse)
        
        else:
        
            xPredict = self.model(xUse)
    
        predictError = xPredict - xCompare
    
        errorNumpy = predictError.detach().numpy()
    
        # training error mean and standard deviation
        self.meanError = np.mean(errorNumpy)    
        self.stdError = np.std(errorNumpy)
            
    # save trained data    
    def saveModel(self, saveName):
        torch.save(self.model, saveName)
    
    # reload trained model    
    def loadModel(self, loadName):
        self.model = torch.load(loadName)
        self.model.eval()
         
            
            
            
        
        



