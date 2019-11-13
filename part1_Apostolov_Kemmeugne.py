#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:24:10 2019

@author: anthonykemmeugne
@author: AlexanderApostolov

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest
 
#Shuffles the train Data (to be used at the end of each epoch)
def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def softmax(z):
    expon = np.exp(z)
    A = expon/np.sum(expon, axis=1, keepdims=True)
    return A
    
#This function accepts 3 arguments: a weight, an input, and a bias matrix and
#returns the product between the weights and input, plus the biases.
def compute(X_prev, W, b):
    pre = np.matmul(X_prev,W)
    return np.add(pre,b)
    
def relu(x):
    return np.maximum(x, 0)

#This function accepts two arguments, the targets (e.g. labels - not onehot encoded!!!) and predic-
#tions. It returns a number, the AVERAGE cross entropy loss.
def averageCE(target, prediction):
    N = prediction.shape[0]
    #we don't need to sum the logs for incorrect predictions
    correct_logprobs = -np.log(prediction[range(N),target])
    loss = np.sum(correct_logprobs)/N
    return loss

#This function accepts two arguments, the targets (e.g. labels) and predictions.
#It returns the gradient of the average cross entropy loss with respect to the softmax of the
#predictions
def gradAverageCE(target, prediction):
    S = np.apply_along_axis(softmax, 1, prediction)
    S_reciprocal = np.reciprocal(S)
    N = target.shape[0]
    result = (-1.0/N)*np.multiply(target, S_reciprocal)
    result[np.isnan(result)]=0.0
    return result


    
def trainNN(h, epochs, takeOnlyTestStats=False, gamma=0.9):
    dimw = 784 #number of input nodes
    K = 10 # number of classes
    
    #initializing the weights and biases following the Xaiver initialization scheme (zero-mean Gaussians with variance 2(unitsin+unitsout )
    W = (math.sqrt(2/(dimw+h))) * np.random.randn(dimw,h)
    b = np.zeros((1,h))
    W2 =(math.sqrt(2/(h+K))) * np.random.randn(h,K)
    b2 = np.zeros((1,K))
    
    #initialize the datasets
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)
    num_examples = trainTarget.shape[0]
    num_examples_valid = validTarget.shape[0]
    num_examples_test = testTarget.shape[0]
    X = np.zeros((num_examples,dimw))
    Xvalid = np.zeros((num_examples_valid,dimw))
    Xtest = np.zeros((num_examples_test,dimw))
    
    #initialize the velocity to 1e-5    
    vnewW = np.zeros((dimw,h))+1e-5
    vnewW2 = np.zeros((h,K))+1e-5
    vnewb = np.zeros((1,h))+1e-5
    vnewb2 = np.zeros((1,K))+1e-5
    alpha = 1-gamma
    
    #prepareTables
    trainLoss = np.zeros(epochs)
    trainAccuracy = np.zeros(epochs)
    validationLoss = np.zeros(epochs)
    validationAccuracy = np.zeros(epochs)
    testLoss = np.zeros(epochs)
    testAccuracy = np.zeros(epochs)
    
    for i in range(0,num_examples):
        X[i]=trainData[i].flatten()
    for i in range(0,num_examples_valid):
        Xvalid[i]=validData[i].flatten()
    for i in range(0,num_examples_test):
        Xtest[i]=testData[i].flatten()
        
        
    for i in range(epochs):
        #shuffle data at each epoch
        X,trainTarget=shuffle(X,trainTarget)
        
        #forward propagation
        z = compute(X,W,b);
        X_layer1 = relu(z)
        out = compute(X_layer1,W2,b2)
        prediction=np.argmax(out, axis=1)
        Sk = softmax(out)
        loss = averageCE(trainTarget,Sk)
        
        if i % 10 == 0:
            print("iteration %d: loss %f" % (i, loss))
            
        #Calcultate delta outter    
        deltaO = Sk
        deltaO[range(num_examples),trainTarget] -= 1
        deltaO /= num_examples
        
        #Backpropagate to outter layer
        dWo = np.matmul(np.transpose(X_layer1), deltaO)
        dbo = np.sum(deltaO, axis=0, keepdims=True)
        
        #Calculate delta hidden
        deltah = np.matmul(deltaO, np.transpose(W2))
        # backprop the ReLU non-linearity, effect of multiplying by the derivative of ReLu
        deltah[X_layer1 <= 0] = 0
        
        #Backpropagate to hidden layer
        dWh = np.dot(X.T, deltah)
        dbh = np.sum(deltah, axis=0, keepdims=True)
        
        #Update the velocities, weights and biases
        vnewW = gamma*vnewW+alpha*dWh
        vnewb = gamma*vnewb+alpha*dbh
        vnewW2 = gamma*vnewW2+alpha*dWo
        vnewb2 = gamma*vnewb2+alpha*dbo
        W += -vnewW
        b += -vnewb
        W2 += -vnewW2
        b2 += -vnewb2
        
        
        
        
        trainLoss[i]=loss
        trainAccuracy[i]=np.mean(prediction==trainTarget)
        if(takeOnlyTestStats==False):
            z_valid = compute(Xvalid,W,b);
            hidden_layer_valid = relu(z_valid)
            scores_valid = compute(hidden_layer_valid,W2,b2)
            prediction_valid=np.argmax(scores_valid, axis=1)
            probs_valid = softmax(scores_valid)
            loss_valid = averageCE(validTarget,probs_valid)
            validationLoss[i]=loss_valid
            validationAccuracy[i]=np.mean(prediction_valid==validTarget)
            
        z_test = compute(Xtest,W,b);
        hidden_layer_test = relu(z_test)
        scores_test = compute(hidden_layer_test,W2,b2)
        prediction_test=np.argmax(scores_test, axis=1)
        probs_test = softmax(scores_test)
        loss_test = averageCE(testTarget,probs_test)
        testLoss[i]=loss_test
        testAccuracy[i]=np.mean(prediction_test==testTarget)
        
    return W, b, W2, b2, trainLoss, trainAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy   
    

def getDataExercise13():
    print("Getting data for exercise 1.3...")
    epochs = 200
    hiddenUnitSize = 1000
    W_hid, b_hid, W_out, b_out, trainLoss, trainAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy = trainNN(hiddenUnitSize, epochs)
    W_hid2, b_hid2, W_out2, b_out2, trainLoss2, trainAccuracy2, validationLoss2, validationAccuracy2, testLoss2, testAccuracy2 = trainNN(hiddenUnitSize, epochs, gamma=0.99)
    with open('exercise13.pkl', 'wb') as f: 
        pickle.dump([W_hid, b_hid, W_out, b_out, trainLoss, trainAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy, W_hid2, b_hid2, W_out2, b_out2, trainLoss2, trainAccuracy2, validationLoss2, validationAccuracy2, testLoss2, testAccuracy2], f)
    return

def plotExercise13():
    with open('exercise13.pkl', 'rb') as f:  
        _, _, _, _, trainLoss, trainAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy, _, _, _, _, trainLoss2, trainAccuracy2, validationLoss2, validationAccuracy2, testLoss2, testAccuracy2 = pickle.load(f)
        
    startIndex = 0
    endIndex = 200
    x = range(startIndex, endIndex)
    plt.title("Losses (gamma = 0.9)")
    plt.plot(x,trainLoss[startIndex:endIndex], '-b', label='TrainData')
    plt.plot(x,validationLoss[startIndex:endIndex], '-r', label='ValidationData')
    plt.plot(x,testLoss[startIndex:endIndex], '-g', label='TestData')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    
    plt.title("Accuracies (gamma = 0.9)")
    plt.plot(x,trainAccuracy[startIndex:endIndex], '-b', label='TrainData')
    plt.plot(x,validationAccuracy[startIndex:endIndex], '-r', label='ValidationData')
    plt.plot(x,testAccuracy[startIndex:endIndex], '-g', label='TestData')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    
    #GAMMA=0.99
    startIndex = 0
    endIndex = 200
    x = range(startIndex, endIndex)
    plt.title("Losses (gamma = 0.99)")
    plt.plot(x,trainLoss2[startIndex:endIndex], '-b', label='TrainData')
    plt.plot(x,validationLoss2[startIndex:endIndex], '-r', label='ValidationData')
    plt.plot(x,testLoss2[startIndex:endIndex], '-g', label='TestData')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    
    plt.title("Accuracies (gamma = 0.99)")
    plt.plot(x,trainAccuracy2[startIndex:endIndex], '-b', label='TrainData')
    plt.plot(x,validationAccuracy2[startIndex:endIndex], '-r', label='ValidationData')
    plt.plot(x,testAccuracy2[startIndex:endIndex], '-g', label='TestData')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    return

def getDataExercise14():
    print("Getting data for exercise 1.4...")
    epochs = 200

    hiddenUnitSize = 100
    _, _, _, _, _, _, _, _, testLoss_1, testAccuracy_1 = trainNN(hiddenUnitSize, epochs, takeOnlyTestStats=True)
    
    hiddenUnitSize = 500
    _, _, _, _, _, _, _, _, testLoss_2, testAccuracy_2 = trainNN(hiddenUnitSize, epochs, takeOnlyTestStats=True)
    
    hiddenUnitSize = 2000
    _, _, _, _, _, _, _, _, testLoss_3, testAccuracy_3 = trainNN(hiddenUnitSize, epochs, takeOnlyTestStats=True)
    with open('exercise14.pkl', 'wb') as f: 
        pickle.dump([testLoss_1, testAccuracy_1, testLoss_2, testAccuracy_2, testLoss_3, testAccuracy_3], f)
    return

def plotExercise14():
    with open('exercise14.pkl', 'rb') as f:  
        testLoss_1, testAccuracy_1, testLoss_2, testAccuracy_2, testLoss_3, testAccuracy_3 = pickle.load(f)
        
    startIndex = 0
    endIndex = 200
    x = range(startIndex, endIndex)
    plt.title("Losses with different amount of hidden units")
    plt.plot(x,testLoss_1[startIndex:endIndex], '-b', label='100 hidden units')
    plt.plot(x,testLoss_2[startIndex:endIndex], '-r', label='500 hidden units')
    plt.plot(x,testLoss_3[startIndex:endIndex], '-g', label='2000 hidden units')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    
    plt.title("Accuracies with different amount of hidden units")
    plt.plot(x,testAccuracy_1[startIndex:endIndex], '-b', label='100 hidden units')
    plt.plot(x,testAccuracy_2[startIndex:endIndex], '-r', label='500 hidden units')
    plt.plot(x,testAccuracy_3[startIndex:endIndex], '-g', label='2000 hidden units')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    return

def checkSomeValues(start, end):
    if(start<0 or end >10000):
        print("Incorrect bounds")
        return
    with open('exercise13.pkl','rb') as f:
           W_hid, b_hid, W_out, b_out, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = pickle.load(f)
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    X = np.zeros((10000,784))
    for i in range(0,10000):
            X[i]=trainData[i].flatten()
    for i in range(start,end):
        
        print("Image: ", i ,": \n")
        plt.imshow(trainData[i], cmap="gray")
        plt.show()
        z = compute(X[i],W_hid,b_hid);
        hidden_layer = relu(z)
        scores = compute(hidden_layer,W_out,b_out)
        probs = softmax(scores)
#        print("Prediction : ", probs)
        x=np.array(['A','B','C','D','E','F','G','H','I','J'])
        print("Predicted value :", x[np.argmax(probs)])
        print("Real value : ", x[trainTarget[i]])

#getDataExercise13()
#getDataExercise14()

#plotExercise13()
#plotExercise14()

#checkSomeValues(10,20)
