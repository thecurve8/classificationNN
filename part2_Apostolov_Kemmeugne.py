# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:02:33 2019
@author: anthonykemmeugne
@author: AlexanderApostolov
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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

def cnn(learning_rate, epochs=50, batch_size=32, L2_loss=False, regularization=0.0, dropLayer=False, keep = 1.0, onlyFinal=False):
    tf.set_random_seed(421)
    #Xavier initializer
    initializer = tf.contrib.layers.xavier_initializer() 

    wc= tf.Variable(initializer([3,3,1,32]))
    bc =tf.Variable(initializer([32]))
    #zero-mean Gaussians initialization with variance 2/unitsin+unitsout
    sig1 = np.sqrt(2/((14*14*32)+784))
    sig2 = np.sqrt(2/(784+10)) 
    w1 = tf.Variable(tf.random_normal([14*14*32, 784], stddev = sig1), name='w1')
    b1 = tf.Variable(tf.random_normal([784] ,stddev = sig1), name = 'b1')
    
    w2 = tf.Variable(tf.random_normal([784, 10],stddev = sig2), name='w2')
    b2 = tf.Variable(tf.random_normal([10], stddev = sig2),  name = 'b2')
    
    
    x = tf.placeholder(tf.float32, shape=(None, 784), name='x')
    y = tf.placeholder(tf.float32, shape=(None, 10), name='y')
    reg = tf.placeholder(tf.float32, name='reg')
    
    #1. Input Layer
    input_layer = tf.reshape(x, shape=[-1, 28, 28, 1])
    
    #2. Convolutional layer
    conv = tf.nn.conv2d(input=input_layer, filter=wc, strides=[1,1,1,1], padding="SAME")
    conv = tf.nn.bias_add(conv, bc)
    
    #3. Relu activation
    relu1 = tf.nn.relu(conv)
    
    #4. Batch normalization layer 
    
    batch_mean, batch_var = tf.nn.moments(relu1,[0])
    normal = tf.nn.batch_normalization(relu1, batch_mean, batch_var, offset = None, scale = None, variance_epsilon = 1e-3)
    
    #5 A 2  2 max pooling layer
    
    maxpool = tf.nn.max_pool(normal, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    
    #6 Flatten layer 
    
    flat = tf.reshape(maxpool, [-1, 14*14*32])
    
    #7. Fully connected layer (with 784 output units, i.e. corresponding to each pixel)
    
    full = tf.add(tf.matmul(flat, w1), b1)
    
    #8.Dropout if needed + ReLU activation
    
    if(dropLayer):
        drop_layer = tf.nn.dropout(full, keep_prob=keep)
        relu2 = tf.nn.relu(drop_layer)
    else:
        relu2=tf.nn.relu(full)
        
    #9. Fully connected layer (with 10 output units, i.e. corresponding to each class)
    
    out = tf.add(tf.matmul(relu2, w2), b2)
    
    #10. Softmax output
    
    softmax_layer = tf.nn.softmax(out)
    
    #11. Cross Entropy loss
    
    if(L2_loss):
        loss_op = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=out, labels=y)) +
        reg*tf.nn.l2_loss(wc) +
        reg*tf.nn.l2_loss(w1) +
        reg*tf.nn.l2_loss(w2))
    else:
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    
    correct_pred = tf.equal(tf.argmax(softmax_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
     
    init = tf.global_variables_initializer()
    
    dimw = 784
    #initialize the datasets
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)
    num_examples = trainTarget.shape[0]
    num_examples_valid = validTarget.shape[0]
    num_examples_test = testTarget.shape[0]
    X = np.zeros((num_examples,dimw))
    Xvalid = np.zeros((num_examples_valid,dimw))
    Xtest = np.zeros((num_examples_test,dimw))
    for i in range(0,num_examples):
        X[i]=trainData[i].flatten()
    for i in range(0,num_examples_valid):
        Xvalid[i]=validData[i].flatten()
    for i in range(0,num_examples_test):
        Xtest[i]=testData[i].flatten()
    
    
    #prepareTables
    trainLoss = np.zeros(epochs)
    trainAccuracy = np.zeros(epochs)
    validationLoss = np.zeros(epochs)
    validationAccuracy = np.zeros(epochs)
    testLoss = np.zeros(epochs)
    testAccuracy = np.zeros(epochs)
    
    
    with tf.Session() as sess:
        sess.run(init)
        number_of_batches = num_examples//batch_size
    
        for step in range(0, epochs):
            #Shuffle after each epcoh
            flat_x_shuffled,trainingLabels_shuffled = shuffle(X, newtrain)
            
            for minibatch_index in range(0,number_of_batches):
                #select miniatch and run optimizer
                minibatch_x = flat_x_shuffled[minibatch_index*batch_size: (minibatch_index + 1)*batch_size, :]
                minibatch_y = trainingLabels_shuffled[minibatch_index*batch_size: (minibatch_index + 1)*batch_size, :]           
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y, reg: regularization})
            
            if(step==epochs-1 or not(onlyFinal)):
                lossTrain, accTrain = sess.run([loss_op, accuracy], feed_dict={x: flat_x_shuffled, y: trainingLabels_shuffled, reg: regularization})
                lossValid, accValid = sess.run([loss_op, accuracy], feed_dict={x: Xvalid, y: newvalid, reg: regularization})
                lossTest, accTest = sess.run([loss_op, accuracy], feed_dict={x: Xtest, y: newtest, reg: regularization})
                trainLoss[step]=lossTrain
                trainAccuracy[step]=accTrain
                validationLoss[step]=lossValid
                validationAccuracy[step]=accValid
                testLoss[step]=lossTest
                testAccuracy[step]=accTest
                print("Step " + str(step+1) + ", Train Loss= " + \
                              "{:.8f}".format(lossTrain) + ", Training Accuracy= " + \
                              "{:.8f}".format(accTrain))
            else:
                print("Step " + str(step+1) + " (no data, only final losses and accuracies will be saved)")
        print("Optimization Finished!")
    return trainLoss, trainAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy

def getDataExercise22():
    print("Getting data for exercise 2.2...")
    trainLoss, trainAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy = cnn(1e-4)
    with open('exercise22.pkl', 'wb') as f: 
        pickle.dump([trainLoss, trainAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy], f)
    return

def getDataExercise231():
    print("Getting data for exercise 2.3.1...")
    _, trainAccuracy1, _, validationAccuracy1, _, testAccuracy1 = cnn(1e-4, L2_loss=True, regularization=0.01, onlyFinal=True)
    _, trainAccuracy2, _, validationAccuracy2, _, testAccuracy2 = cnn(1e-4, L2_loss=True, regularization=0.1, onlyFinal=True)
    _, trainAccuracy3, _, validationAccuracy3, _, testAccuracy3 = cnn(1e-4, L2_loss=True, regularization=0.5, onlyFinal=True)
    with open('exercise231.pkl', 'wb') as f: 
        pickle.dump([trainAccuracy1, trainAccuracy2, trainAccuracy3, validationAccuracy1, validationAccuracy2, validationAccuracy3, testAccuracy1, testAccuracy2, testAccuracy3], f)
    return

def getDataExercise232():
    print("Getting data for exercise 2.3.2...")
    _, trainAccuracy1, _, validationAccuracy1, _, testAccuracy1 = cnn(1e-4, dropLayer=True, keep=0.9)
    _, trainAccuracy2, _, validationAccuracy2, _, testAccuracy2 = cnn(1e-4, dropLayer=True, keep=0.75)
    _, trainAccuracy3, _, validationAccuracy3, _, testAccuracy3 = cnn(1e-4, dropLayer=True, keep=0.5)
    with open('exercise232.pkl', 'wb') as f: 
        pickle.dump([trainAccuracy1, trainAccuracy2, trainAccuracy3, validationAccuracy1, validationAccuracy2, validationAccuracy3, testAccuracy1, testAccuracy2, testAccuracy3], f)
    return

def plotExercise22():
    with open('exercise22.pkl', 'rb') as f:  
        trainLoss, trainAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy = pickle.load(f)
        
    startIndex = 0
    endIndex = 50
    x = range(startIndex, endIndex)
    plt.title("Losses")
    plt.plot(x,trainLoss[startIndex:endIndex], '-b', label='TrainData')
    plt.plot(x,validationLoss[startIndex:endIndex], '-r', label='ValidationData')
    plt.plot(x,testLoss[startIndex:endIndex], '-g', label='TestData')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    
    plt.title("Accuracies")
    plt.plot(x,trainAccuracy[startIndex:endIndex], '-b', label='TrainData')
    plt.plot(x,validationAccuracy[startIndex:endIndex], '-r', label='ValidationData')
    plt.plot(x,testAccuracy[startIndex:endIndex], '-g', label='TestData')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    return

def printExercise231():
    ##getting all the accuracies
    with open('exercise231.pkl', 'rb') as f:  
        tr1, tr2, tr3, v1, v2, v3, te1, te2, te3 = pickle.load(f)
    print("data\treg=0.01\t\treg=0.1\t\t\treg=0.5")
    print("train\t"+str(tr1[-1])+"\t"+str(tr2[-1])+"\t"+str(tr3[-1]))
    print("valid\t"+str(v1[-1])+"\t"+str(v2[-1])+"\t"+str(v3[-1]))
    print("test\t"+str(te1[-1])+"\t"+str(te2[-1])+"\t"+str(te3[-1]))
    return

def plotExercise232():
    with open('exercise232.pkl', 'rb') as f:
        trainAccuracy1, trainAccuracy2, trainAccuracy3, validationAccuracy1, validationAccuracy2, validationAccuracy3, testAccuracy1, testAccuracy2, testAccuracy3 = pickle.load(f)
    startIndex = 0
    endIndex = 50
    x = range(startIndex, endIndex)
    
    plt.title("Accuracies with p=0.9")
    plt.plot(x,trainAccuracy1[startIndex:endIndex], '-b', label='TrainData')
    plt.plot(x,validationAccuracy1[startIndex:endIndex], '-r', label='ValidationData')
    plt.plot(x,testAccuracy1[startIndex:endIndex], '-g', label='TestData')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    
    plt.title("Accuracies with p=0.75")
    plt.plot(x,trainAccuracy2[startIndex:endIndex], '-b', label='TrainData')
    plt.plot(x,validationAccuracy2[startIndex:endIndex], '-r', label='ValidationData')
    plt.plot(x,testAccuracy2[startIndex:endIndex], '-g', label='TestData')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    
    plt.title("Accuracies with p=0.5")
    plt.plot(x,trainAccuracy3[startIndex:endIndex], '-b', label='TrainData')
    plt.plot(x,validationAccuracy3[startIndex:endIndex], '-r', label='ValidationData')
    plt.plot(x,testAccuracy3[startIndex:endIndex], '-g', label='TestData')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    return    

getDataExercise22()
getDataExercise231()
getDataExercise232()

#plotExercise22()
#printExercise231()
#plotExercise232()