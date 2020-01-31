
import numpy as np
import sys
import os
import cv2
import matplotlib.pyplot as plt
import math
import seaborn as sns


def TahirAhmet_Golge_21501627_hw1(question):
    if question == '1':
        ##question 1 code goes here
        return None
    elif question == '2':
        question2()
        ##question 2 code goes here
    elif question == '3':
        print(question)
        ##question 3 code goes here
        question3()

def question3():
    ##Loading the data
    trainimages = np.load('train_images.npy')
    trainlabels = np.load('train_labels.npy')
    testimages = np.load('test_images.npy')
    testlabels = np.load('test_labels.npy')
    print(type(max(trainlabels)))

    ##Selecting a random image from each category to display
    indexes = []
    for label in range(1, max(trainlabels)[0] + 1):
        choose = np.where(trainlabels == label)
        choose = choose[0]
        i = np.random.choice(choose)
        indexes.append(i)

    ##Preparing the data for training, normalizing
    testimagesflat = testimages.T
    testimagesflat = testimagesflat.reshape(testimagesflat.shape[0],
                                              testimagesflat.shape[2] * testimagesflat.shape[1])
    trainimagesflat = trainimages.T
    trainimagesflat = trainimagesflat.reshape(trainimagesflat.shape[0] , trainimagesflat.shape[2]* trainimagesflat.shape[1])
    trainimagesflat = trainimagesflat / 255.0

    testimagesflat = testimagesflat / 255.0
    displayimgs(indexes, trainimages)
    correlations(indexes, trainimages)

    ##One-hot encoding the labels for mse
    encodedlabels = encode(trainlabels)
    encvallabels = encode(testlabels)
    lrs = [0.8, 0.1, 0.01]

#-----------------------------------------------------------------------------------------------------------------#
    ##IF YOU WANT TO TRAIN THE ALGORITHM PLEASE UNCOMMENT THESE CODES
    ##TRAINING FOR 3 DIFFERENT LEARNING RATES 0.1- 0.01- 0.001
    #accuracies = []
#
    #for lr in lrs:
    #    msehist = []
    #    valmsehist = []
#
    #    ##784 weights 1 for each 26 neuron 784x26 matrix
    #    weights = np.random.normal(0, 0.01, 784 * 26)
    #    weights = weights.reshape(784, 26)
#
    #    ##A bias term for each neuron
    #    bias = np.random.normal(0, 0.01, 26)
    #    bias = bias.reshape(1, 26)
#
    #    ##Train for 10000 epochs
    #    output, loss = forward(weights, bias, trainimagesflat, encodedlabels)
    #    for i in range(10000):
    #        index = np.random.randint(low = 0, high=len(trainlabels))
    #        weights, bias = update(lr,output,encodedlabels,trainimagesflat,weights,bias,index)
    #        output, loss = forward(weights,bias,trainimagesflat,encodedlabels)
    #        msehist.append(loss)
    #        valoutput, valloss = forward(weights,bias,testimagesflat,encvallabels)
    #        valmsehist.append(valloss)
#
    #        print('Epoch = ', i)
    #        print('Training loss = ' , loss)
    #        print('Validation loss = ', valloss)
    #        print('----------------------------------')
#
    #    ##Saving and plotting the results
    #    np.save('weights_for_n' + str(lr),weights)
    #    np.save('bias_for_n' + str(lr), bias)
    #    np.save('loss_n' + str(lr), msehist)
    #    np.save('valloss_n' + str(lr), valmsehist)
    #    plotmse(msehist,valmsehist, lr)
#
    #    ##Calculating the accuracy through decoding the highest probability as label like given labels
    #    count = 0
    #    for i in range(len(testlabels)):
    #        maxindex = valoutput[i].argmax()
    #        if testlabels[i]-1 == maxindex:
    #            count += 1
    #    accuracies.append(count/len(testlabels))
    #    print('ACCURACY = ', count/len(testlabels))
    #    print('---------------------------------------------------------')
    #for i in range(3):
    #    print('Accuracy for lr ' + str(lrs[i]) + ' = ', accuracies[i])
#-------------------------------------------------------------------------------------------------------------#

    ##THESE PART IS FOR PRETRAINED WEIGHTS
    ##COMMENT THESE LINES IF YOU WILL TRAIN THE ALGORITHM
    for lr in lrs:
        count = 0
        weights = np.load('weights_for_n' + str(lr) +'.npy')
        bias = np.load('bias_for_n'+ str(lr) +'.npy')
        valmsehist = np.load('valloss_n'+ str(lr) +'.npy')
        msehist = np.load('loss_n' + str(lr) +'.npy')
        valoutput, valloss = forward(weights, bias, testimagesflat, encvallabels)
        plotmse(msehist, valmsehist, lr)
        displayweights(weights)
        for i in range(len(testlabels)):
            maxindex = valoutput[i].argmax()
            if testlabels[i]-1 == maxindex:
                count += 1
        print('ACCURACY for ' + str(lr) + ' = ', count/len(testlabels))
    mselow = np.load('loss_n0.01.npy')
    mse = np.load('loss_n0.1.npy')
    msehigh = np.load('loss_n0.8.npy')
    singlegraph(mselow,mse,msehigh)
    print('IF YOU WANT TO TRAIN THE ALGORITHM, PLEASE GO TO SOURCE CODE AND UNCOMMENT TRAINING PART')
#----------------------------------------------------------------------------------------------------------------------#
    return trainimages
#----------------------------------------------------------------------------------------------------------------------#
def singlegraph(mselow,mse,msehigh):
    plt.plot(mse, label='lr')
    plt.plot(mselow, label='lr_low')
    plt.plot(msehigh, label ='lr_high')
    plt.legend(['lr', 'lr_low','lr_high' ], loc='upper right')
    plt.title('losses for different lrs ')
    plt.savefig('singleplot.png')
    plt.show()
    return plt

def plotmse(mse,valmse, lr):
    plt.plot(mse,label = 'training_loss')
    plt.plot(valmse,label = 'validation_loss')
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    plt.title('losses for lr = ' + str(lr))
    plt.savefig('plotmsefor' + str(lr) + '.png')
    plt.show()
    return plt
#----------------------------------------------------------------------------------------------------------------------#

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y
#----------------------------------------------------------------------------------------------------------------------#

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r
#----------------------------------------------------------------------------------------------------------------------#
##Finding correlations between 2 images and showing in a matrix format
def correlations(indexes, images):
    count = 0
    cormat = []
    for i in (indexes):
        tmp = []
        img = images.T[i].T
        for j in indexes:
            count += 1
            img2 = images.T[j].T
            cor = corr2(img,img2)
            tmp.append(cor)
        cormat.append(tmp)
    cormat = np.array(cormat)
    #print(cormat)
    labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    ax = sns.heatmap(cormat, xticklabels=labels,yticklabels=labels)
    ax.set_ylim(26.0, 0)
    ax.set_xlim(26.0, 0)
    plt.savefig('correlations')
    plt.show()
    return cormat
#----------------------------------------------------------------------------------------------------------------------#
def displayweights(weights):
    fig = plt.figure(figsize=(8, 8))
    plt.title('represantation of weights as images')
    weights = weights.reshape(28,28,26)
    for i in (range(26)):
        fig.add_subplot(13, 2, i+1)
        plt.imshow(weights.T[i].T, cmap='Greys_r')
        #plt.savefig('images_each_weight')

    plt.show()
##WAITS A KEY TO PASS THE IMAGES
def displayimgs(indexes, images):
    fig = plt.figure(figsize=(8, 8))
    count = 0
    #cor = np.corrcoef(images.T[indexes[1]].T, images.T[indexes[6]].T)
    #print(cor)
    for i in (indexes):
        count +=1
        img = images.T[i].T
        fig.add_subplot(13, 2, count)
        plt.imshow(img, cmap='Greys_r')
        #plt.savefig('images_each_class')
    plt.show()

#----------------------------------------------------------------------------------------------------------------------#
##One hot encoding
def encode(labels):
    enclabels = np.zeros(len(labels)*26)
    enclabels = enclabels.reshape(len(labels),26)
    for i in range(len(labels)):
        enclabels[i][labels[i]-1] = 1
    return enclabels
#----------------------------------------------------------------------------------------------------------------------#

def forward(weights, bias, input, enclabels):
    V = input.dot(weights) + bias
    outputmatrix = sigmoid(V)
    encodedlabels = enclabels
    loss = mse(outputmatrix,encodedlabels)
    return outputmatrix, loss
#----------------------------------------------------------------------------------------------------------------------#

def dSigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = s * (1 - s)
    return dZ
#----------------------------------------------------------------------------------------------------------------------#

def update(lr, output, enclabels, train,oldweights, bias, index):
    upd = (enclabels[index] - output[index]) * dSigmoid(output[index])
    t = train[index].reshape(784,1)
    upd = upd.reshape(26,1)
    upd = (t.dot(upd.T))
    upd = lr*upd
    upd = oldweights + upd
    bias = bias + lr * ( (enclabels[index] - output[index]))

    return upd, bias
#----------------------------------------------------------------------------------------------------------------------#

def mse(yh,y):
    a = np.square(yh-y)
    return (a.sum() / len(y))
#----------------------------------------------------------------------------------------------------------------------#

def sigmoid(V):
    return 1 / (1 + np.exp(-V))
#----------------------------------------------------------------------------------------------------------------------#

def question2():
    inputs = []
    noise = np.random.normal(0,0.20,400*4)
    noise = noise.reshape(400,4)
    desiredo = np.array([[0],
                         [0],
                         [0],
                         [1],
                         [1],
                         [1],
                         [1],
                         [0],
                         [0],
                         [0],
                         [0],
                         [1],
                         [0],
                         [0],
                         [0],
                         [1]])
    ##Generating input samples
    for j in range(25):
        for i in range(16):
            x = np.binary_repr(i, width=4)
            inputs.append(np.array([int(x[0]), int(x[1]), int(x[2]), int(x[3])]))
    inputs = np.array(inputs)
    print('INPUTS(just 16 showed): ', inputs[0:16])
    inputs = inputs + noise
    print('NOISY INPUTS(just 16 showed): ', inputs[0:16])
    ##Weights that has been found
    weights = np.array([[-1, -1, 0.7, 0.7],
                  [-2, 2.5, -1, -1],
                  [0.7, -2, 0.2, 0.2],
                  [0.3, 0.3, 0.3, 0.3],
                  ])

    output = []
    doutputlist = []
    ##Finding the value which will go in to the activation function ( v = x.w )
    for i in range(25):
        output.append((inputs[16*(i):16*(i+1)]).dot(weights.T))
        doutputlist.append(desiredo)
    doutputlist= np.array(doutputlist)
    doutputlist = doutputlist.reshape(400, 1)
    print(doutputlist.shape)
    output = np.array(output)
    output = output.reshape(400,4)

    ##Activation function(step)
    for i in range(400):
        for j in range(4):
            if output[i][j] - 1 < 0:
                output[i][j] = 0
            else:
                output[i][j] = 1
    o = []

    ##Output neuron implemantation( or gate )
    for i in range(400):
        if output[i][0] == 1 or output[i][1]== 1 or output[i][2]== 1 or output[i][3] == 1:
            o.append(1)
        else:
            o.append(0)
    o = np.array(o)
    o = o.reshape(400,1)

    print('-------------------------------')
    print('DESIRED OUTPUTS: ', desiredo)
    print('PREDICTED OUTPUTS(just 16 showed): ', o[0:16])
    accuracy = np.sum(o == doutputlist)
    print('PERCENTAGE CORRECT = ' , accuracy/400)


question = sys.argv[1]
#question = '3'
TahirAhmet_Golge_21501627_hw1(question)

