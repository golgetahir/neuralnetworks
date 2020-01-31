import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2 as cv
import math

def TahirAhmet_Golge_21501627_hw2(question):
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
    ##Dimensions are given in initalization method of class
    data = h5py.File('assign2_data2.h5', 'r')
    print(data.keys())
    model = NLP(data, 0.15, 0.85)
    model.train(26, 200)
    model.graphmse()
    model.graphacc()
    model.find5sample()

    print('test', model.calculatetest())

def question2():
    ##Dimensions are given in initalization method of class
    data = h5py.File('assign2_data1.h5', 'r')
    network = tanhnet(data, 0.1)
    ## 1 hidden layer training without momentum
    network.train(250,256)
    ## 2 hidden layer training with momentum
    network2 = tanhnet(data, 0.1, 0.5)
    network2.train2layer(250, 256)


class tanhnet:
    def __init__(self,data,lr, momentum = None):

        ##INITIALIZATION OF DATASETS
        print(data.keys())
        self.trainlbls = np.array(data['trainlbls'])
        self.Y = self.trainlbls
        self.Y = self.Y.reshape(1900,1)
        self.trainims = data['trainims']
        self.trainims = np.array(self.trainims)
        self.X = self.trainims.reshape(self.trainims.shape[0], self.trainims.shape[1] * self.trainims.shape[2])
        self.X = self.X - np.mean(self.X)
        self.X = self.X / 255.0
        print(self.trainims.shape)
        self.testlbls = np.array(data['testlbls'])
        self.testims = np.array(data['testims'])
        self.testims = self.testims.reshape(self.testims.shape[0], self.testims.shape[1] * self.testims.shape[2] )
        self.testims = self.testims - np.mean(self.testims)
        self.testims = self.testims / 255.0
        self.testlbls = self.testlbls.reshape(len(self.testlbls),1)

        ##DIMENSIONS IS THE NEURON NUMBERS
        ##CAN CHANGE HIDDEN LAYER NEURON NUMBER WITH CHANGING DIMS[1]
        self.params = {}
        self.dims2layer = [self.X.shape[1], 200 , 100, 1]
        self.dims = [self.X.shape[1], 200, 1]
        self.gradients = {}
        self.loss = []
        self.holder = {}
        self.lr = lr
        self.momentum = momentum
        self.mcetrain = []
        self.mcetest = []
        self.valloss = []
        ##PARAMETER INITIALIZATION
        self.holder['dEdW3old'] = 0
        self.holder['dEdW2old'] = 0
        self.holder['dEdW1old'] = 0
        self.holder['dEdB3old'] = 0
        self.holder['dEdB2old'] = 0
        self.holder['dEdB1old'] = 0

    # ----------------------------------------------------------------------------------------------------------------------#
    def train2layer(self, epoch, batch_size):
        self.params['B1'] = np.zeros((1, self.dims2layer[1]))
        self.params['B2'] = np.zeros((1, self.dims2layer[2]))

        self.params['W1'] = np.random.normal(0, 0.01, self.dims2layer[1] * self.dims2layer[0])
        self.params['W1'] = self.params['W1'].reshape(self.dims2layer[0], self.dims2layer[1])
        #
        self.params['W2'] = np.random.normal(0, 0.01, self.dims2layer[2] * self.dims2layer[1])
        self.params['W2'] = self.params['W2'].reshape(self.dims2layer[1], self.dims2layer[2])

        # FOR 2 HIDDEN LAYERED NEURAL NETWORK
        self.params['W3'] = np.random.normal(0, 0.01, self.dims2layer[3] * self.dims2layer[2])
        self.params['W3'] = self.params['W3'].reshape(self.dims2layer[2], self.dims2layer[3])
        self.params['B3'] = np.zeros((1, self.dims2layer[3]))

        self.Vt3 = 0
        self.Vt2 = 0
        self.Vt1 = 0
        self.Vtb = 0
        self.Vtbo = 0
        self.Vtb3 = 0
        for i in range(epoch):
            indexes = np.random.sample(batch_size)
            indexes = np.round(indexes * 1899)
            indexes = indexes.astype(int)
            mini_batch = []
            self.mini_batchlabels = []
            for k in indexes:
                mini_batch.append(self.X[k])
                self.mini_batchlabels.append(self.Y[k])
            mini_batch = np.array(mini_batch)

            O, loss = self.forward2layer(mini_batch, self.mini_batchlabels)

            self.update2layer(self.mini_batchlabels)
            Ofull, lossfull = self.forward2layer(self.X, self.Y)
            self.loss.append(lossfull)
            self.mcetrain.append(self.predicttrain(Ofull))
            #self.loss.append(loss)
            #self.mcetrain.append(self.predicttrain(O))

            Valo, valloss = self.forward2layer(self.testims, self.testlbls)
            self.mcetest.append(self.predict(Valo))
            self.valloss.append(valloss)

            print('Epoch = ', i)
            print('MSE = ', self.loss[i])
            print('MSEtest = ', self.valloss[i])
            print('MCE(train) = ', self.mcetrain[i])
            print('MCE(test) = ', self.mcetest[i])
            print('---------------------------------------------------')
        self.graphmse()
    def train(self, epoch, batch_size):
        self.params['B1'] = np.zeros((1, self.dims[1]))
        self.params['B2'] = np.zeros((1, self.dims[2]))

        self.params['W1'] = np.random.normal(0, 0.01, self.dims[1] * self.dims[0])
        self.params['W1'] = self.params['W1'].reshape(self.dims[0], self.dims[1])
        #
        self.params['W2'] = np.random.normal(0, 0.01, self.dims[2] * self.dims[1])
        self.params['W2'] = self.params['W2'].reshape(self.dims[1], self.dims[2])


        for i in range(epoch):
            indexes = np.random.sample(batch_size)
            indexes = np.round(indexes * 1899)
            indexes = indexes.astype(int)
            mini_batch = []
            self.mini_batchlabels = []
            for k in indexes:
                mini_batch.append(self.X[k])
                self.mini_batchlabels.append(self.Y[k])
            mini_batch = np.array(mini_batch)

            O, loss = self.forward(mini_batch, self.mini_batchlabels)

            self.update(self.mini_batchlabels)
            Ofull , lossfull = self.forward(self.X, self.Y)
            self.loss.append(lossfull)
            self.mcetrain.append(self.predicttrain(Ofull))

            Valo, valloss = self.forward(self.testims,self.testlbls)

            self.mcetest.append(self.predict(Valo))
            self.valloss.append(valloss)


            print('Epoch = ' , i)
            print('MSE = ', self.loss[i])
            print('MSEtest = ', self.valloss[i])
            print('MCE(train) = ', self.mcetrain[i])
            print('MCE(test) = ', self.mcetest[i])
            print('---------------------------------------------------')
        self.graphmse()
    # ----------------------------------------------------------------------------------------------------------------------#
    def graphmse(self):
        plt.plot(self.loss, label='trainmse')
        plt.plot(self.valloss, label = 'testmse')
        plt.plot(self.mcetrain, label='mcetrain')
        plt.plot(self.mcetest, label= 'mcetest')
        plt.legend(['trainmse', 'testmse', 'mcetrain', 'mcetest'], loc='upper right')

        plt.savefig('graph-low-neuron')
        plt.show()

    def forward(self,X, Y):

        self.holder['X'] = X
        Z = X.dot(self.params['W1']) + self.params['B1']

        H = np.tanh(Z)
        self.holder['Z'], self.holder['H'] = Z, H

        V = H.dot(self.params['W2']) + self.params['B2']
        O = np.tanh(V)
        self.holder['V'], self.holder['O'] = V, O

        self.Yh = O
        loss = self.mse(O, Y)
#
        return O, loss

    def forward2layer(self,X, Y):
        self.holder['X'] = X

        Z = X.dot(self.params['W1']) + self.params['B1']
        H = np.tanh(Z)
        self.holder['Z'], self.holder['H'] = Z, H

        V = H.dot(self.params['W2']) + self.params['B2']
        O = np.tanh(V)
        self.holder['V'], self.holder['O'] = V, O

        V2 = O.dot(self.params['W3']) + self.params['B3']
        O2 = np.tanh(V2)
        self.holder['V2'], self.holder['O2'] = V2, O2

        self.Yh = O2
        loss = self.mse(O2, Y)

        return O2, loss
    #----------------------------------------------------------------------------------------------------------------------#
    def sigmoid(self,V):
        return 1 / (1 + np.exp(-V))

    def dSigmoid(self,Z):
        s = 1 / (1 + np.exp(-Z))
        dZ = s * (1 - s)
        return dZ
    #----------------------------------------------------------------------------------------------------------------------#
    def tanh(self,Z):
        s = ((np.exp(Z)-np.exp(-Z)) / (np.exp(-Z) + np.exp(Z)))
        return s

    def dtanh(self,Z):
        s = np.tanh(Z)
        dZ = 1 - np.square(s)
        return dZ
    #----------------------------------------------------------------------------------------------------------------------#
    def predicttrain(self,out):
        # out , loss = self.forward(X,Y)
        for i in range(len(out)):
            if out[i] > 0.5:
                out[i] = 1
            else:
                out[i] = 0
        count = 0
        for i in range(len(out)):
            if out[i] == self.Y[i]:
                count += 1

        return count / len(out)
    def predict(self, out):
        #out , loss = self.forward(X,Y)
        for i in range(len(out)):
            if out[i] > 0.5:
                out[i] = 1
            else:
                out[i] = 0
        count = 0
        conttest = 0
        for i in range(len(out)):
            if out[i] == self.testlbls[i]:
                count += 1

        return count/len(out)

    # ----------------------------------------------------------------------------------------------------------------------#
    def update(self, Y , momentum = None):
        dEdO = (self.Yh - Y)

        dOdV = self.dtanh(self.holder['V'])

        dVdW2 = self.holder['H']

        dVdH = self.params['W2']
        dHdZ = self.dtanh(self.holder['Z'])
        dZdW1 = self.holder['X']

        dError = np.multiply((self.Yh - self.mini_batchlabels), self.dtanh(self.holder['V']))
        dEdW2 =  dError.T.dot(dVdW2)
        dEdW2 =  (1 / len(Y)) * dEdW2

        #dEdW1 = dEdO * (dOdV)
        dEdW1 = dError.dot(dVdH.T)
        dEdW1 = dEdW1 * (dHdZ)
        dEdW1 =(1 / len(Y)) * dEdW1.T.dot(dZdW1)

        dEdB2 = dEdO.T.dot(dOdV)
        dEdB2 =(1 / len(Y)) * dEdB2

        dEdB1 = dEdO * (dOdV)
        dEdB1 = dEdB1.dot(dVdH.T)
        dEdB1 = dEdB1 * (dHdZ)
        dEdB1 = (1 / len(Y)) * np.sum(dEdB1, axis= 0, keepdims=True)
        #print('dedB1 shape = ', dEdB1.shape)

        self.params['W2'] = self.params['W2'] - (self.lr * dEdW2.T)
        self.params['W1'] = self.params['W1'] - (self.lr * dEdW1.T)

        self.params['B2'] = self.params['B2'] - (self.lr * dEdB2)
        #print('UPDATED B2 shape = ', self.params['B2'].shape)
        self.params['B1'] = self.params['B1'] - (self.lr * dEdB1)
        #print('UPDATED B1 shape = ', self.params['B1'].shape)
        return self.params

    def update2layer(self, Y):
        dEdO2 = (self.Yh - Y)

        dO2dV2 = self.dtanh(self.holder['V2'])
        dV2dW3 = self.holder['O']

        dV2dO = self.params['W3']
        dOdV = self.dtanh(self.holder['V'])
        dVdW2 = self.holder['H']

        dVdH = self.params['W2']
        dHdZ = self.dtanh(self.holder['Z'])
        dZdW1 = self.holder['X']

        dEdW3old = self.holder['dEdW3old']
        dError = np.multiply(dEdO2 , dO2dV2)
        dEdW3 = dError.T.dot(dV2dW3)
        dEdW3 = (1 / len(Y)) * dEdW3
        self.holder['dEdW3old'] = dEdW3

        dEdW2old = self.holder['dEdW2old']
        dEdW2 = dError.dot(dV2dO.T)
        dEdW2 = dEdW2 * (dOdV)
        dEdW2 = dEdW2.T.dot(dVdW2)
        dEdW2 =  (1 / len(Y)) * dEdW2
        self.holder['dEdW2old'] = dEdW2

        dEdW1old = self.holder['dEdW1old']

        #dEdW1 = dEdO * (dOdV)
        dEdW1 = dError.dot(dV2dO.T)
        dEdW1 = dEdW1 * dOdV
        dEdW1 = dEdW1.dot(dVdH.T)
        dEdW1 = dEdW1 * dHdZ
        dEdW1 = dEdW1.T.dot(dZdW1)
        dEdW1 =(1 / len(Y)) * dEdW1
        self.holder['dEdW1old'] = dEdW1

        dEdB3old = self.holder['dEdB3old']
        dEdB3 = (1 / len(Y)) * dError.T.dot(dO2dV2)
        self.holder['dEdB3old'] = dEdB3

        dEdB2old = self.holder['dEdB2old']
        dEdB2 = dError.dot(dV2dO.T)
        dEdB2 = dEdB2 * dOdV
        dEdB2 = np.sum(dEdB2, axis= 0, keepdims=True)
        dEdB2 =(1 / len(Y)) * dEdB2
        self.holder['dEdB2old'] = dEdB2

        dEdB1old = self.holder['dEdB1old']
        dEdB1 = dError.dot(dV2dO.T)
        dEdB1 = dEdB1 * dOdV
        dEdB1 = dEdB1.dot(dVdH.T)
        dEdB1 = dEdB1 * (dHdZ)
        dEdB1 = (1 / len(Y)) * np.sum(dEdB1, axis= 0, keepdims=True)
        self.holder['dEdB1old'] = dEdB1
        #print('dedB1 shape = ', dEdB1.shape)

        if self.momentum == None:
            self.params['W3'] = self.params['W3'] - (self.lr * dEdW3.T)
            self.params['W2'] = self.params['W2'] - (self.lr * dEdW2.T)
            self.params['W1'] = self.params['W1'] - (self.lr * dEdW1.T)

            self.params['B3'] = self.params['B3'] - (self.lr * dEdB3)
            self.params['B2'] = self.params['B2'] - (self.lr * dEdB2)
            self.params['B1'] = self.params['B1'] - (self.lr * dEdB1)
        else:
            self.Vt3 = self.momentum * self.Vt3 + self.lr * dEdW3
            self.Vt2 = self.momentum * self.Vt2 + self.lr * dEdW2
            self.Vt1 = self.momentum * self.Vt1 + self.lr * dEdW1
            self.Vtb3 = self.momentum * self.Vtb3 + self.lr * dEdB3
            self.Vtb = self.momentum * self.Vtb + self.lr * dEdB2
            self.Vtbo = self.momentum * self.Vtbo + self.lr * dEdB1
           #dEdW3 = self.lr * dEdW3 + self.momentum * dEdW3old
           #dEdW2 = self.lr * dEdW2 + self.momentum * dEdW2old
           #dEdW1 = self.lr * dEdW1 + self.momentum * dEdW1old
            self.params['W3'] = self.params['W3'] - self.Vt3.T
            self.params['W2'] = self.params['W2'] - self.Vt2.T
            self.params['W1'] = self.params['W1'] - self.Vt1.T

            self.params['B3'] = self.params['B3'] - self.Vtb3
            self.params['B2'] = self.params['B2'] - self.Vtb
            self.params['B1'] = self.params['B1'] - self.Vtbo
        return self.params
    #----------------------------------------------------------------------------------------------------------------------#

    def mse(self, O, Y):
        a = np.square(O-Y)
        return a.sum() / len(O)
    #----------------------------------------------------------------------------------------------------------------------#


class NLP:
    def __init__(self,data, lr, momentum):
        self.testx = np.array(data['testx'])
        self.testd = np.array(data['testd'])
        self.trainx = np.array(data['trainx'])
        self.traind = np.array(data['traind'])
        self.valx = np.array(data['valx'])
        self.vald = np.array(data['vald'])
        self.words = np.array(data['words'])

        ##DIMENSIONS IS THE NEURON NUMBERS
        ##CAN CHANGE HIDDEN LAYER NEURON NUMBER WITH CHANGING DIMS[1]
        self.params = {}
        self.dims = [250, 32, 5, 250]
        self.gradients = {}
        self.loss = []
        self.holder = {}
        self.lr = lr
        self.momentum = momentum
        self.mcetrain = []
        self.mcetest = []
        self.valloss = []

#----------------------------------------------------------------------------------------------------------------------#
    def encode(self, data):
        encoded = []
        for i in range(len(data)):
            vector1 = np.zeros((250, 1))
            vector2 = np.zeros((250, 1))
            vector3 = np.zeros((250, 1))
            if isinstance(data[i], np.int32):
                vector1[data[i] - 1] = 1
                encoded.append(vector1)
            else:
                for k in range(len(data[i])):
                    if k == 0:
                        vector1[data[i][0] - 1] = 1
                    elif k == 1:
                        vector2[data[i][1] - 1] = 1
                    elif k == 2:
                        vector3[data[i][2] - 1] = 1
                encoded.append([vector1, vector2, vector3])
        encoded = np.array(encoded)
        if isinstance(data[i], np.int32):
            encoded = encoded.reshape(len(data), 250)
            return encoded
        else:
            encoded = encoded.reshape(len(data),3, 250)
            return np.array(encoded)

# ----------------------------------------------------------------------------------------------------------------------#
    def train(self,epoch,batch_size):
        self.params['Bh'] = np.random.normal(0, 1, self.dims[2])
        self.params['Bh'] = self.params['Bh'].reshape(1, self.dims[2])
        self.params['Bo'] = np.random.normal(0, 1, self.dims[3])
        self.params['Bo'] = self.params['Bo'].reshape(1, self.dims[3])
        #self.params['Bh'] = np.zeros((1, self.dims[2]))
        #self.params['Bo'] = np.zeros((1, self.dims[3]))

        self.params['R'] = np.random.normal(0, 0.1, self.dims[1] * self.dims[0])
        self.params['R'] = self.params['R'].reshape(self.dims[0], self.dims[1])

        self.params['W'] = np.random.normal(0, 0.1, self.dims[2]* 3 * self.dims[1])
        self.params['W'] = self.params['W'].reshape(self.dims[1]*3, self.dims[2])
        #
        self.params['Wo'] = np.random.normal(0, 0.1, self.dims[2] * self.dims[3])
        self.params['Wo'] = self.params['Wo'].reshape(self.dims[2], self.dims[3])

        self.Vt3 = 0
        self.Vt2 = 0
        self.Vt1 = 0
        val = self.encode(self.valx)
        vald = self.encode(self.vald)
        for i in range(epoch):
            ## MINI BATCH CREATION
            indexes = np.random.sample(batch_size)
            indexes = np.round(indexes * (len(self.trainx)-1))
            indexes = indexes.astype(int)
            self.mini_batch = []
            self.mini_batchlabels = []
            for k in indexes:
                self.mini_batch.append(self.trainx[k])
                self.mini_batchlabels.append(self.traind[k])
            self.mini_batch = np.array(self.mini_batch)

            enctrain = self.encode(self.mini_batch)
            self.mini_batchlabels = np.array(self.mini_batchlabels)
            enclabel = self.encode(self.mini_batchlabels)
            ##MINI BATCHES READY

            O, loss = self.forward(enctrain,enclabel)
            self.loss.append(loss)
            self.update(enclabel)
            a = self.calculateacc(val[:2000],vald[:2000])
            self.valloss.append(a)

            print('Epoch = ', i)
            print('LOSS = ', loss)
            print('Vallacc = ', a)
            print('--------------------------------')

    def graphacc(self):
        plt.plot(self.valloss, label='valacc')
        plt.title('Validation Accuracy')
        plt.savefig('valacc')
        plt.show()
    def graphmse(self):
        plt.plot(self.loss, label='trainmse')
        plt.title('Cross Entropy Loss')
        plt.savefig('graph-low-neuron')
        plt.show()
    def forward(self, X, t):
        i = []
        for j in range(len(X)):
            a = []
            for k in range(3):
                a.append(X[j][k].dot(self.params['R']))
            a = np.array(a)

            a = a.reshape(1,self.dims[1]* 3)
            i.append(a)
        i = np.array(i)

        i = i.reshape(len(i), self.dims[1]*3)

        V = i.dot(self.params['W']) + self.params['Bh']

        h = self.sigmoid(V)

        Z = h.dot(self.params['Wo']) + self.params['Bo']

        O = self.softmax(Z)


        self.holder['X'] = X
        self.holder['i'] = i
        self.holder['V'] = V
        self.holder['h'] = h
        self.holder['Z'] = Z
        self.holder['O'] = O

        los = self.error(O, t)
        return O, los


    def sigmoid(self,V):
        return 1 / (1 + np.exp(-V))

    def dsigmoid(self,Z):
        s = 1 / (1 + np.exp(-Z))
        dZ = s * (1 - s)
        return dZ


    def update(self,t):
        ##DERIVATIVES
        ## t is the one hot encoded word should be appear
        dEdZ = (self.holder['O'] - t)
        dZdh = self.params['Wo']
        dhdV = self.dsigmoid(self.holder['V'])
        dVdi = self.params['W']
        ##DERIVATIVES WRT WEIGHTS
        didR = self.holder['X']
        dZdWo = self.holder['h']
        dVdW = self.holder['i']

        dVdi = dVdi.reshape(dVdi.shape[1], 3 , 32)
        dVdi = dVdi.sum(axis = 1, keepdims = False)
        didR = didR.sum(axis = 1, keepdims = False)

        ## dEdWo = dE/dZ * dZ/dWo
        ## dEdBo = dE/dZ * dZ/dBo
        dEdWo = dEdZ.T.dot(dZdWo)
        dEdBo = np.sum(dEdZ, axis= 0, keepdims=True)
        dEdWo = dEdWo * 1 / len(self.mini_batch)
        dEdBo = dEdBo * 1 / len(self.mini_batch)

        ## dEdW = dE/dZ * dZ/dh * dh/dV * dV/dW
        ## dEdW = dE/dZ * dZ/dh * dh/dV * dV/dBh
        dEdW = dEdZ.dot(dZdh.T)
        dEdW = dEdW * dhdV
        dEdBh = np.sum(dEdW, axis= 0, keepdims=True)
        dEdBh = dEdBh * 1 / len(self.mini_batch)
        dEdW = dEdW.T.dot(dVdW)
        dEdW = dEdW * 1 / len(self.mini_batch)

        ## dEdW = dE/dZ * dZ/dh * dh/dV * dV/di * di/dR
        dEdR = dEdZ.dot(self.params['Wo'].T)
        dEdR = dEdR * dhdV
        dEdR = dEdR.dot(dVdi)
        dEdR = dEdR.T.dot(didR)
        dEdR = 1/len(self.mini_batch) * dEdR

        self.Vt3 = self.momentum * self.Vt3 + self.lr * dEdWo
        self.Vt2 = self.momentum * self.Vt2 + self.lr * dEdW
        self.Vt1 = self.momentum * self.Vt1 + self.lr * dEdR

        self.params['Wo'] = self.params['Wo'] - self.Vt3.T
        self.params['W'] = self.params['W'] - self.Vt2.T

        self.params['R'] = self.params['R'] - self.Vt1.T

        self.params['Bo'] = self.params['Bo'] - self.lr * dEdBo
        self.params['Bh'] = self.params['Bh'] - self.lr * dEdBh

    def error(self, O, t):
        K = -1 * np.log(O)
        K = K * t
        return np.sum(K)/len(K)

    def softmax(self,x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def calculatetest(self):
        testx = self.encode(self.testx)
        testd = self.encode(self.testd)
        O, loss = self.forward(testx, testd)
        count = 0
        for i in range(len(testd)):
            maxindex = O[i].argmax()
            if self.vald[i] - 1 == maxindex:
                count += 1
        return count / len(testx)

    def find5sample(self):
        val = self.encode(self.valx[:5])
        vald = self.encode(self.vald[:5])
        O, loss = self.forward(val, vald)
        lists = []
        for i in range(len(O)):
            A = O[i].argsort()[-10:][::-1]
            lists.append(A)
        lists = np.array(lists)
        for k in range(len(lists)):
            for z in self.valx[k]:
                print(' For words = ', self.words[z-1])
            for j in lists[k]:
                print(self.words[j])
            print('------------------')
        return lists

    def calculateacc(self,val,vald):

        O,loss = self.forward(val,vald)
        count = 0
        for i in range(len(vald)):
            maxindex = O[i].argmax()
            if self.vald[i]-1 == maxindex:
                count += 1
        return count / len(val)

    def calculatetrainingacc(self):
        val = self.encode(self.trainx[:20000])
        vald = self.encode(self.traind[:20000])
        O, loss = self.forward(val, vald)
        count = 0
        for i in range(len(vald)):
            maxindex = O[i].argmax()

            if self.vald[i] - 1 == maxindex:
                count += 1
        return count / len(val)

question = sys.argv[1]
TahirAhmet_Golge_21501627_hw2(question)
