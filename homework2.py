import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math

def question2(network):


    network.train(5)

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
        self.dims = [self.X.shape[1], 5, 1]
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

        #self.params['B1'] = np.zeros((1, self.dims[1]))
        #self.params['B2'] = np.zeros((1, self.dims[2]))
#
        #self.params['W1'] = np.random.normal(0, 0.01, self.dims[1] * self.dims[0])
        #self.params['W1'] = self.params['W1'].reshape(self.dims[0] , self.dims[1])
#
        #self.params['W2'] = np.random.normal(0, 0.01, self.dims[2] * self.dims[1])
        #self.params['W2'] = self.params['W2'].reshape(self.dims[1], self.dims[2])

        #FOR 2 HIDDEN LAYERED NEURAL NETWORK
        #self.params['W3'] = np.random.normal(0, 0.01, self.dims[3] * self.dims[2])
        #self.params['W3'] = self.params['W3'].reshape(self.dims[2], self.dims[3])
        #self.params['B3'] = np.zeros((1, self.dims[3]))
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

            print('Mini batch shape = ', mini_batch.shape)

            O, loss = self.forward2layer(mini_batch, self.mini_batchlabels)
            print('Output dim = ', O.shape)
            self.update2layer(self.mini_batchlabels)
            Ofull, lossfull = self.forward2layer(self.X, self.Y)
            self.loss.append(lossfull)
            self.mcetrain.append(self.predicttrain(Ofull))
            #self.loss.append(loss)
            #self.mcetrain.append(self.predicttrain(O))

            Valo, valloss = self.forward2layer(self.testims, self.testlbls)
            print('Valo dim =', Valo.shape)
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

            print('Mini batch shape = ', mini_batch.shape)

            O, loss = self.forward(mini_batch, self.mini_batchlabels)
            print('Output dim = ', O.shape)
            self.update(self.mini_batchlabels)
            Ofull , lossfull = self.forward(self.X, self.Y)
            self.loss.append(lossfull)
            self.mcetrain.append(self.predicttrain(Ofull))

            Valo, valloss = self.forward(self.testims,self.testlbls)
            print('Valo dim =', Valo.shape)
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


    #----------------------------------------------------------------------------------------------------------------------#


data = h5py.File('dataset/assign2_data1.h5','r')
network = tanhnet(data, 0.1, 0.5)
network.train2layer(250, 256)
