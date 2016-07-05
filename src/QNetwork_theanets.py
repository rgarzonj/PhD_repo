import numpy as np
import random
import pandas as pd
import theanets
#from util import loadContents

#from util import saveContents
class QValueNetwork:
    P_MIN = -1.2;
    P_MAX = 0.5;
    V_MIN = -0.07;
    V_MAX = 0.07;
    INPUT_LAYER_SIZE = 5
    N_HIDDEN_1 = 5
    N_HIDDEN_2 = 5
    
    def __init__(self,num_actions,dim_states):
        print 'Initializing Q network'

        self.num_states = dim_states
        self.num_actions = num_actions
        #self.weights = -2500*np.random.rand(dim_states+num_actions)
        #self.weights = 1000*np.random.randn(dim_states+num_actions)
        self.weights = np.random.randn(dim_states+ dim_states + dim_states + num_actions)
        self.trainSet = pd.DataFrame(columns=['Pos','Vel','A0','A1','A2','Q'])
        self.net = theanets.Regressor(layers=[self.INPUT_LAYER_SIZE,self.N_HIDDEN_1,1])
        
    def trainNetwork(self):
        print 'Training network'
        trainData = self.trainSet.drop('Q',axis=1)
        labels = self.trainSet['Q'].values.astype('float64')
        train = trainData.values,labels[:, None]
        self.net.train(train, verbose='True',learning_rate=0.05)
        print self.trainSet
        #print 'Reset trainset'
        #self.trainSet = pd.DataFrame(columns=['Pos','Vel','A0','A1','A2','Q'])       
        #print self.trainSet
        
    def resetTrainSet(self):
        print 'Reset trainset'
        self.trainSet = pd.DataFrame(columns=['Pos','Vel','A0','A1','A2','Q'])       
        #print self.trainSet
    def resetNetwork(self):
        print 'Reset Q network'
        self.net = theanets.Regressor(layers=[self.INPUT_LAYER_SIZE,self.N_HIDDEN_1,self.N_HIDDEN_2,1])
        
    def getQValue (self,action, state):
        qVal = random.random()
        return qVal
    
    def computeQFunction (self,action, state):
#        print action
#        print state
        #f = state
        onerow = pd.DataFrame(columns=['Pos','Vel','A0','A1','A2'])
#        a = np.zeros(self.num_actions)
#        a[action] = 1
        #f = np.concatenate((f,a),axis=0)
        f = {'Pos':state[0],'Vel':state[1],'A0':int(action==0),'A1': int(action==1),'A2' : int(action==2)}
 #       print f
        onerow = onerow.append(f,ignore_index=True)
        q = self.net.predict(onerow)
        #print q
        return q
          
    def collectSample(self,observedUtility,action,state):
        #Add one sample to the train dataset
        #a = np.zeros(self.num_actions)
        #a[action] = 1
        onerow = {'Pos':state[0],'Vel':state[1],'A0':int(action==0),'A1': int(action==1),'A2' : int(action==2), 'Q':observedUtility}
        #print onerow
        self.trainSet = self.trainSet.append(onerow,ignore_index=True)
        #print self.trainSet
        

    
#     def saveWeights(self,fileName):
#         saveContents(self.weights,fileName)
# 
#     def loadWeights(self,fileName):
#         self.weights = loadContents(fileName)        
    #def updateWeights(self,weights):
    #    self.weights = weights
