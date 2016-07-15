import numpy as np
import random
import theanets

class QValueNetwork:
    P_MIN = -1.2;
    P_MAX = 0.5;
    V_MIN = -0.07;
    V_MAX = 0.07;

    N_HIDDEN_1 = 5
    N_HIDDEN_2 = 5
    
    def __init__(self,num_actions,dim_states):
#        print 'Initializing Q network'
        #print dim_states
        #print num_actions
        self.INPUT_LAYER_SIZE = dim_states + num_actions
        self.num_states = dim_states
        self.num_actions = num_actions
        self.trainset = np.empty([0,dim_states+ num_actions])
        self.labels = np.empty([0,1])
        self.net = theanets.Regressor(layers=[self.INPUT_LAYER_SIZE,self.N_HIDDEN_1,1])
        
    def trainNetwork(self):
        #print 'Training network with the following train set'
        train = self.trainset,self.labels
        self.net.train(train, verbose='True',learning_rate=0.05)
        #print 'Reset trainset'    
        #print self.trainset
        #print self.labels
        
    def resetTrainSet(self):
        print 'Reset trainset'
        self.trainset = np.empty([1,self.num_states+ self.num_actions + 1])
        print self.trainSet
    def resetNetwork(self):
#      print 'Reset Q network'
        self.net = theanets.Regressor(layers=[self.INPUT_LAYER_SIZE,self.N_HIDDEN_1,self.N_HIDDEN_2,1])
        
    def getQValue (self,action, state):
        qVal = random.random()
        return qVal
    
    def computeQFunction (self,action, state):
#        print action
#        print state
        onerow = np.empty([0,4])
        onerow = np.vstack([onerow,np.array([state[0],state[1],state[2],action])])
        #print onerow
        q = self.net.predict(onerow)
        #print q
        return q
          
    def collectSample(self,observedUtility,action,state):
        #Add one sample to the train dataset
        onerow = np.empty([0,4])
        onerow = np.vstack([onerow,np.array([state[0],state[1],state[2],action])])
        #print onerow
        #self.trainSet = self.trainSet.append(onerow,ignore_index=True)
        self.trainset = np.vstack ([self.trainset,onerow])
        self.labels = np.vstack([self.labels,observedUtility])
        #print self.trainset
        #print self.labels
