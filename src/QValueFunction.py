import numpy as np
import random
#from util import loadContents

#from util import saveContents
class QValueFunction:
    P_MIN = -1.2;
    P_MAX = 0.5;
    V_MIN = -0.07;
    V_MAX = 0.07;
    def __init__(self,num_actions,dim_states):
        self.num_states = dim_states
        self.num_actions = num_actions
#        self.weights = -2500*np.random.rand(dim_states+num_actions)
        #self.weights = 1000*np.random.randn(dim_states+num_actions)
        self.weights = np.random.rand(dim_states+num_actions)
        self.inputDim = dim_states+num_actions

    def getQValue (self,action, state):
        qVal = random.random()
        return qVal
    
    def computeQFunction (self,action, state):
        f = np.zeros(self.inputDim)
        f[action+2] = 1
        f[0] = state[0]
        f[1] = state[1]
        #f = [(state[0]-self.P_MIN)/(self.P_MAX-self.P_MIN),(state[1]-self.V_MIN)/(self.V_MAX-self.V_MIN)]
        #print f
        q = np.dot(self.weights,f)
        return q
    
    def getWeights(self):
        print 'getWeights'
        print self.weights
        return self.weights
    
    def updateWeights(self,action_value, finalUtility,action,state,learning_rate):
#    def updateWeights(self,action_value, finalUtility,action,state,learning_rate,LAMBDA):
        f = np.zeros(self.inputDim)
        f[action+2] = 1
        f[0] = state[0]
        f[1] = state[1]
        #print f
        #print state
        #f = [(state[0]-self.P_MIN)/(self.P_MAX-self.P_MIN),(state[1]-self.V_MIN)/(self.V_MAX-self.V_MIN)]
        increment = learning_rate*(finalUtility-action_value)*f
#        increment = learning_rate*(finalUtility-action_value)*f
        self.weights = self.weights + increment 
        #print 'Updating weights'
        #print self.weights
        return self.weights
    
#     def saveWeights(self,fileName):
#         saveContents(self.weights,fileName)
# 
#     def loadWeights(self,fileName):
#         self.weights = loadContents(fileName)        
    #def updateWeights(self,weights):
    #    self.weights = weights
