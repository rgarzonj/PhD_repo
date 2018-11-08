#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:33:33 2018

@author: rgarzon
"""

from keras.models import Model
from keras.layers import Input, LSTM
from PointerLSTM import PointerLSTM

import numpy as np
import random


MAX_NUM_BLOCKS = 10
#Consider the current and goal state (*2) and also the blank spaces between state representation (*2)
MAX_INPUT_SEQ_LENGTH = 4*MAX_NUM_BLOCKS-1 #-1 because the last block do not have a blank space after it



class PointerNetworkAgent:
    '''Implements an agent with a pointer network model trained on optimal plans
   '''
    
    def __init__(self,numBlocks,modelPath):
        """ 
        Args:

        """
        self.numBlocks = numBlocks
        hidden_size = 128
        self.seq_len = 20

        print("building model...")
        main_input = Input(shape=(self.seq_len, 1), name='main_input')
        
        encoder = LSTM(output_dim = hidden_size, return_sequences = True, name="encoder")(main_input)
        decoder = PointerLSTM(hidden_size, output_dim=hidden_size, name="decoder")(encoder)
        
        self.model = Model(input=main_input, output=decoder)
        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        self.model.load_weights(modelPath)
        print ("Pointer Network model loaded correctly")
    
    def _padInputWithCharacter (self,inputSeq,padChar):
        # Generate data method feeds an string as inputSeq
        # OpenAI Gym feeds an array as inputSeq
        # If we receive an array we move it to a string first
        returnString = True
        if (isinstance(inputSeq,np.ndarray)):
            returnString = False
            newSeq = str(inputSeq[0]) 
            for item in inputSeq[1:]:
                newSeq = newSeq + " " + str(item)
            inputSeq = newSeq
            #print (newSeq + 'EEEEE')
            #InputSeq should be a string
            #print (inputSeq)
            #print (type(inputSeq))
        ret = inputSeq
        #TODO Support more than 10 blocks
        if (len(inputSeq)<MAX_INPUT_SEQ_LENGTH):
            numBlocks = int((1+len(inputSeq))/4)      
            #We need to pad with unknowns
            ret_current = inputSeq[0:numBlocks*2-1]
            ret_goal = inputSeq[numBlocks*2:]
            #Multiply by 2 because need to consider the blank spaces
            for i in range(MAX_NUM_BLOCKS-numBlocks-1):
                ret_current = ret_current + ' ' + padChar
                ret_goal = ret_goal + ' ' + padChar
            ret = ret_current + ' ' + padChar + ' ' + ret_goal + ' ' + padChar
        #We must return a list of vocabulary items or a string
        if returnString==True:
            #When generating data
            return ret
        else:
            #When infering
            return ret.split(' ')
    
    
    def sampleAction(self,input_seq):        
        # Input seq is an int64 array when called from the openAI Gym

        #print ('Input sequence')
        #print (input_seq)
        #print (type(input_seq[0]))
        #print (type(input_seq))
        # Although the agent is trained on optimal plans
        # we may reach the situation where the model do not emit a good action for the current state
        # and this could take the episode to never complete because the agent is always suggesting the same
        # incorrect action. We add an exploration factor to unlock this case
        input_seq = self._padInputWithCharacter(input_seq,'0')
        r = random.random()
        if (r<0.1):
            return([random.randint(0,self.numBlocks),random.randint(0,self.numBlocks)])

        input_seq_formatted = np.reshape(input_seq,(-1,self.seq_len,1)) #1,20,1
        predictions = self.model.predict(input_seq_formatted) #(1,20,20)
        print ('predictions')
        #print (predictions)
        print (type(predictions))
        print (predictions.shape)
        #print(predictions)
        #print (predictions.shape)
        predictions_idx = np.argmax(predictions,axis=2) #1,20
        #print (predictions_idx.shape)
        print ('predictions_idx')
        print (predictions_idx)
        print (type(predictions_idx))
        print (predictions_idx.shape)
        block_to_move = predictions_idx[0][0]
        destination = predictions_idx[0][1]
        #print ('block_to_move')
        #print (block_to_move)
        #print ('destination')
        #print (destination)
        #TODO
        #Hack until we can improve the 20 classes output instead of 10
        if (block_to_move>self.numBlocks-1 or destination>self.numBlocks-1):
            return([random.randint(0,self.numBlocks),random.randint(0,self.numBlocks)])
        return ([block_to_move,destination])
    
    