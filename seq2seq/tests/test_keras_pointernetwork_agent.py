#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:46:10 2018

@author: rgarzon
"""
import unittest
import sys
import numpy as np

sys.path.append("..")

from keras_pointer_network_agent import PointerNetworkAgent

MODEL_PATH = '../model_weights_88.h5'

class TestPointerNetworkAgent(unittest.TestCase):

    def setUp(self):
        numBlocks = 5
#        modelPath = "tests/data/s2s.h5"
#        modelConfig = {"latent_dim" : 256,
#                       #"num_encoder_tokens" : 6,
#                       #"num_decoder_tokens" : 8,
#                       "input_characters": ['0', '1', '2', '3', '4', '5'],
#                       "target_characters" : ['\t', '\n', '0', '1', '2', '3', '4', '5'],
#                       "max_decoder_seq_length":4,
#                       "max_encoder_seq_length":10}
        
        self.PointerNetworkAgent = PointerNetworkAgent(numBlocks,MODEL_PATH)


#    def test_sampleAction(self):
#        inputSequence = ['2', '0', '0', '0', '1', '5', '4', '2', '1', '0']
#        ret = self.basicsS2SAgent.sampleAction(inputSequence)
#        self.assertEqual(ret,'50\n')
#        print (ret)
    
    
    def test_sampleAction(self):
#        inputSequence = np.array((0,4,2,5,1,5,0,1,3,0))
        inputSequence = np.array((0,0,1,3,2,0,0,4,5,2))
#        inputSequence = np.array((0,3,5,0,4,2,3,5,0,4))
 #        #INCORRECT        inputSequence = '0 1 2 3 4 <UNK> <UNK> <UNK> <UNK> <UNK> 0 5 1 0 3 <UNK> <UNK> <UNK> <UNK> <UNK>'
##        inputSequence = ['0', '1', '0', '0', '0', '<UNK>', '<UNK>', '<UNK>', '<UNK>','<UNK>', '4', '5', '0', '3', '1','<UNK>', '<UNK>', '<UNK>', '<UNK>','<UNK>']
        ret = self.PointerNetworkAgent.sampleAction(inputSequence)
#        self.assertEqual(ret,[3,0])
        self.assertEqual(ret,[4,5])
#        self.assertEqual(ret,[1,2])
        
        print (ret)


if __name__ == '__main__':
    unittest.main()
