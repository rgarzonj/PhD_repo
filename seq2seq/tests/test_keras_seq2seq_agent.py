#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:46:10 2018

@author: rgarzon
"""
import unittest
import sys

sys.path.append("..")

from keras_seq2seq_agent import BasicSeq2SeqAgent

class TestBasicSeq2SeqAgent(unittest.TestCase):

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
         
        self.basicsS2SAgent = BasicSeq2SeqAgent(numBlocks)


#    def test_sampleAction(self):
#        inputSequence = ['2', '0', '0', '0', '1', '5', '4', '2', '1', '0']
#        ret = self.basicsS2SAgent.sampleAction(inputSequence)
#        self.assertEqual(ret,'50\n')
#        print (ret)
    
    def test_sampleAction_with_Unknowns(self):
        inputSequence = np.array((0,1,0,0,0,4,5,0,3,1))
        #INCORRECT        inputSequence = '0 1 2 3 4 <UNK> <UNK> <UNK> <UNK> <UNK> 0 5 1 0 3 <UNK> <UNK> <UNK> <UNK> <UNK>'
#        inputSequence = ['0', '1', '0', '0', '0', '<UNK>', '<UNK>', '<UNK>', '<UNK>','<UNK>', '4', '5', '0', '3', '1','<UNK>', '<UNK>', '<UNK>', '<UNK>','<UNK>']
        ret = self.basicsS2SAgent.sampleAction(inputSequence)
        self.assertEqual(ret,'43\n')
        print (ret)


if __name__ == '__main__':
    unittest.main()
