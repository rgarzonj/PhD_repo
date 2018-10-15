#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:46:10 2018

@author: rgarzon
"""
import unittest

from keras_seq2seq_agent import BasicSeq2SeqAgent

class TestBasicSeq2SeqAgent(unittest.TestCase):

    def setUp(self):
        modelPath = "tests/data/s2s.h5"
        modelConfig = {"latent_dim" : 256,
                       #"num_encoder_tokens" : 6,
                       #"num_decoder_tokens" : 8,
                       "input_characters": ['0', '1', '2', '3', '4', '5'],
                       "target_characters" : ['\t', '\n', '0', '1', '2', '3', '4', '5'],
                       "max_decoder_seq_length":4,
                       "max_encoder_seq_length":10}
         
        self.basicsS2SAgent = BasicSeq2SeqAgent(modelPath,modelConfig)


    def test_predict_sequence(self):
        inputSequence = ['2', '0', '0', '0', '1', '5', '4', '2', '1', '0']
        ret = self.basicsS2SAgent.predict_sequence(inputSequence)
        print (ret)
            
if __name__ == '__main__':
    unittest.main()
