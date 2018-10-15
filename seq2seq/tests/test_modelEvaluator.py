#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:32:25 2018

@author: rgarzon
"""

import unittest
import sys
import logging
import tensorflow as tf
sys.path.append("..")

from modelEvaluator import modelEvaluator

from seq2seq import seq2seq
from seq2seq import make_input_fn
from seq2seq import load_vocab

params = {
        'vocab_size': 9,
        'batch_size': 32,
        'input_max_length': 10,
        'output_max_length': 2,
        'embed_dim': 6,
        'num_units': 256
        }
model_dir = '../model/seq2seq'
tf.logging._logger.setLevel(logging.DEBUG)
vocab = vocab = load_vocab('../vocab')
print (vocab)

input_fn, feed_fn = make_input_fn(params['batch_size'],'../input','../output',vocab, params['input_max_length'],params['output_max_length'])

print (input_fn)

est = tf.estimator.Estimator(
                model_fn=seq2seq,
                model_dir=model_dir, params=params)

#ret = est.predict(input_fn)
ret = est.evaluate(input_fn=input_fn)
print ('Predicting')
print (ret)

#
#class TestModelEvaluator(unittest.TestCase):
#
#    def setUp(self):
#        self.params = {
#        'vocab_size': 9,
#        'batch_size': 32,
#        'input_max_length': 10,
#        'output_max_length': 2,
#        'embed_dim': 6,
#        'num_units': 256
#        }
#        model_dir = 'tests/model/seq2seq'
#        self.mE = modelEvaluator (seq2seq,model_dir,self.params)
#
#       
#    
##    def test_load_model(self):
##        params = {
##        'vocab_size': 9,
##        'batch_size': 32,
##        'input_max_length': 10,
##        'output_max_length': 2,
##        'embed_dim': 6,
##        'num_units': 256
##        }
##        model_dir = 'tests/model/seq2seq'
##        mE = modelEvaluator (self.seq2seq,model_dir,params)
##        self.assertTrue (True)
#        
#    def test_predict(self):               
##        input_sequence = "0 4 0 5 3 0 0 2 1 3"
#        tf.logging._logger.setLevel(logging.DEBUG)
#        vocab = vocab = load_vocab('../vocab')
#        print (vocab)
#        input_fn, feed_fn = make_input_fn(
#        self.params['batch_size'],
#        'input_eval',
#        'output_eval',
#        vocab, self.params['input_max_length'], self.params['output_max_length'])
#        print ('Input fn')
#        print (input_fn)
#        print (self.mE.estimator)        
#        ret = self.mE.predict(input_fn)
#        print (ret)
#        self.assertTrue(True)
#        
#if __name__ == '__main__':
#    unittest.main()