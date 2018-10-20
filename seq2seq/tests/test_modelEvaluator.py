#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:32:25 2018

@author: rgarzon
"""



import unittest
import sys

sys.path.append("..")

from keras_seq2seq_agent import BasicSeq2SeqAgent
#from keras_seq2seq_agent import BasicSeq2SeqAgent
from RandomAgent import RandomAgent
from modelEvaluator import modelEvaluator

class TestModelEvaluator(unittest.TestCase):

#    def setUp(self):
        
#    def test_runEpisodesForRandomAgent(self):
#        numEpisodes = 10
#        numBlocks = 5
#        randomAgent = RandomAgent(numBlocks)
#        me = modelEvaluator(randomAgent)
#        me.runEpisodesForAgent(numEpisodes,numBlocks)
            
    def test_runEpisodesForKerasBasicSeq2SeqAgent(self):
        numEpisodes = 10
        numBlocks = 5
        basicSeq2SeqAgent = BasicSeq2SeqAgent()
        me = modelEvaluator(basicSeq2SeqAgent)
        epLengths = me.runEpisodesForAgent(numEpisodes,numBlocks)
        self.assertEqual(len(epLengths),numEpisodes)
        
if __name__ == '__main__':
    unittest.main()
        
        