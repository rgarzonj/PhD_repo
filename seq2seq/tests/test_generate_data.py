#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:13:14 2018

@author: rgarzon
"""

import unittest
import sys

sys.path.append("..")

from generate_data import BW_data_generator

bwstates_path = '/Users/rgarzon/Documents/Projects/Ruben/phD/Repository/LSTMs/Blocksworld/GENERATOR/bwstates.1/bwstates'
bwopt_path = '/Users/rgarzon/Documents/Projects/Ruben/phD/Repository/LSTMs/Blocksworld/BWOPT/optimal/bwopt'    
numBlocks = 12

class TestGenerateData(unittest.TestCase):

    def setUp(self):
        self.bwgenerator = BW_data_generator(bwstates_path,bwopt_path,numBlocks)
    
    def test_generate_random_state(self):
        res = self.bwgenerator._generate_random_state()
        self.assertTrue(len(res)>0)
 
    def test_extract_state_from_bw_syntax(self):
        state_sample = " 4\n 2 4 0 3\n0"     
        res = self.bwgenerator._extract_state_from_bw_syntax(state_sample)
        self.assertEqual(res,"2 4 0 3")

    def test_get_optimal_plan(self):
        problem_setting = " 3\n 0 1 0\n 3\n 3 0 0\n0"
        res = self.bwgenerator._get_optimal_plan(problem_setting)
        #print ('Optimal Plan')
        #print (res)
        self.assertEqual(len(res),2)
    
    def test_generateVocabulary(self):
        ret = self.bwgenerator.generateVocabulary()
        self.assertTrue(ret)
        
    def test_generateInputAndOutputs(self):
        numSequences = 10
        inputs,outputs = self.bwgenerator.generateInputAndOutputs(numSequences)
        self.assertTrue(len(inputs)>0)
        self.assertTrue(len(outputs)>0)
        
    def test_addInputOutputToData(self):
        current_state = ['0' '1' '0' '0']
        goal_state = ['2', '3', '0', '1']
        nextAction = ['4' '3']
        inp,outp = self.bwgenerator._addInputOutputToDataset(current_state,goal_state,nextAction)
        self.assertTrue(len(inp)>0)
        self.assertTrue(len(outp)>0)
        
if __name__ == '__main__':
    unittest.main()