#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 08:36:44 2018

@author: rgarzon
"""
import random


class RandomAgent:
    '''Implements an agent with a seq2seq model trained on optimal plans
   '''
    
    def __init__(self,numBlocks):
            self.numBlocks = numBlocks
        
        
    def sampleAction(self,observation):
        nextAction = [random.randint(0,self.numBlocks),random.randint(0,self.numBlocks)]
        return nextAction
        