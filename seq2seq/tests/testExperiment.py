#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:11:03 2018

@author: rgarzon
"""

import unittest
import sys
sys.path.append("..")

from experiment import experimentSet
from experiment import experiment
import numpy as np

class TestExperimentSet(unittest.TestCase):

#    def setUp(self):
        
#    def test_runEpisodesForRandomAgent(self):
#        numEpisodes = 10
#        numBlocks = 5
#        randomAgent = RandomAgent(numBlocks)
#        me = modelEvaluator(randomAgent)
#        me.runEpisodesForAgent(numEpisodes,numBlocks)
            
    def test_plotExperiment(self):
        numExperiments = 3
        maxEpisodeLengths = 1000
        numEpisodesPerTrial = 50
        eS = experimentSet()
        for i in range(numExperiments):
            exp = experiment('experiment ' +str(i))
            exp.addTrial(np.random.randint(maxEpisodeLengths, size=numEpisodesPerTrial))
            eS.addExperiment(exp)
        eS.plotExperimentSet()

if __name__ == '__main__':
    unittest.main()