#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 13:46:34 2018

@author: rgarzon
"""

import matplotlib.pyplot as plt

class experimentSet ():
    
    def __init__(self):
        self.experiments = []    
        
    def addExperiment (self,experiment):
        self.experiments.append(experiment)

    def plotExperimentSet(self):
       for exp in self.experiments:
           #print (exp.getTrial())
           #print (len(exp.getTrial()))
           plt.plot(exp.getTrial(),label = exp.getName())
           plt.ylabel('Episode length')
           plt.xlabel ("Episode")
       legend = plt.legend(loc='upper right', shadow=False, fontsize='x-large')
       # Put a nicer background color on the legend.
       legend.get_frame().set_facecolor('C0')
       plt.show()
                    
class experiment ():

    def __init__(self,name):
        self.name = name
        
    def addTrial(self,trial):
        self.trial = trial

    def getTrial(self):
        return self.trial

    def getName(self):
        return self.name            