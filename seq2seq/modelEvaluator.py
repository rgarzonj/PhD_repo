#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:30:06 2018

@author: rgarzon
"""
# Used to evaluate how an agent performs in the blocksworld environment
# Basically it is used to run episodes of the blocksworld problem and reports
# a list of the lengths (number of steps) of all the episodes.
# If the agent is performing well it should complete the problem in a limited number of steps

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from RandomAgent import RandomAgent
import gym
import random

class modelEvaluator:
    
    def __init__(self,agent):
        self.agent = agent
        self.ep_lengths = []


   
    def runEpisodesForAgent (self,num_episodes,numBlocks):
        ''' Runs numEpisodes of the agent
        ''' 
        
        #numBlocks = 3
        env = gym.make('BlocksWorld-v0')
        env.seed(0)
        env.reset()       
        done = False
#        num_episodes = 1000
        ep_lengths = []
        n = 0
        while (n<num_episodes):    
            steps =1
            done = False
            env.reset()
            next_action = [random.randint(0,numBlocks),random.randint(0,numBlocks)]
            while (done == False):
                obs, reward, done, empty = env.step (next_action)
                #print ('Next action ' + str(next_action))
                #print ('Obs ' + str(obs))
                next_action = self.agent.sampleAction(obs)
                #env.render()
                steps +=1    
            print (done)
            print ('New episode')
            ep_lengths.append(steps)
            n+=1
        
        print ("Average episode length " + str(sum(ep_lengths) / float(len(ep_lengths))))
            #input("Press Enter to continue...")
        self.ep_lengths = ep_lengths
        return ep_lengths