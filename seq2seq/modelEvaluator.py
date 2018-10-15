#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:30:06 2018

@author: rgarzon
"""

import tensorflow as tf
from tensorflow.python import debug as tf_debug

class modelEvaluator:
    
    def __init__(self,model_fn,model_dir,params):
        self.estimator = self._loadModel(model_fn,model_dir,params)
        
    def _loadModel(self,model_fn,model_dir,params):
        est = tf.estimator.Estimator(
                model_fn=model_fn,
                model_dir=model_dir, params=params)
        print (est)
        return (est)               
        
    def predict (self,inputs):        
        #BATCH_SIZE = 1
#        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#                x={"input": inputs},
#                num_epochs=1,
#                shuffle=False,
#                batch_size=BATCH_SIZE)
        hooks = [tf_debug.LocalCLIDebugHook(ui_type="readline")]
        predictions = self.estimator.predict(input_fn=inputs,predict_keys=['pred_outputs'],hooks=hooks)
        print (predictions)
        return (predictions)
    
    def runEpisodesRandomAgent (numEpisodes):
        ''' Runs numEpisodes of the agent
        ''' 
        numBlocks = 3
        env = gym.make('BlocksWorld-v0')
        env.seed(0)
        env.reset()


#done = False
#num_episodes = 1000
#ep_lengths = []
#n = 0
#while (n<num_episodes):    
#    steps =1
#    done = False
#    env.reset()
#    while (done == False):
#        next_action = [random.randint(0,numBlocks),random.randint(0,numBlocks)]    
#        obs, reward, done, empty = env.step (next_action)    
#        print ('Next action ' + str(next_action))
#        print ('Obs ' + str(obs))
#        #env.render()
#        steps +=1    
#    print (done)
#    print ('New episode')
#    ep_lengths.append(steps)
#    n+=1
#
#print ("Average episode length " + str(sum(ep_lengths) / float(len(ep_lengths))))
#    #input("Press Enter to continue...")
