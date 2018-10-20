#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 00:03:44 2018

@author: rgarzon
"""
import gym
import random
import subprocess
import os
import numpy as np



class BW_data_generator:
      
    def __init__(self,bwstates_path,bwopt_path,numBlocks):
        self.bwstates_command = bwstates_path + ' -n ' + str(numBlocks) 
        self.bwopt_command = bwopt_path + ' -v 3'
        self.numBlocks = numBlocks
        self.inputs = []
        self.outputs = []
    
    def _generate_random_state (self):
        """ Generates valid initial state from the implementation of Slaney & Thiébaux"""        
        proc = subprocess.Popen(self.bwstates_command,stdout=subprocess.PIPE,shell=True)   
        (out, err) = proc.communicate()
        out_str = out.decode('utf8')
        return out_str
    
    def _extract_state_from_bw_syntax(self,state):
        """Extracts just the state information from the syntax from Slaney & Thiébaux
        #Args:
        #state: String in the format
        # 4
        # 2 4 0 3
        #0     
        #Returns just the state information
        #2 4 0 3"""
        lines = state.split('\n')
        return lines[1][1:]
    
    def _get_optimal_plan(self,problem_setting):
        """Extracts just the state information from the syntax from Slaney & Thiébaux
        #Args:
        #problem_setting: String that contains current state and goal state in the format from Slaney & Thiébaux
        #Returns a list of arrays. Each array has 2 positions (block to move, destination)"""
        
        auxfilename = "auxtestfile.txt"
        file = open(auxfilename,"w")
        file.write(problem_setting)
        file.close()
        file = open(auxfilename,"r")
        proc = subprocess.Popen(self.bwopt_command,stdin=file,stdout=subprocess.PIPE,shell=True)   
        (out, err) = proc.communicate()
        out_str = out.decode('utf8')
        if os.path.exists(auxfilename):
            os.remove(auxfilename)
        lines = out_str.split('\n')
        ret = []
        plan_found = False
        for oneline in lines:
            if oneline.startswith("1:"):
                plan_found = True                                            
                dummy,block_to_move,dummy,destination =oneline.split()
                ret.append(np.array([block_to_move,destination]))
            else:
                if (plan_found==True):
                    splitResult = oneline.split()
                    if (len (splitResult)==0):
                        #print ('Parsing finished')
                        #Plan parsing has finished
                        break
                    else:
                        #print ('Parsing new line')
                        #Continue parsing
                        dummy,block_to_move,dummy,destination= splitResult
                        ret.append(np.array([block_to_move,destination]))            
        return (ret)


    def _addInputOutputToDataset (self,current_state,goal_state,output):
        """Adds input and output sequences to the class variables, then will be stored in a file
        Basically used to format all the stuff to string before saving to a file
        Returns:
           List of 2 elements, inputSequence and outputSequence added to the class variables
        """        
        inputSequence = "" 
        outputSequence = ""   
        for oneItem in current_state:
            if (len(inputSequence) == 0):
               inputSequence = oneItem              
            else:
                inputSequence = inputSequence + " " + oneItem
        for oneItem in goal_state:
            inputSequence = inputSequence + " " + oneItem
        for oneItem in output:
            if (len(outputSequence) == 0):
                outputSequence = oneItem              
            else:
                outputSequence = outputSequence + " " + oneItem
        self.inputs.append(inputSequence)
        self.outputs.append(outputSequence)
        return (inputSequence,outputSequence)
    
    def _writeSequencesToFile(self):
        """Just saves the input and output sequences to files
        Returns:
            True if the files were saved
        """
        file = open ('input',"w")
        for oneItem in self.inputs:
            file.write(oneItem + "\n")
        file.close()
        file = open('output',"w")
        for oneItem in self.outputs:
            file.write(oneItem + "\n")

        file.close()
        return True
    
    def generateVocabulary(self):
        with open('vocab', 'w') as f:
            f.write("<S>\n</S>\n<UNK>\n")
            for i in range(self.numBlocks+1):
                f.write("%d\n" % i)
        if os.path.exists('vocab'):
            return True
        else:
            return False


    def generateInputAndOutputs(self,numSequences):
        #Generates sequences of data following optimal policies
        #input format is current state + goal state Ex: "0 1 0 3 0 0" (3 blocks, current state and goal state)   
        #output format is block to be moved destination Ex: "2 0" (move block 2 to the table(0))  
        #args:
        #numSequences: Number of steps to play (one step is one sequence)        
        i = 0
        while (i <numSequences):
            episodeCompleted = False
            #Generate current state & goal state
            current_state = self._generate_random_state()
            goal_state = self._generate_random_state()
            #Trick to remove the last part of the current_state syntax
            splitted = current_state.split("\n")          
            current_state = splitted[0] + "\n" + splitted[1]           
            #We clean also the goal, we just need the string with the goal
            splitted_goal = goal_state.split("\n")
            problem_setting = current_state + goal_state
            goal_state = splitted_goal[1].split(" ")[1:]
            while (episodeCompleted==False):
                #Get the optimal plan
                plan = self._get_optimal_plan(problem_setting)
                #Play the plan to generate the sequences
                current_state_array = np.array(splitted[1].split(" ")[1:])               
                for oneAction in plan:
                    inputs,outputs = self._addInputOutputToDataset(current_state_array,goal_state,oneAction)
                    i = i + 1
                    block_to_move = int(oneAction[0])
                    destination = int(oneAction[1])
                    new_state = current_state_array
                    new_state[block_to_move-1] = destination
                    current_state_array = new_state
                episodeCompleted=True
                print ('******* Episode completed')
        self._writeSequencesToFile()
        return inputs,outputs

     
if __name__ == '__main__':    
    bwstates_path = '/Users/rgarzon/Documents/Projects/Ruben/phD/Repository/LSTMs/Blocksworld/GENERATOR/bwstates.1/bwstates'
    bwopt_path = '/Users/rgarzon/Documents/Projects/Ruben/phD/Repository/LSTMs/Blocksworld/BWOPT/optimal/bwopt'    
    numBlocks = 5
    bwgenerator = BW_data_generator(bwstates_path,bwopt_path,numBlocks)
    numSequences = 20000
    bwgenerator.generateInputAndOutputs(numSequences)
    bwgenerator.generateVocabulary()
    