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


    def _addInputOutputToPtNetDataset (self,current_state,goal_state,output):
        """Adds input and output sequences to the class variables, then will be stored in a file
        Basically used to format all the stuff to string before saving to a file
        Returns:
           List of 2 elements, inputSequence and outputSequence added to the class variables
        """        
        inputSequence = "0" 
        outputSequence = " output "   
        for oneItem in current_state:
            inputSequence = inputSequence + " " + oneItem
        inputSequence = inputSequence + " 0"
        for oneItem in goal_state:            
            inputSequence = inputSequence + " " + oneItem
        for oneItem in output:
            #Add +1 because we added the block 0 at the beginning 
            #We want to be the output to the position of the output we want to move
            outputSequence = outputSequence + str(int(oneItem)+1) + " "     
        self.inputs.append(inputSequence + outputSequence)
#        self.outputs.append(outputSequence)
        return (inputSequence,outputSequence)

    def _addPlanToInputOutputDataset (self,current_state,goal_state,plan):
        """Adds input and output sequences to the class variables, then will be stored in a file
        Basically used to format all the stuff to string before saving to a file
        Returns:
           List of 2 elements, inputSequence and outputSequence added to the class variables
        """        
        inputSequence = "0" 
        outputSequence = " output"   
        for oneItem in current_state:
            inputSequence = inputSequence + " " + oneItem
        inputSequence = inputSequence + " 0"
        for oneItem in goal_state:            
            inputSequence = inputSequence + " " + oneItem

        for oneAction in plan:
            #inputs,outputs = self._addInputOutputToPtNetDataset(current_state_array,goal_state,oneAction)
            #i = i + 1
            block_to_move = int(oneAction[0])
            destination = int(oneAction[1])
            outputSequence = outputSequence + " " + str(int(block_to_move)) + " " + str(int(destination)) 
            #new_state = current_state_array
            #new_state[block_to_move-1] = destination
            #current_state_array = new_state

#        for oneItem in output:
            #Add +1 because we added the block 0 at the beginning 
            #We want to be the output to the position of the output we want to move
#            outputSequence = outputSequence + str(int(oneItem)+1) + " "     
        self.inputs.append(inputSequence + outputSequence)
#        self.outputs.append(outputSequence)
        return (inputSequence,outputSequence)

    def _writencoptSequencesToFile(self):
        """Just saves the input and output sequences to files
        Returns:
            True if the files were saved
        """
        file = open ('input_ncopt_'+ str(self.numBlocks) + ".txt","w")
        for oneItem in self.inputs:
            file.write(oneItem + "\n")
        file.close()
        return True
    

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
            plan = self._get_optimal_plan(problem_setting)
            if (len(plan)>0):
                current_state_array = np.array(splitted[1].split(" ")[1:])  
                inputs,outputs = self. _addPlanToInputOutputDataset(current_state_array,goal_state,plan)
                i = i + 1
                print ('Episode ' + str(i))
        self._writencoptSequencesToFile()
        return inputs,outputs


    def ptnetInputOutputFiles(self):
        with open('input', 'r', encoding='utf-8') as f:
            lines_in = f.read().split('\n')
            
        with open('output', 'r', encoding='utf-8') as f:
            lines_out = f.read().split('\n')
        i = 0
        for line in lines_in:
            if (len(line)>0):
                inp = self._PtNetInputDataset(line,lines_out[i])
                self.inputs_generalized.append(inp)
                self.outputs_generalized.append(outp)
            i = i + 1            
        file = open ('input_ptnet_',"w")
        for oneItem in self.inputs_generalized:
            oneItem = self._padInputWithCharacter(oneItem,'0')
            file.write(oneItem + "\n")
        file.close()
#        file = open('output_ptnet_' + numBlocks,"w")
#        for oneItem in self.outputs_generalized:
#            file.write(oneItem + "\n")
#        file.close()
        return True

         
if __name__ == '__main__':    
    bwstates_path = '/Users/rgarzon/Documents/Projects/Ruben/phD/Repository/LSTMs/Blocksworld/GENERATOR/bwstates.1/bwstates'
    bwopt_path = '/Users/rgarzon/Documents/Projects/Ruben/phD/Repository/LSTMs/Blocksworld/BWOPT/optimal/bwopt'    
    numBlocks = 3
    bwgenerator = BW_data_generator(bwstates_path,bwopt_path,numBlocks)
    numSequences = 100
    bwgenerator.generateInputAndOutputs(numSequences)
    #bwgenerator.generalizeInputOutputFiles()
    #bwgenerator.generateVocabulary()
    