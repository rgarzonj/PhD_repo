#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:33:33 2018

@author: rgarzon
"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from keras.models import load_model
import os
import random

MAX_NUM_BLOCKS = 10
#Consider the current and goal state (*2) and also the blank spaces between state representation (*2)
MAX_INPUT_SEQ_LENGTH = 4*MAX_NUM_BLOCKS-1 #-1 because the last block do not have a blank space after it

class BasicSeq2SeqAgent:
    '''Implements an agent with a seq2seq model trained on optimal plans
   '''
    
    def __init__(self,numBlocks):
        """ 
        Args:

        """
        self.numBlocks = numBlocks
        batch_size = 64  # Batch size for training.
        epochs = 40  # Number of epochs to train for.
        latent_dim = 256  # Latent dimensionality of the encoding space.
        num_samples = 10000  # Number of samples to train on.
        # Hack to support calling the file from the tests folder or the root folder
        if (os.path.isfile('input')):
            input_data_path = "input"
            target_data_path = "output"
            vocab_data_path = "vocab"
        else:
            input_data_path = "../input"
            target_data_path = "../output"
            vocab_data_path = "../vocab"
        # Vectorize the data.
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
               
        with open(input_data_path, 'r', encoding='utf-8') as f:
            lines_in = f.read().split('\n')
            
        with open(target_data_path, 'r', encoding='utf-8') as f:
            lines_out = f.read().split('\n')
        
        i = 0
        for line in lines_in[: min(num_samples, len(lines_in) - 1)]:
            input_text = line.split(" ")
            target_text = lines_out[i].split(" ")
            target_text.insert(0,'\t')
            target_text.append('\n')
            #print (input_text)
            #print (target_text)
            i = i + 1
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)
        with open(vocab_data_path,'r',encoding='utf-8') as f:
            lines_in = f.read().split('\n')
        for char in lines_in:
            if (char not in input_characters):
                input_characters.add(char)
            if (char not in target_characters):
                target_characters.add(char)
                
        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in target_texts])
        
        print('Number of samples:', len(input_texts))
        print('Number of unique input tokens:', self.num_encoder_tokens)
        print('Number of unique output tokens:', self.num_decoder_tokens)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)
        
        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])
        
        encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')



        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.        
        
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]
        
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
                
                # Define sampling models
        self.encoder_model = Model(encoder_inputs, encoder_states)
        
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        
        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())
        
        

    def _decode_sequence(self,input_seq_encoded):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq_encoded)
    
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index['\t']] = 1.
    
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)
    
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char
    
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
               len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True
    
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
    
            # Update states
            states_value = [h, c]
    
        return decoded_sentence
    
    def _padInputWithCharacter (self,inputSeq,padChar):
        # Generate data method feeds an string as inputSeq
        # OpenAI Gym feeds an array as inputSeq
        # If we receive an array we move it to a string first
        returnString = True
        if (isinstance(inputSeq,np.ndarray)):
            returnString = False
            newSeq = str(inputSeq[0]) 
            for item in inputSeq[1:]:
                newSeq = newSeq + " " + str(item)
            inputSeq = newSeq
            #print (newSeq + 'EEEEE')
            #InputSeq should be a string
            #print (inputSeq)
            #print (type(inputSeq))
        ret = inputSeq
        #TODO Support more than 10 blocks
        if (len(inputSeq)<MAX_INPUT_SEQ_LENGTH):
            numBlocks = int((1+len(inputSeq))/4)      
            #We need to pad with unknowns
            ret_current = inputSeq[0:numBlocks*2-1]
            ret_goal = inputSeq[numBlocks*2:]
            #Multiply by 2 because need to consider the blank spaces
            for i in range(MAX_NUM_BLOCKS-numBlocks-1):
                ret_current = ret_current + ' ' + padChar
                ret_goal = ret_goal + ' ' + padChar
            ret = ret_current + ' ' + padChar + ' ' + ret_goal + ' ' + padChar
        #We must return a list of vocabulary items or a string
        if returnString==True:
            #When generating data
            return ret
        else:
            #When infering
            return ret.split(' ')
    
    
    def sampleAction(self,input_seq):        
        # Input seq is an int64 array when called from the openAI Gym

        #print ('Input sequence')
        #print (input_seq)
        #print (type(input_seq[0]))
        #print (type(input_seq))
        # Although the agent is trained on optimal plans
        # we may reach the situation where the model do not emit a good action for the current state
        # and this could take the episode to never complete because the agent is always suggesting the same
        # incorrect action. We add an exploration factor to unlock this case
        input_seq = self._padInputWithCharacter(input_seq,'0')
        r = random.random()
        if (r<0.1):
            return([random.randint(0,self.numBlocks),random.randint(0,self.numBlocks)])
        encoder_input_data = np.zeros(
                (1, self.max_encoder_seq_length, self.num_encoder_tokens),
                dtype='float32')
        #print (encoder_input_data)
        #print (encoder_input_data.shape)
        #print (self.input_token_index)        
        for t,char in enumerate(input_seq):
            #print (t)
            #print (char)
            #print (type(char))
            if isinstance(char, (int, np.integer)):
                char = str(char)
            #print (t)
            #print (char)
            #print (type(char))
            #print (self.input_token_index[char])
            encoder_input_data[0, t, self.input_token_index[char]] = 1.
        #print (type(encoder_input_data))
        #print (encoder_input_data)
        decoded_sentence = self._decode_sequence(encoder_input_data)
        ret = []
        #TODO This implies a the limitation to 10 blocks because every char encodes the oneblock
        for char in decoded_sentence:            
            #print ('START')
            #print (char)
            #print ('END')
            if (char.isdigit()):
                ret.append(int(char))
            else:
                ret.append(0)
           
        return ([ret[0],ret[1]])
    
    