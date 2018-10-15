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

class BasicSeq2SeqAgent:
    '''Implements an agent with a seq2seq model trained on optimal plans
   '''
    
    def __init__(self,modelPath,modelConfig):
        """ 
        Args:
        modelPath: path to the Keras trained model
        modelConfig: dictionary with the following keys: latent_dim,input_characters,target_characters,max_decoder_seq_length

        """
        self.modelConfig = modelConfig
        
        self.input_token_index = dict(
                [(char, i) for i, char in enumerate(modelConfig["input_characters"])])
        self.target_token_index = dict(
                [(char, i) for i, char in enumerate(modelConfig["target_characters"])])
        
        self.num_encoder_tokens = len(self.modelConfig["input_characters"])
        self.num_decoder_tokens = len(self.modelConfig["target_characters"])
       
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(modelConfig['latent_dim'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]
        
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(modelConfig['latent_dim'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
                
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(modelConfig['latent_dim'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]
        
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(modelConfig['latent_dim'],))
        decoder_state_input_c = Input(shape=(modelConfig['latent_dim'],))
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

        self.model = load_model(modelPath)
    
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
               len(decoded_sentence) > self.modelConfig["max_decoder_seq_length"]):
                stop_condition = True
    
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
    
            # Update states
            states_value = [h, c]
    
        return decoded_sentence
    
    def predict_sequence(self,input_seq):
        encoder_input_data = np.zeros(
                (1, self.modelConfig["max_encoder_seq_length"], self.num_encoder_tokens),
                dtype='float32')
        print (encoder_input_data)
        print (encoder_input_data.shape)
        for t,char in enumerate(input_seq):
            print (t)
            encoder_input_data[0, t, self.input_token_index[char]] = 1.
        print (encoder_input_data)
        print (type(encoder_input_data))
        decoded_sentence = self._decode_sequence(encoder_input_data)
        return (decoded_sentence)
        