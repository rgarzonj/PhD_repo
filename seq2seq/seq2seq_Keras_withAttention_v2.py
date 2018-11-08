'''Sequence to sequence example in Keras (character-level).
This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.
# Summary of the algorithm
- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
# Data download
English to French sentence pairs.
http://www.manythings.org/anki/fra-eng.zip
Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/
# References
- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
'''
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

import keras.backend as K

K.set_session

from keras.layers.core import Reshape
from keras.layers import TimeDistributed,Activation,dot,concatenate
from keras.activations import tanh,softmax

import tensorflow as tf

batch_size = 64  # Batch size for training.
epochs = 20  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 1000  # Number of samples to train on.
# Path to the data txt file on disk.
input_data_path = "input"
target_data_path = "output"

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

#with open(input_filename) as finput:
#                with open(output_filename) as foutput:
#                    for in_line in finput:
#                        out_line = foutput.readline()

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

#target_characters.add('\t')

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(max_encoder_seq_length, num_encoder_tokens))
print ('encoder_inputs')
print (encoder_inputs)
encoder = LSTM(latent_dim, return_state=True,return_sequences=True,unroll=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
print (encoder_outputs)
print (state_h)
print (state_c)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(max_decoder_seq_length, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_state=True, return_sequences =True)
decoder_outputs, decoder_hidden, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

print ("decoder_outputs")
print (decoder_outputs)

#decoder_dense = Dense(num_decoder_tokens, activation='softmax')
#decoder_outputs = decoder_dense(decoder_outputs)

vt = K.random_normal_variable(shape=
                              (1,latent_dim), mean=0, scale=1) # Gaussian distribution (input_seq_lenth,1))
print ("vt")
print (vt)

print ("decoder_hidden")
print (decoder_hidden)
#en_seq = Reshape((-1,1,latent_dim))(encoder_outputs) #?,latent_dim
#en_seq =K.squeeze(en_seq,0)
en_seq = encoder_outputs

#en_seq = K.repeat(en_seq, max_encoder_seq_length)
print ("en_seq")
print (en_seq)

#dec_seq = Reshape((-1,1,latent_dim))(decoder_hidden)
dec_seq = K.repeat(decoder_hidden, max_encoder_seq_length)
#dec_seq = Reshape((-1,1,latent_dim))(dec_seq) 
#dec_seq = K.squeeze(dec_seq,0)
print ("dec_seq")
print (dec_seq)

blendW1 = TimeDistributed(Dense(latent_dim))(en_seq)
#blendW1 = TimeDistributed(Dense(latent_dim)(en_seq) #?,input_seq_length,latent_dim
print ('blendW1')
print (blendW1)

#blendW2 = TimeDistributed(Dense(latent_dim),ouput_dim=1)(dec_seq)
blendW2 = TimeDistributed(Dense(latent_dim))(dec_seq)
print ('blendW2')
print (blendW2)

blend3 = tanh(blendW1+blendW2)
print ("blend3")
print (blend3)
#blend3 = K.squeeze(blend3,0)
#print ("blend3 squeezed")
#print (blend3)
U = dot([blend3,vt],(0,1))
print ('U')
print (U)
U = K.squeeze(U, 0)
print ('U squeezed')
print (U)
# make probability tensor

decoder_dense = Dense(num_encoder_tokens, activation='softmax')
outputs = decoder_dense(U)

print ('outputs')
print (outputs)

#outputs = K.slice(outputs,(0,0),(max_decoder_seq_length,num_encoder_tokens))(outputs)
print ('outputs2')
print (outputs)

#pointer = softmax(U,1)
#print ("pointer")
#print (pointer)
#
#maxIndex = K.argmax(pointer,1)
#print ("maxIndex")
#print (maxIndex)
#outputs = K.gather(encoder_inputs,maxIndex)
#print ("outputs")
#print (outputs)
#outputs = Reshape((-1,outputs.shape[-1]))(outputs)
#print ("outputs reshaped")
#print (outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], U)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence



for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    #print (input_seq)
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
    
#print (model.summary())