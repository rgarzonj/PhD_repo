from __future__ import print_function

from keras.models import Model
from keras.layers import LSTM, Input
from keras.callbacks import LearningRateScheduler
from keras.utils.np_utils import to_categorical
from PointerLSTM import PointerLSTM
import pickle
import numpy as np

def scheduler(epoch):
    if epoch < nb_epochs/4:
        return learning_rate
    elif epoch < nb_epochs/2:
        return learning_rate*0.5
    return learning_rate*0.1

#from keras.layers.core import Reshape
#from keras.layers import TimeDistributed,Activation
#from keras.activations import tanh,softmax

#import tensorflow as tf

#batch_size = 64  # Batch size for training.
#epochs = 20  # Number of epochs to train for.
#latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 100  # Number of samples to train on.
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
    input_text_int = []
    target_text_int = []
    input_text = line.split(" ")
    target_text = lines_out[i].split(" ")
    #target_text.insert(0,'\t')
    #target_text.append('\n')
    #print (input_text)
    #print (target_text)
    i = i + 1
    for element in input_text:
        input_text_int.append(int(element))
    for element in target_text:
        target_text_int.append(int(element))
    for a in range(9):
        target_text_int.append(target_text_int[0])
        target_text_int.append(target_text_int[1])
#    target_texts.append(target_text)
    target_texts.append(target_text_int)
#    input_texts.append(input_text)
    input_texts.append(input_text_int)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(int(char))
    for char in target_text:
        if char not in target_characters:
            target_characters.add(int(char))

#target_characters.add('\t')

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = 10
#num_encoder_tokens = len(input_characters)
#num_decoder_tokens = len(target_characters)
num_decoder_tokens = num_encoder_tokens
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
#
#encoder_input_data = np.zeros(
#    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
#    dtype='float32')
#decoder_input_data = np.zeros(
#    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
#    dtype='float32')
#decoder_target_data = np.zeros(
#    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
#    dtype='float32')
#
#for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
#    for t, char in enumerate(input_text):
#        encoder_input_data[i, t, input_token_index[char]] = 1.
#    for t, char in enumerate(target_text):
#        # decoder_target_data is ahead of decoder_input_data by one timestep
#        decoder_input_data[i, t, target_token_index[char]] = 1.
#        if t > 0:
#            # decoder_target_data will be ahead by one timestep
#            # and will not include the start character.
#            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
X = np.array(input_texts)
X = np.reshape(X,(num_samples,max_encoder_seq_length,-1))
print (X.shape)
print (X[0])

Y = np.array(target_texts)
print (Y[0])
print (Y.shape)

YY = []
for y in Y:
    #print (y)
    #print (type(y))
    #print (y.shape)
    YY.append(to_categorical(y,num_decoder_tokens))
YY = np.asarray(YY)
print (YY[0])
print (YY.shape)


hidden_size = 128
seq_len = 20
nb_epochs = 50
learning_rate = 0.1

print("building model...")
main_input = Input(shape=(seq_len, 1), name='main_input')

encoder = LSTM(output_dim = hidden_size, return_sequences = True, name="encoder")(main_input)
decoder = PointerLSTM(hidden_size, output_dim=hidden_size, name="decoder")(encoder)

model = Model(input=main_input, output=decoder)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#model.fit(X, YY, nb_epoch=nb_epochs, batch_size=64,callbacks=[LearningRateScheduler(scheduler),])
ret = model.fit(X, YY, nb_epoch=nb_epochs, batch_size=64)

#acc = 0
#while (acc<0.9))
#    ret = ret = model.fit(X, YY, nb_epoch=nb_epochs, batch_size=64) 
#    acc = np.mean(ret.history['acc'][-5:])

x_test =X[0:10]

y_test = []
Y_test = Y[0:10]
for y in Y_test:
    y_test.append(to_categorical(y,num_decoder_tokens))
y_test = np.asarray(y_test)  
y_test = np.float32(y_test)
predictions = model.predict(x_test)

#print(model.predict(x_test))
#print (y_test)
print("------")
#print(to_categorical(y_test,num_decoder_tokens))
model.save_weights('model_weights_bad.h5')


del model  # deletes the existing model
# returns a compiled model
# identical to the previous one


print("building model...")
main_input = Input(shape=(seq_len, 1), name='main_input')

encoder = LSTM(output_dim = hidden_size, return_sequences = True, name="encoder")(main_input)
decoder = PointerLSTM(hidden_size, output_dim=hidden_size, name="decoder")(encoder)

model = Model(input=main_input, output=decoder)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights('model_weights.h5')
predictions2 = model.predict(x_test)

print (model.summary())