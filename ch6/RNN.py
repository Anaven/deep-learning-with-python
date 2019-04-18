import numpy as np

# timesteps = 100
# input_features = 32
# output_features = 64

# inputs = np.random.random((timesteps, input_features))
# state_t = np.zeros((output_features,))

# W = np.random.random((output_features, input_features))
# U = np.random.random((output_features, output_features))
# b = np.random.random((output_features,))

# successive_outputs = []
# for input_t in inputs:
#     output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
#     successive_outputs.append(output_t)
#     state_t = output_t

# final_output_sequence = np.stack(successive_outputs, axis=0)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, LSTM
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence

max_features = 10000
maxlen = 500
batch_size = 32

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences') 
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)') 
input_train = sequence.pad_sequences(input_train, maxlen=maxlen) 
input_test = sequence.pad_sequences(input_test, maxlen=maxlen) 
print('input_train shape:', input_train.shape) 
print('input_test shape:', input_test.shape)

# rnn
rnn_model = Sequential() 
rnn_model.add(Embedding(max_features, 32)) 
rnn_model.add(SimpleRNN(32)) 
rnn_model.add(Dense(1, activation='sigmoid'))
rnn_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
rnn_history = rnn_model.fit(input_train, y_train,
    epochs=10, batch_size=128, validation_split=0.2)

# lstm
lstm_model = Sequential() 
lstm_model.add(Embedding(max_features, 32)) 
lstm_model.add(LSTM(32)) 
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
lstm_history = lstm_model.fit(input_train, y_train,
    epochs=10, batch_size=128, validation_split=0.2)

import matplotlib.pyplot as plt

rnn_val_acc = rnn_history.history['val_acc'] 
lstm_val_acc = lstm_history.history['val_acc'] 

epochs = range(1, len(rnn_val_acc) + 1)

plt.title('validation accuracy') 
plt.plot(epochs, rnn_val_acc, 'b', label='RNN') 
plt.plot(epochs, lstm_val_acc, 'r', label='LSTM') 
plt.xlabel('Epochs')
plt.ylabel('Val Loss')
plt.legend()
plt.show()