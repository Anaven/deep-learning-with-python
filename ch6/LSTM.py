from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
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

# lstm
lstm_model = Sequential() 
lstm_model.add(Embedding(max_features, 32)) 
lstm_model.add(LSTM(32)) 
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
lstm_history = lstm_model.fit(input_train, y_train,
    epochs=10, batch_size=128, validation_split=0.2)

# reverse lstm
r_input_train = [x[::-1] for x in input_train]
r_input_test = [x[::-1] for x in input_test]
r_input_train = sequence.pad_sequences(r_input_train, maxlen=maxlen) 
r_input_test = sequence.pad_sequences(r_input_test, maxlen=maxlen) 

rlstm_model = Sequential() 
rlstm_model.add(Embedding(max_features, 32)) 
rlstm_model.add(LSTM(32))
rlstm_model.add(Dense(1, activation='sigmoid'))
rlstm_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
rlstm_history = rlstm_model.fit(r_input_train, y_train,
    epochs=10, batch_size=128, validation_split=0.2)

# bi lstm
bilstm_model = Sequential() 
bilstm_model.add(Embedding(max_features, 32)) 
bilstm_model.add(Bidirectional(LSTM(32))) 
bilstm_model.add(Dense(1, activation='sigmoid'))
bilstm_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
bilstm_history = bilstm_model.fit(input_train, y_train,
    epochs=10, batch_size=128, validation_split=0.2)

import matplotlib.pyplot as plt

lstm_val_acc = lstm_history.history['val_acc'] 
rlstm_val_acc = rlstm_history.history['val_acc'] 
bilstm_val_acc = bilstm_history.history['val_acc'] 

epochs = range(1, len(lstm_val_acc) + 1)

plt.title('validation accuracy') 
plt.plot(epochs, lstm_val_acc, 'b', label='LSTM') 
plt.plot(epochs, rlstm_val_acc, 'r', label='RLSTM') 
plt.plot(epochs, bilstm_val_acc, 'y', label='BiLSTM') 
plt.xlabel('Epochs')
plt.ylabel('Val Loss')
plt.legend()
plt.show()