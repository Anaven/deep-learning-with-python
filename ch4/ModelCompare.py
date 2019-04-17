from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) 
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# large model
orig_model = keras.models.Sequential() 
orig_model.add(keras.layers.Dense(16, activation='relu', input_shape=(10000,))) 
orig_model.add(keras.layers.Dense(16, activation='relu')) 
orig_model.add(keras.layers.Dense(1, activation='sigmoid'))

orig_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
    loss='binary_crossentropy', 
    metrics=['accuracy'])

# small model
small_model = keras.models.Sequential() 
small_model.add(keras.layers.Dense(4, activation='relu', input_shape=(10000,))) 
small_model.add(keras.layers.Dense(4, activation='relu')) 
small_model.add(keras.layers.Dense(1, activation='sigmoid'))

small_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
    loss='binary_crossentropy', 
    metrics=['accuracy'])

# regular model
reg_model = keras.models.Sequential() 
reg_model.add(keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), 
    activation='relu', input_shape=(10000,))) 
reg_model.add(keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
    activation='relu')) 
reg_model.add(keras.layers.Dense(1, activation='sigmoid'))

reg_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
    loss='binary_crossentropy', 
    metrics=['accuracy'])

x_val = x_train[:10000] 
y_val = train_labels[:10000]

partial_x_train = x_train[10000:]
partial_y_train = train_labels[10000:]

orig_history = orig_model.fit(partial_x_train,
    partial_y_train, 
    epochs=20, 
    batch_size=512, 
    validation_data=(x_val, y_val))

small_history = small_model.fit(partial_x_train,
    partial_y_train, 
    epochs=20, 
    batch_size=512, 
    validation_data=(x_val, y_val))

reg_history = reg_model.fit(partial_x_train,
    partial_y_train, 
    epochs=20, 
    batch_size=512, 
    validation_data=(x_val, y_val))

orig_history_dict = orig_history.history
small_history_dict = small_history.history 
reg_history_dict = reg_history.history 

orig_val_loss_values = orig_history_dict['val_loss'] 
small_val_loss_values = small_history_dict['val_loss']
reg_val_loss_values = reg_history_dict['val_loss']

epochs = range(1, len(orig_val_loss_values) + 1)

plt.plot(epochs, orig_val_loss_values, 'b', label='Ori 2-16')
plt.plot(epochs, small_val_loss_values, 'r', label='Small 2-4')
plt.plot(epochs, reg_val_loss_values, 'y', label='Ori Reg 2-16')
plt.title('Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Val Loss')
plt.legend()

plt.show()