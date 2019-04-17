from tensorflow import keras
import numpy as np

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

print train_data.shape, train_labels.shape, test_data.shape, test_labels.shape

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# x_train[0] array([ 0., 1., 1., ..., 0., 0., 0.])
print x_train.shape, x_test.shape

model = keras.models.Sequential() 
model.add(keras.layers.Dense(16, activation='relu', input_shape=(10000,))) 
model.add(keras.layers.Dense(16, activation='relu')) 
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
    loss='binary_crossentropy', 
    metrics=['accuracy'])

x_val = x_train[:10000] 
partial_x_train = x_train[10000:]

y_val = train_labels[:10000] 
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
    partial_y_train, 
    epochs=20, 
    batch_size=512, 
    validation_data=(x_val, y_val))

print model.evaluate(x_test, test_labels)

history_dict = history.history

import matplotlib.pyplot as plt

history_dict = history.history 
loss_values = history_dict['loss'] 
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'b', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()