from tensorflow import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28)) 
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28)) 
test_images = test_images.astype('float32') / 255

print train_images.shape, train_labels.shape, test_images.shape, test_images.shape

epochs = 5

history = model.fit(train_images, train_labels, epochs=epochs, batch_size=128)

acc = history.history['accuracy']
loss = history.history['loss']

import matplotlib.pyplot as plt 

plt.plot(range(epochs), loss, 'b', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()