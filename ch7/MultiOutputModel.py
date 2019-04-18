from tensorflow.keras import layers 
from tensorflow.keras import Input 
from tensorflow.keras.models import Model
from tensorflow.keras import utils
import numpy as np



income_size = 500
num_samples = 1000
max_length = 100

vocabulary_size = 10000
num_income_groups = 10
age_size = 10

# input
posts = np.random.randint(1, vocabulary_size, size=(num_samples, max_length))

# output
age_targets = np.random.randint(age_size, size=(num_samples))
income_targets = np.random.randint(num_income_groups, size=(num_samples))
income_targets = utils.to_categorical(income_targets, num_income_groups)
gender_targets = np.random.randint(2, size=(num_samples))

print posts.shape, age_targets.shape

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(vocabulary_size, 256)(posts_input)
x = layers.Conv1D(128, 3, padding='same', activation='relu')(embedded_posts)
x = layers.MaxPooling1D(3)(x)
x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling1D(3)(x)
x = layers.Conv1D(256, 2, activation='relu')(x)
x = layers.Conv1D(256, 2, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

# output layers have names
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups,
    activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, 
    activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

model.compile(optimizer='rmsprop', loss={'age': 'mse', 
    'income': 'categorical_crossentropy', 
    'gender': 'binary_crossentropy'})

history = model.fit(posts, {'age': age_targets, 'income': income_targets, 'gender': gender_targets}, 
    epochs=10, batch_size=64)

print history.history

# import matplotlib.pyplot as plt

# epochs = range(1, len(val_acc) + 1)

# plt.title('validation accuracy') 
# plt.plot(epochs, val_acc, 'b', label='RNN') 
# plt.xlabel('Epochs')
# plt.ylabel('Val Loss')
# plt.legend()
# plt.show()