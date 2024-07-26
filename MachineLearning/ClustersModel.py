import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

train_df = pd.read_csv('./DataFolder/clusters/data/train.csv')
np.random.shuffle(train_df.values)

test_df = pd.read_csv('./DataFolder/clusters/data/test.csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

mapping = {'red': 0, 'blue': 1, 'green': 2, 'teal': 3, 'orange': 4, 'purple': 5}
train_df['color'] = train_df.color.apply(lambda i: mapping[i])
test_df['color'] = test_df.color.apply(lambda i: mapping[i])

model = keras.Sequential([
    keras.layers.Dense(32, input_shape=(2,), activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(6, activation='sigmoid')])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values))

model.fit(x, train_df.color.values, batch_size=4, epochs=5)

model.evaluate(test_x, test_df.color.values)
