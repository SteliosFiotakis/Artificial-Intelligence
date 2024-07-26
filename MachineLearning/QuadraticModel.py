import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

train_df = pd.read_csv('./DataFolder/quadratic/data/train.csv')
np.random.shuffle(train_df.values)

test_df = pd.read_csv('./DataFolder/quadratic/data/test.csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

model = keras.Sequential([
    keras.layers.Dense(32, input_shape=(2,), activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values))

model.fit(x, train_df.color.values, batch_size=4, epochs=10)

model.evaluate(test_x, test_df.color.values)
