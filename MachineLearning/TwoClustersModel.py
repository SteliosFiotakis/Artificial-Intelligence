import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

train_df = pd.read_csv('./DataFolder/clusters_two_categories/data/train.csv')
train_features = np.column_stack((train_df.x.values, train_df.y.values))
train_one_hot_color = pd.get_dummies(train_df.color).values
train_one_hot_marker = pd.get_dummies(train_df.marker).values
train_labels = np.concatenate((train_one_hot_color, train_one_hot_marker), axis=1)
np.random.RandomState(seed=42).shuffle(train_features)
np.random.RandomState(seed=42).shuffle(train_labels)

test_df = pd.read_csv('./DataFolder/clusters_two_categories/data/test.csv')
test_features = np.column_stack((test_df.x.values, test_df.y.values))
test_one_hot_color = pd.get_dummies(test_df.color).values
test_one_hot_marker = pd.get_dummies(test_df.marker).values
test_labels = np.concatenate((test_one_hot_color, test_one_hot_marker), axis=1)
np.random.RandomState(seed=42).shuffle(test_features)
np.random.RandomState(seed=42).shuffle(test_labels)


model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(2,), activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(9, activation='sigmoid')])

model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_features, train_labels, batch_size=4, epochs=10)

model.evaluate(test_features, test_labels)

print("Correct: (red, ^),Prediction: ", np.round(model.predict(np.array([[0, 1]]))))
print("Correct: (blue, ^),Prediction: ", np.round(model.predict(np.array([[0, 5]]))))
print("Correct: (green, +),Prediction: ", np.round(model.predict(np.array([[-2, 1]]))))
print("Correct: (teal, +), Prediction: ", np.round(model.predict(np.array([[-2, 5]]))))
print("Correct: (orange, +),Prediction: ", np.round(model.predict(np.array([[-2, 3]]))))
print("Correct: (purple, *),Prediction: ", np.round(model.predict(np.array([[0, 3]]))))
