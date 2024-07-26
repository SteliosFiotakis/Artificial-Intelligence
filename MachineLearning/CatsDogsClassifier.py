from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
from PartitionEs import tr_dogs_df, tr_cats_df, te_dogs_df, te_cats_df

# Data Preparation

# tr_cats_df = []
# tr_dogs_df = []
#
# for i in range(1, 4001):
#     temp_cat = cv2.imread(f'./CatsAndDogs/training_set/cats/cat.{str(i)}.jpg')
#     tr_cats_df.append(temp_cat)
#     temp_dog = cv2.imread(f'./CatsAndDogs/training_set/dogs/dog.{str(i)}.jpg')
#     tr_dogs_df.append(temp_dog)
#
# te_cats_df = []
# te_dogs_df = []
#
# for i in range(4001, 5001):
#     temp_cat = cv2.imread(f'./CatsAndDogs/training_set/cats/cat.{str(i)}.jpg')
#     te_cats_df.append(temp_cat)
#     temp_dog = cv2.imread(f'./CatsAndDogs/training_set/dogs/dog.{str(i)}.jpg')
#     te_dogs_df.append(temp_dog)

print(len(tr_cats_df))
print(len(tr_dogs_df))

print(len(te_cats_df))
print(len(te_dogs_df))

'''
# Model Creation

model = keras.Sequential([
    keras.layer.Dense(512, activation='relu'),
    keras.layer.Dense(512, activation='relu'),
    keras.layer.Dense(512, activation='relu'),
    keras.layer.Dense(2, activation='softmax')])

# Model Tuning

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy,
              metrics=['accuracy'])

# Model Training

model.fit()

# Model Validation

model.evaluation()
'''