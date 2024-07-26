import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

print(train_data[0])

word_index = data.get_word_index()

word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


print(decode_review(test_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

#
# train_images = train_images/255.0
# test_images = test_images/255.0
#
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(10, activation="softmax"),
#     ])
#
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#
# model.fit(train_images, train_labels, epochs=5)
#
# prediction = model.predict(test_images)
# print(class_names[np.argmax(prediction[512])])
#
# print(class_names[test_labels[512]])
# plt.imshow(test_images[512], cmap=plt.cm.binary)
# plt.show()
