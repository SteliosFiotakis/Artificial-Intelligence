import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# model = keras.Sequential([
#     keras.Input(shape=(32, 32, 3)),
#     keras.layers.Conv2D(32, 3, activation='relu'),
#     keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     keras.layers.Conv2D(64, 3, activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Conv2D(128, 3, activation='relu'),
#     keras.layers.Flatten(),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(10)])

model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)])


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, batch_size=64, epochs=15)

model.evaluate(x_test, y_test, batch_size=64)
