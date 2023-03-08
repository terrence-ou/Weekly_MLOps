import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess


if __name__ == "__main__":

    print("Tensorflow version: ", tf.__version__)

    # MNIST Data
    fashion_mnist = keras.datasets.fashion_mnist
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

    train_x = train_x / 255.0
    test_x = test_x / 255.0
    
    train_x = np.expand_dims(train_x, axis=-1)
    test_x = np.expand_dims(test_x, axis=-1)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print('\ntrain_images.shape: {}, of {}'.format(train_x.shape, train_x.dtype))
    print('test_images.shape: {}, of {}'.format(test_x.shape, test_x.dtype))

    # Train and evaluate
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=3,
                            strides=2, activation="relu", name="Conv1"),
        keras.layers.Flatten(),
        keras.layers.Dense(10, name="Dense")
    ])

    model.summary()

    epochs = 5
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    model.fit(train_x, train_y, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_x, test_y)
    print(f"Test accuracy: {test_acc}")

    # Save model
    MODEL_DIR = "./saved_model"
    version = 1
    export_path = os.path.join(MODEL_DIR, "version_" + str(version))
    print(f"export path = {export_path} \n")

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    print("Model saved!")