import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 30  # 10
IMG_WIDTH = 32  # 30
IMG_HEIGHT = 32  # 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.1  # 0.4


def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit('Usage: python traffic.py data_directory [model.h5]')

    images, labels = load_data(sys.argv[1])

    labels = tf.keras.utils.to_categorical(labels)
    categories = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), categories,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=np.array(labels)
    )

    model = get_model()
    # model.fit(x_train, y_train, epochs=EPOCHS)

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, validation_split=0.1, callbacks=[es, rlr])

    model.evaluate(x_test, y_test, verbose=2)

    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f'Model saved to {filename}.')

    plot_training(history)


def load_data(data_dir):
    """
    Load image data from the directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
     the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    print(f'(load_data) start on data_dir: {data_dir}')

    images, labels = [], []
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        for file_path in os.listdir(category_dir):
            image = cv2.imread(os.path.join(category_dir, file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            image = image.astype('float32') / 255.0
            images.append(image)
            labels.append(int(category))

    print(f'(load_data) end. Total images: {len(images)}')
    return images, labels


def _gen_model_v1():
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.Flatten(),
        layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    return model


def _gen_model_v2():
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    return model


def _gen_model_v3():
    from tensorflow.keras import layers, models
    from keras.src.layers import Dropout

    model = models.Sequential([
        layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    return model


def _gen_model_v4():
    from tensorflow.keras import layers, models
    from keras import regularizers

    weight_decay = 1e-4

    model = models.Sequential([
        layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        layers.Conv2D(
            32, (3, 3),
            activation='relu', padding='same',
            kernel_regularizer=regularizers.l2(weight_decay)
        ),
        layers.Conv2D(
            32, (3, 3),
            activation='relu', padding='same',
            kernel_regularizer=regularizers.l2(weight_decay)
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(
            64, (3, 3),
            activation='relu', padding='same',
            kernel_regularizer=regularizers.l2(weight_decay)
        ),
        layers.Conv2D(
            64, (3, 3),
            activation='relu', padding='same',
            kernel_regularizer=regularizers.l2(weight_decay)
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    return model

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    print('(get_model) start')

    model = _gen_model_v4()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f'(get_model) end')
    return model


def plot_training(history):
    import matplotlib.pyplot as plt

    # plt.plot(history.history['accuracy'], label='train acc')
    # plt.plot(history.history['val_accuracy'], label='val acc')
    # plt.legend()
    # plt.show()

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    sys.argv = ['traffic.py', './gtsrb']
    main()
