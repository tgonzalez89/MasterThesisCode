import cv2
import glob
import numpy as np
import tensorflow as tf
from random import shuffle

print("Loading data...")

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = 255 - train_data
test_data = 255 - test_data
train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))

# Load custom dataset
max_imgs = 3800
new_train_data = []
new_train_labels = []
new_test_data = []
new_test_labels = []
for digit in range(10):
    print(f"Loading class {digit}")
    images = glob.glob(f"dataset/{digit}/*.png")
    shuffle(images)
    count = 0
    total = len(images)
    for f in images:
        count += 1
        if count % 10 == 0:
            print(f"Loading {count}/{total}", end='\r')
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
        img = img.reshape(img.shape[0], img.shape[1], 1)
        if count < max_imgs:
            new_train_data.append(img)
            new_train_labels.append(digit)
        else:
            new_test_data.append(img)
            new_test_labels.append(digit)
    print('')
train_data = np.append(train_data, new_train_data, axis=0)
train_labels = np.append(train_labels, new_train_labels, axis=0)
test_data = np.append(test_data, new_test_data, axis=0)
test_labels = np.append(test_labels, new_test_labels, axis=0)

# Transform labels from single int to probabilities
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Shuffle training dataset
all_train_data = [(train_data[i], train_labels[i]) for i in range(train_data.shape[0])]
indexes = [i for i in range(len(all_train_data))]
shuffle(indexes)
train_data = [all_train_data[i][0] for i in indexes]
train_labels = [all_train_data[i][1] for i in indexes]
train_data = np.array(train_data)
train_labels = np.array(train_labels)

print("Training...")

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(25, kernel_size=(5, 5), padding='same', activation='relu', input_shape=train_data.shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(25, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(25, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(train_labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_data, train_labels, batch_size=64, epochs=10, validation_split=0.1)
model.save(f'digits.h5', )

print("Evaluating...")
result = model.evaluate(test_data, test_labels)
print(result)
