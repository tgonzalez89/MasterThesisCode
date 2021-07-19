#!/usr/bin/python3

import numpy as np
import os
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

def filter_image_sobel(img):
    # Perform filtering to the input image
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    return sobel

# Get the paths to the training images
data_dir = 'data'
imagepaths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.png')]

# Load and pre-process the training data
images = []
filteredimages = []

np.random.shuffle(imagepaths)
for imagepath in imagepaths:
    img_full = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_full = img_full / 255.0
    filtered_img_full = filter_image_sobel(img_full)
    for y in range(img_full.shape[0]-2):
        for x in range(img_full.shape[1]-2):
            img = [img_full[y-1][x-1],img_full[y-1][x  ],img_full[y-1][x+1],
                   img_full[y  ][x-1],img_full[y  ][x  ],img_full[y  ][x+1],
                   img_full[y+1][x-1],img_full[y+1][x  ],img_full[y+1][x+1]]
            filtered_img = filtered_img_full[y][x]
            images.append(img)
            filteredimages.append(filtered_img)

images = np.array(images, dtype='float32')
filteredimages = np.array(filteredimages, dtype='float32')

while True:
    model = Sequential()
    model.add(Dense(9, activation='relu', input_shape=images.shape[1:]))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(1, activation='relu'))
    optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    loss = []
    val_loss = []
    for epoch in range(100):
        history = model.fit(images, filteredimages, batch_size=512, epochs=1, validation_split=0.2)
        loss.append(history.history['loss'][0])
        val_loss.append(history.history['val_loss'][0])
        if loss[epoch] > 0.03:
            print("Starting weights/biases not good. Trying again...")
            break
        if epoch > 0 and (abs(loss[epoch-1] - loss[epoch]) < 0.0005 and loss[epoch] < 0.005):
            print(f"Acceptable loss reached ({loss[epoch]}) and small loss change detected ({abs(loss[epoch-1] - loss[epoch])}). Stopping...")
            break
    if (abs(loss[epoch-1] - loss[epoch]) < 0.0005 and loss[epoch] < 0.005):
        break

model.save("nn_sobel-0D.h5")
