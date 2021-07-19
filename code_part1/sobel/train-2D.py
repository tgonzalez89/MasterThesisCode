#!/usr/bin/python3

import sys
sys.path.insert(0, '../common')
from histogram import Histogram
import cv2
import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.activations import relu
from tensorflow.keras import optimizers


# Helper functions

def apply_sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    #retval, sobel = cv2.threshold(sobel, 3.0, 3.0, cv2.THRESH_TRUNC)
    return sobel


# Parameters
block_size = 16
data_dir = 'data'

# Get and prepare training data
print(f"\nGetting and preparing training data...\n")
imagepaths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.png')]
inputs = []
outputs = []
np.random.shuffle(imagepaths)
for imagepath in imagepaths:
    img_full = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_full = img_full / 255.0
    for x in range(0, img_full.shape[1], block_size):
        for y in range(0, img_full.shape[0], block_size):
            img = img_full[y:y+block_size, x:x+block_size]
            filtered_img = apply_sobel(img)
            inputs.append(img)
            outputs.append(filtered_img)

inputs = np.array(inputs, dtype='float32')
outputs = np.array(outputs, dtype='float32')
inputs = np.expand_dims(inputs, -1)
outputs = np.expand_dims(outputs, -1)
samples = inputs.shape[0]
print('Input shape: ', inputs.shape)
print('Output shape:', outputs.shape)

# Train the model
print(f"\nTraining the model...\n")
try:
    while True:
        model = Sequential()
        model.add(Conv2D(8, (3,3), activation='relu', padding='same', input_shape=inputs.shape[1:]))
        model.add(Conv2D(4, (1,1), activation='relu', padding='same'))
        model.add(Conv2D(1, (1,1), activation='linear', padding='same'))
        model.add(ReLU(max_value=4.47))
        optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss='mse')
        model.summary()
        for epoch in range(128):
            print(f"Epoch #{epoch}")
            history = model.fit(inputs, outputs, batch_size=8, epochs=1, validation_split=0.2)
            if history.history['loss'][0] >= 0.15:
                print("Starting weights/biases not good. Trying again...")
                break
            if epoch >= 7 and history.history['loss'][0] >= 0.05:
                print("Not converging fast enough. Trying again...")
                break
            if epoch >= 15 and history.history['loss'][0] >= 0.02:
                print("Not converging fast enough. Trying again...")
                break
            if epoch >= 31 and history.history['loss'][0] >= 0.015:
                print("Not converging fast enough. Trying again...")
                break
            if epoch >= 63 and history.history['loss'][0] >= 0.01:
                print("Not converging fast enough. Trying again...")
                break
            if history.history['loss'][0] < 0.0025:
                print(f"Acceptable state reached. Breaking early...")
                break
        if history.history['loss'][0] < 0.005:
            break

except KeyboardInterrupt as e:
    print("Stopping...")

# Test the model
print(f"\nTesting the model...\n")
actual    = outputs[samples*4//5:]
predicted = model.predict(inputs[samples*4//5:])
errors = []
for r in range(actual.shape[0]):
    for i in range(actual.shape[1]):
        for j in range(actual.shape[2]):
            for c in range(actual.shape[3]):
                a = actual[r][i][j][c]
                p = predicted[r][i][j][c]
                errors.append(abs(a-p))
print(f"Min err: {min(errors):.6f} Max err: {max(errors):.6f} Avg err: {sum(errors)/len(errors):.6f}")
print(f"WMAPE: {100*sum(errors)/actual.sum():.2f}%")
Histogram(errors, 10).show(precision=6, max_bar_len=40)

# Save the model to a file
model.save("nn_sobel-2D.h5", )
