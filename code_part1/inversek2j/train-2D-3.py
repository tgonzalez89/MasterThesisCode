#!/usr/bin/python3

import sys
sys.path.insert(0, '../common')
from histogram import Histogram
import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import optimizers

# Parameters
block_size = 5
in_size = 2
out_size = 2

# Get and prepare training data
print(f"\nGetting and preparing training data...\n")
with open('data/inputs_nn_1000K.txt') as f:
    lines = f.readlines()
inputs = np.array([list(map(float, line.split())) for line in lines], dtype='float32')
with open('data/inputs_1000K.txt') as f:
    lines = f.readlines()
outputs = np.array([list(map(float, line.split())) for line in lines[1:]], dtype='float32')

samples = inputs.shape[0] // (block_size*block_size)
inputs  = np.reshape(inputs, (samples, block_size, block_size*in_size, 1))
outputs = np.reshape(outputs, (samples, block_size, block_size, out_size))
print('Input shape: ', inputs.shape)
print('Output shape:', outputs.shape)

# Train the model
print(f"\nTraining the model...\n")
try:
    while True:
        model = Sequential()
        model.add(Conv2D(8, (1,in_size), strides=(1,in_size), padding='valid', activation='relu', input_shape=inputs.shape[1:]))
        model.add(Conv2D(8, (1,1), strides=(1,1), padding='valid', activation='relu'))
        model.add(Conv2D(8, (1,1), strides=(1,1), padding='valid', activation='relu'))
        model.add(Conv2D(out_size, (1,1), strides=(1,1), padding='valid', activation='relu'))
        optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mape'])
        model.summary()
        for epoch in range(1000):
            print(f"Epoch #{epoch}")
            history = model.fit(inputs, outputs, batch_size=5, epochs=1, validation_split=0.2)
            if history.history['loss'][0] >= 0.02:
                print("Starting weights/biases not good. Trying again...")
                break
            if epoch >= 25 and history.history['loss'][0] >= 0.001:
                print("Not converging fast enough. Trying again...")
                break
            if epoch >= 50 and history.history['loss'][0] >= 0.0008:
                print("Not converging fast enough. Trying again...")
                break
            if epoch >= 100 and history.history['loss'][0] >= 0.0007:
                print("Not converging fast enough. Trying again...")
                break
            '''if history.history['loss'][0] < 0.0005:
                print("Acceptable state reached. Breaking early...")
                break'''
        if history.history['loss'][0] < 0.0005:
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
model.save("nn_inversek2j-2D-3.h5")
