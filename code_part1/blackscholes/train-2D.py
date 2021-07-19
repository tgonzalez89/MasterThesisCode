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
in_size = 6
out_size = 1

# Get and prepare training data
print(f"\nGetting and preparing training data...\n")
with open('data/inputs_nn_200K.txt') as f:
    lines = f.readlines()
inputs = np.array([list(map(float, line.split())) for line in lines], dtype='float32')
with open('data/outputs_200K.txt') as f:
    lines = f.readlines()
outputs = np.array([[float(line)] for line in lines], dtype='float32')

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
        model.add(Conv2D(32, (1,in_size), strides=(1,in_size), padding='valid', activation='relu', input_shape=inputs.shape[1:]))
        model.add(Conv2D(16, (1,1), strides=(1,1), padding='valid', activation='relu'))
        model.add(Conv2D(out_size, (1,1), strides=(1,1), padding='valid', activation='relu'))
        optimizer = optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mape'])
        model.summary()
        for epoch in range(128):
            print(f"Epoch #{epoch}")
            history = model.fit(inputs, outputs, batch_size=1, epochs=1, validation_split=0.2)
            if history.history['loss'][0] >= 0.001:
                print("Starting weights/biases not good. Trying again...")
                break
            if epoch >= 31 and history.history['loss'][0] >= 1e-5:
                print("Not converging fast enough. Trying again...")
                break
            if history.history['loss'][0] < 5e-6:
                print("Acceptable state reached. Breaking early...")
                break
        if history.history['loss'][0] < 6e-6:
            break

except KeyboardInterrupt as e:
    print("Stopping...")

# Test the model
print(f"\nTesting the model...\n")
actual    = outputs[samples*4//5:]
predicted = model.predict(inputs[samples*4//5:])
matches = 0
for r in range(actual.shape[0]):
    for i in range(actual.shape[1]):
        for j in range(actual.shape[2]):
            for c in range(actual.shape[3]):
                a = actual[r][i][j][c]
                p = predicted[r][i][j][c]
                if a == p:
                    matches += 1
acc = matches / actual.size
print(f"Accuracy: {100*acc:.2f} %")

# Save the model to a file
model.save("nn_blackscholes-2D.h5")
