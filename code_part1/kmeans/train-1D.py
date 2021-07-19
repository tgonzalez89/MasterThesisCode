#!/usr/bin/python3

import sys
sys.path.insert(0, '../common')
from histogram import Histogram
import numpy as np
import os
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras import optimizers

def euclideanDistance(inputs, i):
    r = 0.0
    r += (inputs[i+0] - inputs[i+3]) ** 2
    r += (inputs[i+1] - inputs[i+4]) ** 2
    r += (inputs[i+2] - inputs[i+5]) ** 2
    return math.sqrt(r)

block_size = 10
inputs_size = 6000000

np.random.seed(1)
inputs = np.random.randint(low=0, high=1000000, size=inputs_size) / 1000000
outputs = np.array([euclideanDistance(inputs, i) for i in range(0, inputs_size, 6)])

inputs = np.reshape(inputs, (inputs_size//6//block_size, 6*block_size, 1))
outputs = np.reshape(outputs, (inputs_size//6//block_size, block_size, 1))

try:
    while True:
        model = Sequential()
        model.add(Conv1D(10, 6, strides=6, padding='valid', activation='relu', input_shape=inputs.shape[1:]))
        model.add(Conv1D(5, 1, strides=1, padding='valid', activation='relu'))
        model.add(Conv1D(1, 1, strides=1, padding='valid', activation='relu'))
        optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mape'])
        model.summary()
        acc = []
        val_acc = []
        for epoch in range(100):
            print(f"Epoch #{epoch}")
            history = model.fit(inputs, outputs, batch_size=10, epochs=1, validation_split=0.2)
            if history.history['mean_absolute_percentage_error'][0] < 1.0:
                break
        break

except KeyboardInterrupt as e:
    print("Stopping...")

results = model.predict(inputs[100000//block_size:100000//block_size+20])
for i in range(100000//block_size,100000//block_size+20):
    real = outputs[i][0][0]
    pred = results[i-100000//block_size][0][0]
    error = abs(real-pred)
    error_perc = 100 * error / real if real > 0 else 100
    print(f"Real: {real:.6f} Predicted: {pred:.6f} Error: {error:.6f} ({error_perc:.2f}%)")

data = inputs[100000//block_size:]
results = model.predict(data)
errors = []
errors_perc = []
for i in range(len(data)):
    for j in range(outputs.shape[1]):
        real = outputs[i+100000//block_size][j][0]
        pred = results[i][j][0]
        error = abs(real-pred)
        errors.append(error)
        if error == 0.0:
            error_perc = 0.0
        else:
            error_perc = 100 * error / real if real > 0 else 100
        errors_perc.append(error_perc)
print(f"Min err: {min(errors):.6f} Max err: {max(errors):.6f} Avg err: {sum(errors)/len(errors):.6f}")
print(f"Min err_per: {min(errors_perc):.2f} Max err_per: {max(errors_perc):.2f} Avg err_per: {sum(errors_perc)/len(errors_perc):.2f}")
Histogram(errors, 10).show(precision=6, max_bar_len=40)
Histogram(errors_perc, 10, 0, 100, True).show(max_bar_len=40)

model.save("nn_kmeans-1D.h5")
