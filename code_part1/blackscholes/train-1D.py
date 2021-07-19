#!/usr/bin/python3

import sys
sys.path.insert(0, '..')
from histogram import Histogram
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras import optimizers

block_size = 10

with open('data/inputs_nn_200K.txt') as f:
    lines = f.readlines()
inputs = [list(map(float, line.split())) for line in lines]
with open('data/outputs_200K.txt') as f:
    lines = f.readlines()
outputs = [[float(line)] for line in lines]

inputs = np.array(inputs)
outputs = np.array(outputs)
inputs = np.expand_dims(inputs, -1)
outputs = np.expand_dims(outputs, -1)
inputs = np.reshape(inputs, (inputs.shape[0]//block_size, inputs.shape[1]*block_size, inputs.shape[2]))
outputs = np.reshape(outputs, (outputs.shape[0]//block_size, outputs.shape[1]*block_size, outputs.shape[2]))

try:
    while True:
        model = Sequential()
        model.add(Conv1D(32, 6, strides=6, padding='valid', activation='relu', input_shape=inputs.shape[1:]))
        model.add(Conv1D(16, 1, strides=1, padding='valid', activation='relu'))
        model.add(Conv1D(1, 1, strides=1, padding='valid', activation='relu'))
        optimizer = optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mape'])
        model.summary()
        acc = []
        val_acc = []
        for epoch in range(1000):
            print(f"Epoch #{epoch}")
            history = model.fit(inputs[:200000//block_size], outputs[:200000//block_size], batch_size=10, epochs=1, validation_split=0.2)
            acc.append(history.history['mean_absolute_percentage_error'][0])
            val_acc.append(history.history['val_mean_absolute_percentage_error'][0])
            if history.history['loss'][0] >= 0.001:
                print("Starting weights/biases not good. Trying again...")
                break
            if epoch >= 49 and acc[epoch] >= 22:
                print(f"Not converging fast enough. Trying again...")
                break
            if epoch >= 499 and acc[epoch] >= 14:
                print(f"Not converging fast enough. Trying again...")
                break
            if acc[epoch] < 12.5 and val_acc[epoch] < 12.5:
                break
        if acc[epoch] < 12.5 and val_acc[epoch] < 12.5:
            print(f"Acceptable acc reached ({acc[epoch]}). Stopping...")
            break
        if epoch >= 999:
            print(f"Done. Final acc = {acc[epoch]}")
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

model.save("nn_blackscholes-1D.h5")
