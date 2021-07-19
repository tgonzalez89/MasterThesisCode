#!/usr/bin/python3

import sys
sys.path.insert(0, '../common')
from histogram import Histogram
import numpy as np
import os

from tensorflow.keras.layers import Conv1D, concatenate
from tensorflow.keras import optimizers, Input, Model

block_size = 10

with open('data/inputs_nn_1000K.txt') as f:
    lines = f.readlines()
inputs = [list(map(float, line.split())) for line in lines]
with open('data/inputs_1000K.txt') as f:
    lines = f.readlines()
outputs = [list(map(float, line.split())) for line in lines[1:]]

inputs = np.array(inputs)
outputs = np.array(outputs)
print(inputs.shape)
print(outputs.shape)
inputs = np.expand_dims(inputs, -1)
outputs = np.expand_dims(outputs, -1)
print(inputs.shape)
print(outputs.shape)
inputs = np.reshape(inputs, (inputs.shape[0]//block_size, inputs.shape[1]*block_size, inputs.shape[2]))
outputs = np.reshape(outputs, (outputs.shape[0]//block_size, block_size, outputs.shape[1]))
print(inputs.shape)
print(outputs.shape)

try:
    while True:
        nn_inputs = Input(shape=inputs.shape[1:])
        x = Conv1D(32, 2, strides=2, padding='valid', activation='relu')(nn_inputs)
        x = Conv1D(8, 1, strides=1, padding='valid', activation='relu')(x)
        x = Conv1D(1, 1, strides=1, padding='valid', activation='elu')(x)
        y = Conv1D(32, 2, strides=2, padding='valid', activation='relu')(nn_inputs)
        y = Conv1D(8, 1, strides=1, padding='valid', activation='relu')(y)
        y = Conv1D(1, 1, strides=1, padding='valid', activation='elu')(y)
        nn_outputs = concatenate([x, y])
        model = Model(inputs=[nn_inputs], outputs=[nn_outputs])
        optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mape'])
        model.summary()
        acc = []
        val_acc = []
        for epoch in range(1000):
            print(f"Epoch #{epoch}")
            for i in range(10):
                history = model.fit(inputs[i*1000000//10//block_size:(i+1)*1000000//10//block_size], outputs[i*1000000//10//block_size:(i+1)*1000000//10//block_size], batch_size=10, epochs=1, validation_split=0.2)
            '''acc.append(history.history['mean_absolute_percentage_error'][0])
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
            break'''
        break
except KeyboardInterrupt as e:
    print("Stopping...")

results = model.predict(inputs[900000//block_size:900000//block_size+20])
for i in range(900000//block_size,900000//block_size+20):
    for o in range(2):
        real = outputs[i][0][o]
        pred = results[i-900000//block_size][0][o]
        error = abs(real-pred)
        error_perc = 100 * error / real if real > 0 else 100
        print(f"Real: {real:.6f} Predicted: {pred:.6f} Error: {error:.6f} ({error_perc:.2f}%)")

data = inputs[800000//block_size:]
results = model.predict(data)
errors = []
errors_perc = []
for i in range(len(data)):
    for j in range(outputs.shape[1]):
        for o in range(2):
            real = outputs[i+800000//block_size][j][o]
            pred = results[i][j][o]
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

model.save("nn_inversek2j-1D.h5")
