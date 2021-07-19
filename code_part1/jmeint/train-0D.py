#!/usr/bin/python3

import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

with open('data/inputs_1000K.txt') as f:
    lines = f.readlines()
inputs = [list(map(float, line.split())) for line in lines[1:500001]]
with open('data/outputs_1000K.txt') as f:
    lines = f.readlines()
outputs = [int(line) for line in lines[:500000]]

inputs = np.array(inputs)
outputs = np.array(outputs)

while True:
    model = Sequential()
    model.add(Dense(18, activation='relu', input_shape=inputs.shape[1:]))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    model.summary()
    acc = []
    val_acc = []
    for epoch in range(600):
        print(f"Epoch #{epoch}")
        history = model.fit(inputs, outputs, batch_size=200, epochs=1, validation_split=0.2)
        acc.append(history.history['binary_accuracy'][0])
        val_acc.append(history.history['val_binary_accuracy'][0])
        if acc[epoch] < 0.7:
            print("Starting weights/biases not good. Trying again...")
            break
        if epoch > 0 and (abs(acc[epoch-1] - acc[epoch]) <= 0.0001 and acc[epoch] > 0.86):
            print(f"Acceptable acc reached ({acc[epoch]}) and small acc change detected ({abs(acc[epoch-1] - acc[epoch])}). Stopping...")
            break
        if epoch >= 199 and acc[epoch] < 0.84:
            print(f"Not converging fast enough. Trying again...")
            break
    if (abs(acc[epoch-1] - acc[epoch]) <= 0.0001 and acc[epoch] > 0.86):
        break
    if epoch >= 599:
        print(f"Done. Final accuracy = {acc[epoch]}")
        break

model.save("nn_jmeint.h5")
