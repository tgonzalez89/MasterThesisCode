#!/usr/bin/python3

import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras import optimizers

block_size = 10

with open('data/inputs_1000K.txt') as f:
    lines = f.readlines()
inputs = [list(map(float, line.split())) for line in lines[1:]]
with open('data/outputs_1000K.txt') as f:
    lines = f.readlines()
outputs = [[int(line)] for line in lines]

inputs = np.array(inputs)
outputs = np.array(outputs)
inputs = np.expand_dims(inputs, -1)
outputs = np.expand_dims(outputs, -1)
inputs = np.reshape(inputs, (inputs.shape[0]//block_size, inputs.shape[1]*block_size, inputs.shape[2]))
outputs = np.reshape(outputs, (outputs.shape[0]//block_size, outputs.shape[1]*block_size, outputs.shape[2]))

while True:
    model = Sequential()
    model.add(Conv1D(36, 18, strides=18, padding='valid', activation='relu', input_shape=inputs.shape[1:]))
    model.add(Conv1D(18, 1, strides=1, padding='valid', activation='relu'))
    model.add(Conv1D(1, 1, strides=1, padding='valid', activation='hard_sigmoid'))
    optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='mse', metrics=['binary_accuracy'])
    model.summary()
    acc = []
    val_acc = []
    for epoch in range(300):
        print(f"Epoch #{epoch}")
        history = model.fit(inputs[:500000//block_size], outputs[:500000//block_size], batch_size=10, epochs=1, validation_split=0.2)
        acc.append(history.history['binary_accuracy'][0])
        val_acc.append(history.history['val_binary_accuracy'][0])
        if acc[epoch] < 0.7:
            print("Starting weights/biases not good. Trying again...")
            break
        if epoch > 0 and (abs(acc[epoch-1] - acc[epoch]) <= 0.0001 and acc[epoch] > 0.86):
            print(f"Acceptable acc reached ({acc[epoch]}) and small acc change detected ({abs(acc[epoch-1] - acc[epoch])}). Stopping...")
            break
        if epoch >= 149 and acc[epoch] < 0.84:
            print(f"Not converging fast enough. Trying again...")
            break
    if (abs(acc[epoch-1] - acc[epoch]) <= 0.0001 and acc[epoch] > 0.86):
        break
    if epoch >= 299:
        print(f"Done. Final accuracy = {acc[epoch]}")
        break

results = model.predict(inputs[600000//block_size:600000//block_size+20])
for i in range(600000//block_size,600000//block_size+20):
    real = outputs[i][0][0]
    pred = results[i-600000//block_size][0][0]
    pred_round = int(round(pred))
    res = 'match' if real == pred_round else 'error'
    print(f"Real: {real} Predicted: {pred_round} ({pred:.2f}) {res}")

data = inputs[900000//block_size:]
results = model.predict(data)
matches = 0
for i in range(len(data)):
    for j in range(outputs.shape[1]):
        if outputs[i+900000//block_size][j][0] == int(round(results[i][j][0])):
            matches += 1
acc = matches / (len(data) * outputs.shape[1])
print(f"Accuracy: {100*acc:.2f} %")

model.save("nn_jmeint-1D.h5")
