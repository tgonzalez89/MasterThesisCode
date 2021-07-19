#!/usr/bin/python3

import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model_shape = [int(s) for s in sys.argv[1].split(',')]

model = Sequential()
model.add(Dense(model_shape[1], activation='relu', input_shape=(model_shape[0],)))
for s in model_shape[2:]:
    model.add(Dense(s, activation='relu'))
model.compile(optimizer='sgd', loss='mse')
model.summary()
