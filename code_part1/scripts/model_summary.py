#!/usr/bin/python3

import sys
from tensorflow import keras

model = keras.models.load_model(sys.argv[1])

model.summary()
