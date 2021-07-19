#!/usr/bin/python3

import os
import cv2
import sys
import numpy as np


with open(sys.argv[1]) as f:
    rgb_img = f.readlines()
width, height = [int(i) for i in rgb_img[0].split(',')]
metadata = eval(rgb_img[-1])
if type(metadata) == str:
    metadata = eval(metadata)
data = np.array([row.split(',') for row in rgb_img[1:-1]], dtype='uint8')
data = data.reshape(height, width, 3)
out = os.path.splitext(sys.argv[1])[0] + '.png'
if len(sys.argv[2]) > 2:
    out = sys.argv[2]
cv2.imwrite(out, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
