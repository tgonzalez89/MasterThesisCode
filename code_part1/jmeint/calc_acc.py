#!/usr/bin/python3

import sys

with open(sys.argv[1]) as f:
    lines = f.readlines()
data1 = [int(line.strip()) for line in lines]
with open(sys.argv[2]) as f:
    lines = f.readlines()
data2 = [int(line.strip()) for line in lines]

matches = 0
for i in range(len(data1)):
    if data1[i] == data2[i]:
        matches += 1

acc = matches / len(data1)
error = 1.0 - acc

print(f"Error: {100*error:.2f} %")
