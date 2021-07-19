#!/usr/bin/python3

import sys
sys.path.insert(0, '../common')
from histogram import Histogram

with open(sys.argv[1]) as f:
    lines1 = f.readlines()
with open(sys.argv[2]) as f:
    lines2 = f.readlines()
if len(lines1) == len(lines2) + 1:
    lines1 = lines1[1:]
data1 = [[float(t.strip()) for t in l.split()] for l in lines1]
data2 = [[float(t.strip()) for t in l.split()] for l in lines2]

errors = []
data1_no_nan = []
for i in range(len(data1)):
    for j in range(len(data1[i])):
        error = abs(data1[i][j]-data2[i][j])
        if str(error) != 'nan':
            errors.append(error)
            data1_no_nan.append(data1[i][j])

perc_errors = []
for i in range(len(data1)):
    for j in range(len(data1[i])):
        error = abs(data1[i][j]-data2[i][j])
        actual_val = data1[i][j]
        if actual_val == 0.0:
            if error == 0.0:
                perc_error = 0.0
            else:
                perc_error = 1.0
        elif str(actual_val) == 'nan':
            continue
        elif str(error) == 'nan':
            perc_error = 1.0
        else:
            perc_error = error / actual_val
            if perc_error > 1.0:
                perc_error = 1.0
        perc_errors.append(perc_error)

square_errors = []
for error in errors:
    square_errors.append(error**2)

print(f"Min err: {min(errors):.6f} Max err: {max(errors):.6f} Avg err: {sum(errors)/len(errors):.6f}")
print(f"WMAPE: {100*sum(errors)/sum(data1_no_nan):.2f}%")
print(f"MAPE:  {100*sum(perc_errors)/len(perc_errors):.2f}%")
print(f"MSE:  {100*sum(square_errors)/len(square_errors):.8f}")

#Histogram(errors, 10).show(precision=6, max_bar_len=40)
#Histogram(perc_errors, 10).show(precision=6, max_bar_len=40)
