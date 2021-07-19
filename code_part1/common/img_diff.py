#!/usr/bin/python3

import cv2
import os
import sys
import numpy as np
from histogram import Histogram


def run_diff(img1, img2):
    height = img1.shape[0]
    width = img1.shape[1]
    channels = img1.shape[2] if len(img1.shape) >= 3 else 1
    img1 = np.reshape(img1, (height, width, channels))
    img2 = np.reshape(img2, (height, width, channels))

    diff = cv2.absdiff(img1, img2)
    diff = np.reshape(diff, (height, width, channels))

    if channels == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    elif channels == 4:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGRA2GRAY)
    else:
        img1_gray = img1
        diff_gray = diff

    diff_sum = np.sum(diff_gray)
    img1_sum = np.sum(img1_gray)
    wmape = diff_sum / img1_sum
    
    def calc_mape(error, actual_val):
        error = float(error)
        actual_val = float(actual_val)
        if actual_val == 0.0:
            if error == 0.0:
                perc_error = 0.0
            else:
                perc_error = 1.0
        else:
            perc_error = error / actual_val
            if perc_error > 1.0:
                perc_error = 1.0
        return perc_error

    calc_mape_v = np.vectorize(calc_mape)
    errors = calc_mape_v(diff_gray, img1_gray)
    mape = np.sum(errors) / errors.size

    def calc_mse(error):
        error = float(error)
        return error**2

    calc_mse_v = np.vectorize(calc_mse)
    squared_errors = calc_mse_v(diff_gray)
    mse = np.sum(squared_errors) / squared_errors.size

    print(f"Min err: {np.min(diff):.0f} Max err: {np.max(diff):.0f} Avg err: {diff_sum/diff.size:.2f}")
    print(f"WMAPE: {100*wmape:.2f}%")
    print(f"MAPE:  {100*mape:.2f}%")
    print(f"MSE:  {100*mse:.8f}")
    #Histogram(diff.flatten(), 10).show(precision=0, max_bar_len=40)
    #Histogram(errors.flatten(), 10).show(precision=6, max_bar_len=40)

    return diff, diff_gray


image1 = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
image2 = cv2.imread(sys.argv[2], cv2.IMREAD_UNCHANGED)
img_name = sys.argv[3] if len(sys.argv) >= 4 else 'diff.png'
img_name_gray = os.path.splitext(img_name)[0]+'_gray'+os.path.splitext(img_name)[1]
diff, diff_gray = run_diff(image1, image2)
cv2.imwrite(img_name, diff)
if len(image1.shape) >= 3:
    cv2.imwrite(img_name_gray, diff_gray)

'''retval, th = cv2.threshold(diff, 16, 255, cv2.THRESH_BINARY)
img_name_th = os.path.splitext(img_name)[0]+'_th'+os.path.splitext(img_name)[1]
print(f"Error: {100*np.count_nonzero(th)/th.size:.2f}%")
cv2.imwrite(img_name_th, th)'''
