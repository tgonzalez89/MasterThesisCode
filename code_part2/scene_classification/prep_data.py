import glob
import os
import cv2 as cv
import numpy as np
from random import shuffle
from tensorflow.keras.utils import to_categorical


def prep_data(size=150, data_type='train'):
    print("Processing data...")
    all_data = []
    data_types = {'train': 'seg_train', 'test': 'seg_test'}
    data_path = f"dataset/{data_types[data_type]}/"
    categories = os.listdir(data_path)
    categories.sort()
    for label, category in enumerate(categories):
        category_path = data_path + category + '/'
        image_files = os.listdir(category_path)
        for image_file in image_files:
            img = cv.imread(category_path + image_file)
            img = cv.resize(img, (size, size), interpolation=cv.INTER_CUBIC)
            all_data.append((label, img))

    indexes = [i for i in range(len(all_data))]
    shuffle(indexes)
    labels_s = [all_data[i][0] for i in indexes]
    imgs_s = [all_data[i][1] for i in indexes]

    labels = to_categorical(labels_s)
    data = np.array(imgs_s)

    return data, labels, categories
