import glob
import cv2 as cv
import numpy as np
from os import path
from random import shuffle
from scipy.io import wavfile
from scipy import signal
from tensorflow.keras.utils import to_categorical


def prep_data(data_type='train', size=64, nperseg=128, noverlap=64, save=False):
    print("Processing data...")
    all_data = []
    data_path = f"free-spoken-digit-dataset/recordings/{data_type}_data/"
    for fn in glob.glob(data_path + '*.wav'):
        sample_rate, data = wavfile.read(fn)

        if data.dtype == np.uint8:
            data = (data - 2**7) / 2**7
        elif data.dtype == np.int16:
            data = data / 2**15
        elif data.dtype == np.int32:
            data = data / 2**31
        data = data.astype(np.float32)

        f, t, spec = signal.spectrogram(data, sample_rate, nperseg=nperseg, noverlap=noverlap)
        spec = np.array(spec, order='C')
        spec = np.flip(spec, axis=0)
        spec = cv.normalize(spec, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
        spec = cv.resize(spec, (size, size), interpolation=cv.INTER_CUBIC)
        spec = np.reshape(spec, (spec.shape[0], spec.shape[1], 1))

        #img = cv.normalize(spec, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        #cv.imwrite(f"{path.basename(fn)}.png", img)
        #np.savetxt(f"{path.basename(fn)}.csv", spec, delimiter=',')

        label = int(str(path.basename(fn))[0])

        all_data.append((label, spec))

    indexes = [i for i in range(len(all_data))]
    shuffle(indexes)
    labels_s = [all_data[i][0] for i in indexes]
    specs_s = [all_data[i][1] for i in indexes]

    labels = to_categorical(labels_s)
    data = np.array(specs_s)

    if save:
        np.savetxt(f'{data_path}/labels.csv', data, delimiter=',')
        np.savetxt(f'{data_path}/data.csv', data, delimiter=',')

    return data, labels
