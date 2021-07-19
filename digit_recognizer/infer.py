import cv2
import json
import jsonschema
import os
import sys
import numpy as np
import tensorflow as tf
from collections import deque

video_file = sys.argv[1]
video_bn = str(os.path.basename(video_file))
#bbox = [[551, 648, 95, 95], [632, 644, 95, 95]]  # ../../Videos/tesis/train.mp4
#bbox = [[490, 570, 95, 95], [570, 570, 95, 95]]  # ../../Videos/tesis/fruit/fruit_tf_sep_conv.mp4
#x = int(sys.argv[2])
#y = int(sys.argv[3])
#dist = int(sys.argv[4])
#thresh = int(sys.argv[2])
lower_limit = float(sys.argv[2])
upper_limit = float(sys.argv[3])
show = len(sys.argv) > 4 and sys.argv[4] == "show"
model = tf.keras.models.load_model('digits.h5')

settings = {}
settings_schema = {
    "type": "object",
    "properties": {
        "digit_roi": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {
                    "type": "integer"
                }
            }
        },
        "threh": {
            "type": "integer"
        },
        "min_thresh": {
            "type": "integer"
        },
        "max_thresh": {
            "type": "integer"
        }
    },
    "required": [
        "digit_roi",
        "thresh",
        "min_thresh",
        "max_thresh"
    ]
}
with open('settings.json', 'r') as json_fp:
    settings = json.load(json_fp)
    if video_bn not in settings:
        print(f"ERROR: No settings found for video {video_bn}")
        exit()
    else:
        jsonschema.validate(settings[video_bn], settings_schema)
        bbox = settings[video_bn]['digit_roi']
        thresh = settings[video_bn]['thresh']

video = cv2.VideoCapture(video_file)
fps = video.get(cv2.CAP_PROP_FPS)
#print("FPS: ", fps)
results_q = deque(maxlen=15)
ret, frame = video.read()
results = []
#means = []
#count = 0
print("Processing frames...")
while ret:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #digit1 = frame[y:y+100, x:x+100]
    digit1 = frame[bbox[0][1]:bbox[0][1]+bbox[0][3], bbox[0][0]:bbox[0][0]+bbox[0][2]]
    digit1 = cv2.medianBlur(digit1, 5)
    ret, digit1 = cv2.threshold(digit1, thresh, 255, cv2.THRESH_BINARY)
    digit1 = cv2.resize(digit1, (28, 28), interpolation=cv2.INTER_CUBIC)
    digit1 = digit1.reshape((1, digit1.shape[0], digit1.shape[1], 1))

    #digit2 = frame[y:y+100, x+dist:x+dist+100]
    digit2 = frame[bbox[1][1]:bbox[1][1]+bbox[1][3], bbox[1][0]:bbox[1][0]+bbox[1][2]]
    digit2 = cv2.medianBlur(digit2, 5)
    ret, digit2 = cv2.threshold(digit2, thresh, 255, cv2.THRESH_BINARY)
    digit2 = cv2.resize(digit2, (28, 28), interpolation=cv2.INTER_CUBIC)
    digit2 = digit2.reshape((1, digit2.shape[0], digit2.shape[1], 1))

    _result1 = model.predict(digit1)
    _result2 = model.predict(digit2)
    result1 = np.argmax(_result1)
    result2 = np.argmax(_result2)
    result = float(f"{result1}.{result2}")
    results_q.append(result)
    #mean = round(sum(results_q) / len(results_q), 1)
    if show:
        #print(result, mean)
        print(f"{result}", end='\r')
        #print(f"{result} {_result1[0][result1]:.2f} {_result2[0][result2]:.2f}", end='\r')
        #print(f"{result} {_result1[0][result1]:.2f} {_result2[0][result2]:.2f}")
    if result >= lower_limit and result <= upper_limit:
        results.append(result)
    #if mean >= lower_limit and mean <= upper_limit:
    #    means.append(mean)

    if show:
        digit1 = digit1.reshape((digit1.shape[1], digit1.shape[2], 1))
        digit2 = digit2.reshape((digit2.shape[1], digit2.shape[2], 1))
        line = np.ones((28, 2, 1)) * 255
        line = line.astype(np.uint8)
        show_img = np.concatenate((digit1, line, digit2), axis=1)
        show_img = cv2.cvtColor(show_img, cv2.COLOR_GRAY2BGR)
        cv2.line(show_img, (29, 0), (29, 28), (0, 255, 0), thickness=2)
        cv2.imshow("Digits", show_img)
        cv2.waitKey(1)

    #cv2.imwrite(f"{count}.png", digit1)
    #count += 1

    ret, frame = video.read()
print('')

results_avg = round(sum(results) / len(results), 1)
#means_avg = round(sum(means) / len(means), 1)
print(f"results average = {results_avg}")
#print(f"means average = {means_avg}")
