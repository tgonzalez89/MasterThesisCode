import cv2
import json
import jsonschema
import numpy as np
import os
import sys
import termios
import time
import threading

lock = threading.Lock()
frame = None
mod_frame = None
digit_roi_img = [None, None]
img_id = 0
frames = []
digit_roi = [[0, 0, 0, 0], [0, 0, 0, 0]]
annotation = [None, None]
min_thresh = 0
max_thresh = 255
rf_ret = False
mark = 0
marked_ids = [0, 0]
settings = {}
if not os.path.isfile('settings.json'):
    with open('settings.json', 'w') as json_fp:
        json.dump(settings, json_fp)
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

def read_frame():
    global frame
    r, f = video.read()
    if r:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        f = frame
    return r, f

def update_frame():
    global frame, mod_frame, digit_roi, digit_roi_img
    with lock:
        #contrast = cv2.getTrackbarPos('contrast', win_name) - 128
        thresh = cv2.getTrackbarPos('thresh', win_name)
        #blockSize = cv2.getTrackbarPos('blockSize', win_name) * 2 + 3
        #C = cv2.getTrackbarPos('C', win_name) - 25

        #mod_frame = frame / (4/3)
        #mod_frame = mod_frame.astype(np.uint8)
        mod_frame = frame.copy()
        digit_roi_img = [None, None]

        for i in range(2):
            if digit_roi[i][2] == 0 or digit_roi[i][3] == 0:
                continue

            digit_roi_img[i] = frame[digit_roi[i][1]:digit_roi[i][1]+digit_roi[i][3], digit_roi[i][0]:digit_roi[i][0]+digit_roi[i][2]]

            digit_roi_img[i] = cv2.medianBlur(digit_roi_img[i], 5)

            #if contrast != 0:
            #    f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            #    alpha_c = f
            #    gamma_c = 127*(1-f)
            #    digit_roi_img[i] = cv2.addWeighted(digit_roi_img[i], alpha_c, digit_roi_img[i], 0, gamma_c)

            ret, digit_roi_img[i] = cv2.threshold(digit_roi_img[i], thresh, 255, cv2.THRESH_BINARY)
            #digit_roi_img[i] = cv2.adaptiveThreshold(digit_roi_img[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)

            mod_frame[digit_roi[i][1]:digit_roi[i][1]+digit_roi[i][3], digit_roi[i][0]:digit_roi[i][0]+digit_roi[i][2]] = digit_roi_img[i]
        for i in range(2):
            mod_frame = cv2.rectangle(mod_frame, digit_roi[i], 63, 2)

        cv2.imshow(win_name, mod_frame)

def save(save_dir_arg=None):
    global digit_roi, digit_roi_img, annotation, min_thresh, max_thresh, img_id
    error = 0
    for i in range(2):
        if digit_roi[i][2] == 0 or digit_roi[i][3] == 0:
            error = 1
            print("ERROR: Missing bbox.")
            break
        #elif annotation[i] is None:
        #    print("WARNING: Annotation is not set.")
        #    break
    if error == 0:
        for i in range(2):
            if save_dir_arg is None:
                if annotation[i] is not None:
                    save_dir = f"dataset/{annotation[i]}"
                else:
                    save_dir = 'dataset/not_annotated'
            else:
                save_dir = save_dir_arg
            try:
                os.mkdir(save_dir)
            except:
                pass
            thresh_orig = cv2.getTrackbarPos('thresh', win_name)
            for thresh in [min_thresh, (min_thresh+max_thresh)//2, max_thresh]:
                cv2.setTrackbarPos('thresh', win_name, thresh)
                #update_frame()
                img_fn = f"{save_dir}/{i}-{img_id}-{thresh}.png"
                cv2.imwrite(img_fn, digit_roi_img[i])
                print(f"Image saved: {img_fn}")
            cv2.setTrackbarPos('thresh', win_name, thresh_orig)

def next():
    global img_id, frame, frames, rf_ret
    img_id += 1
    if len(frames) <= img_id:
        rf_ret, frame = read_frame()
        if rf_ret:
            frames.append(frame)
        else:
            img_id -= 1
    else:
        frame = frames[img_id]

def annotate():
    global annotation
    for i in range(2):
        dig = {0: 'first', 1: 'second'}[i]
        print(f"Annotate {dig} digit.")
        d = None
        while d is None:
            k = cv2.waitKey(0)
            d = {ord('0'): 0, ord('1'): 1, ord('2'): 2,  ord('3'): 3, ord('4'): 4, ord('5'): 5, ord('6'): 6, ord('7'): 7, ord('8'): 8, ord('9'): 9,}.get(k, None)
            if d is None:
                print("ERROR: Type a digit from 0 to 9.")
        annotation[i] = d
    print(f"Annotation set to {annotation[0]}.{annotation[1]}")

def trackbar_callback(x):
    update_frame()


win_name = 'Digit Recognizer'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

video_file = sys.argv[1]
video_bn = str(os.path.basename(video_file))
video = cv2.VideoCapture(video_file)
rf_ret, frame = read_frame()
frames.append(frame)

cv2.createTrackbar('thresh', win_name, 0, 255, trackbar_callback)
cv2.setTrackbarPos('thresh', win_name, 128)
#cv2.createTrackbar('blockSize', win_name, 0, 50, trackbar_callback)
#cv2.createTrackbar('C', win_name, 0, 50, trackbar_callback)
#cv2.setTrackbarPos('blockSize', win_name, 25)
#cv2.setTrackbarPos('C', win_name, 25)
#cv2.createTrackbar('contrast', win_name, 0, 255, trackbar_callback)
#cv2.setTrackbarPos('contrast', win_name, 128)

try:
    os.mkdir("dataset")
except:
    pass
for i in range(10):
    try:
        os.mkdir(f"dataset/{i}")
    except:
        pass
try:
    os.mkdir(f"dataset/not_annotated")
except:
    pass

print("Main menu. Press 'h' to display options.")
while True:
    update_frame()
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
    elif k == ord('h'):
        print("Options:")
        print("q = quit application")
        print("h = help")
        print("n = next frame")
        print("p = previous frame")
        print("c = create bounding boxes")
        print("d = delete bounding boxes")
        print("o = show bounding boxes")
        print("1 = enter fine tuning menu for bounding box 1")
        print("2 = enter fine tuning menu for bounding box 2")
        print("i = save min threshold")
        print("x = save max threshold")
        print("a = set annotation values that will be used when saving frames")
        print("s = save frame (using annotation values currently set)")
        print("m = mark start/end frame for annotation")
        print("z = annotate and save marked frames")
        print("r = save all without annotating")
        print("l = load settings")
        print("k = save settings")
    elif k == ord('m'):
        if mark == 1 and img_id < marked_ids[0]:
            print("ERROR: End frame must be higher than start frame.")
        else:
            marked_ids[mark] = img_id
            se = {0: 'start', 1: 'end'}[mark]
            print(f"Marked frame {img_id} for annotation {se}")
            mark = int(not mark)
    elif k == ord('z'):
        annotate()
        img_id = marked_ids[0]
        frame = frames[img_id]
        update_frame()
        cv2.waitKey(1)
        for i in range(marked_ids[0], marked_ids[1]):
            save()
            next()
            update_frame()
            cv2.waitKey(1)
    elif k == ord('r'):
        img_id = 0
        frame = frames[img_id]
        update_frame()
        cv2.waitKey(1)
        img_id_prev = -1
        while img_id != img_id_prev:
            img_id_prev = img_id
            save('dataset/not_annotated')
            next()
            update_frame()
            cv2.waitKey(1)
    elif k == ord('i'):
        min_thresh = cv2.getTrackbarPos('thresh', win_name)
        print(f"Min threshold set to {min_thresh}")
    elif k == ord('x'):
        max_thresh = cv2.getTrackbarPos('thresh', win_name)
        print(f"Max threshold set to {max_thresh}")
    elif k == ord('l'):
        '''termios.tcflush(sys.stdin, termios.TCIFLUSH)
        bbox_str = input("Enter bbox: ")
        try:
            bbox_json = json.loads(bbox_str)
            jsonschema.validate(bbox_json, bbox_schema)
            digit_roi = bbox_json
        except:
            print("Error: Invalid bbox.")'''
        with open('settings.json', 'r') as json_fp:
            settings = json.load(json_fp)
            if video_bn not in settings:
                print(f"WARNING: No settings found for video {video_bn}")
            else:
                jsonschema.validate(settings[video_bn], settings_schema)
                digit_roi = settings[video_bn]['digit_roi']
                min_thresh = settings[video_bn]['min_thresh']
                max_thresh = settings[video_bn]['max_thresh']
                cv2.setTrackbarPos('thresh', win_name, settings[video_bn]['thresh'])
                print(f"Loaded settings for video {video_bn} from settings.json")
    elif k == ord('k'):
        with open('settings.json', 'r') as json_fp:
            settings = json.load(json_fp)
        settings[video_bn] = {
            'digit_roi': digit_roi,
            'thresh': cv2.getTrackbarPos('thresh', win_name),
            'min_thresh': min_thresh,
            'max_thresh': max_thresh
        }
        with open('settings.json', 'w') as json_fp:
            json.dump(settings, json_fp)
        print(f"Saved settings for video {video_bn} to settings.json")
    elif k == ord('a'):
        annotate()
    elif k == ord('s'):
        save()
    elif k == ord('n'):
        next()
    elif k == ord('p'):
        if img_id > 0:
            img_id -= 1
            frame = frames[img_id]
    elif k == ord('d'):
        digit_roi = [[0, 0, 0, 0], [0, 0, 0, 0]]
    elif k == ord('c'):
        for i in range(2):
            digit_roi[i] = list(cv2.selectROI(win_name, mod_frame, True, False))
            if digit_roi[i][2] < digit_roi[i][3]:
                digit_roi[i][0] = digit_roi[i][0] - (digit_roi[i][3] - digit_roi[i][2]) // 2
                digit_roi[i][2] = digit_roi[i][3]
            else:
                digit_roi[i][1] = digit_roi[i][1] - (digit_roi[i][2] - digit_roi[i][3]) // 2
                digit_roi[i][3] = digit_roi[i][2]
        print(f"Bounding boxes created: {digit_roi}")
    elif k == ord('o'):
        print(json.dumps(digit_roi))
    elif k == ord('1') or k == ord('2'):
        i = {ord('1'): 0, ord('2'): 1}[k]
        print(f"Menu for fine tuning bounding box {i+1}. Press 'h' to display options.")
        while True:
            update_frame()
            k = cv2.waitKey(0)
            if k == ord('q'):
                print("Main menu. Press 'h' to display options.")
                break
            elif k == ord('h'):
                print("Options:")
                print("q = quit this menu, return to previous menu")
                print("h = help")
                print("w/a/s/d = move bounding box")
                print("r = increase bounding box size")
                print("f = decrease bounding box size")
                print("o = show bounding boxes")
                continue
            elif k == ord('o'):
                print(json.dumps(digit_roi))
            elif k == ord('a'):
                if digit_roi[i][0] > 0:
                    digit_roi[i][0] -= 1
            elif k == ord('d'):
                if digit_roi[i][0] + digit_roi[i][2] < frame.shape[1]-1:
                    digit_roi[i][0] += 1
            elif k == ord('w'):
                if digit_roi[i][1] > 0:
                    digit_roi[i][1] -= 1
            elif k == ord('s'):
                if digit_roi[i][1] + digit_roi[i][3] < frame.shape[0]-1:
                    digit_roi[i][1] += 1
            elif k == ord('r'):
                if digit_roi[i][0] + digit_roi[i][2] < frame.shape[1]-1 and digit_roi[i][1] + digit_roi[i][3] < frame.shape[0]-1:
                    digit_roi[i][2] += 1
                    digit_roi[i][3] += 1
            elif k == ord('f'):
                if digit_roi[i][2] > 0 and digit_roi[i][3] > 0:
                    digit_roi[i][2] -= 1
                    digit_roi[i][3] -= 1

cv2.destroyWindow(win_name)
cv2.destroyAllWindows()
