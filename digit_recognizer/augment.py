import cv2
import glob
import os
import numpy as np

modifications = {
    'rotate': 15,
    'scale': 0.175,
    'translate': 0.075
}
new_size = (28, 28)

print("Removing old images...")
rm_old = set(glob.glob(f"dataset/*/*rotate*.png") + 
             glob.glob(f"dataset/*/*scale*.png") + 
             glob.glob(f"dataset/*/*translate*.png") +
             glob.glob(f"dataset/*/*-{new_size[0]}x{new_size[1]}.png"))
count = 0
for fn in rm_old:
    count += 1
    if count % 10 == 0:
        print(f"Processing {count}/{len(rm_old)}", end='\r')
    os.remove(fn)
print("")

def modify_save(img, mats, num_cols, num_rows, new_size, mod):
    for i in range(len(mats)):
        img2 = cv2.warpAffine(img, mats[i], (num_cols, num_rows), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        cv2.imwrite(os.path.join(dn, f"{bn}-{mod}{i}.png"), img2)
        #img3 = cv2.resize(img2, new_size, interpolation=cv2.INTER_CUBIC)
        #cv2.imwrite(os.path.join(dn, f"{bn}-{mod}{i}-{new_size[0]}x{new_size[1]}.png"), img3)

print("Augmenting and resizing images...")
images = glob.glob(f"dataset/*/*.png")
images.sort()
count = 0
for fn in images:
    count += 1
    if count % 10 == 0:
        print(f"Processing {count}/{len(images)}", end='\r')
    bn = os.path.splitext(os.path.basename(fn))[0]
    dn = os.path.dirname(fn)
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    num_rows, num_cols = img.shape[:2]
    center = (num_cols//2, num_rows//2)
    for k, v in modifications.items():
        if k == 'rotate':
            angle = v
            scale = 1
            mats = [
                cv2.getRotationMatrix2D(center, angle, scale),
                cv2.getRotationMatrix2D(center, angle*-1, scale)
            ]
        elif k == 'scale':
            angle = 0
            scale = v
            mats = [
                cv2.getRotationMatrix2D(center, angle, 1+scale),
                cv2.getRotationMatrix2D(center, angle, 1-scale)
            ]
        elif k == 'translate':
            trans_x = num_cols * v
            trans_y = num_rows * v
            mats = [
                np.float32([[1,0,trans_x], [0,1,trans_y]]),
                np.float32([[1,0,trans_x], [0,1,-trans_y]]),
                np.float32([[1,0,-trans_x], [0,1,trans_y]]),
                np.float32([[1,0,-trans_x], [0,1,-trans_y]])
            ]
        modify_save(img, mats, num_cols, num_rows, new_size, k)

    #img2 = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite(os.path.join(dn, f"{bn}-{new_size[0]}x{new_size[1]}.png"), img2)
print("")
