import json
import os
import cv2



img_dir = './data/dataset/car_train'
json_file = os.path.join(img_dir, "car_train.json")
# if not os.path.isfile(json_file):
# json_file = os.path.join(img_dir, "car_val.json")

with open(json_file,encoding='UTF8') as f:
    imgs_anns = json.load(f)

for idx, v in enumerate(imgs_anns):
    filename = os.path.join("/".join(json_file.split('/')[:-1]), v["filename"])
    im = cv2.imread(filename)
    if not (int(v['height'])==im.shape[0] and int(v['width'])==im.shape[1]):
        print(filename)

        print(im.shape[0],im.shape[1])
        print(v['height'],v['width'])

if not(1 and 0):
    print(1)
if not (0 and 0):
    print(2)
if not (1 and 1):
    print(3)
