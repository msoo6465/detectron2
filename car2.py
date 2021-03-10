import json
import os
import pprint

from tqdm import tqdm

import random
from detectron2.config import get_cfg

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer


img_root = "./dataset"
json_root = "car_train.json"
json_root_val = "car_val.json"
target_classes = [
    "Car",
    "SUV",
    "Van",
    "Truck",
    "SpecialCar",
    "LicensePlate",
    "Person",
    "Motorcycle"
]
for img in sorted(os.listdir(img_root)):
    imgae = os.path.join(img_root, img)
    print("image ->2 " , imgae)
    # ./dataset/train
        # print("imgs_dir -> " , imgae)
    for img2 in sorted(os.listdir(imgae)):
        imgae2 = os.path.join(imgae, img2)
            # print("imgs -> " , imgae2)


# cfg = get_cfg()
# cfg.merge_from_file("./detectron2/model_zoo/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml")
# cfg.DATASETS.TRAIN = (img_root + "/car_train",)
# cfg.DATALOADER.NUM_WORKERS = 2 # 쓰레드 cpu 자원 할당
# cfg.SOLVER.IMS_PER_BATCH = 2 # 크면 클수록 좋음
# cfg.SOLVER.BASE_LR = 0.00025 # 수정
# cfg.SOLVER.MAX_ITER = 50000 # 수정
# cfg.MODEL.RETINANET.NUM_CLASSES = 8
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()
#
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.6
# cfg.DATASETS.TEST = (img_root + "/car_val",)



def get_car_dicts(img_root, json_root):

    json_file = os.path.join(img_root, json_root)
    print("img_root -> " , img_root)
    print("jos" , json_file) # ./dataset/train/train_car_classify.json

    # json_file open
    with open(json_file) as f:
        image_anns = json.load(f)

    dataset_dicts = []
    # tqdm -> 프로세스 진행 상황
    for idx , v in tqdm(enumerate(image_anns)):
        recode = {}

        # get image metadata
        filename = os.path.join(img_root, v["filename"])

        # h , w = cv2.imread(filename).shape[:2]
        recode["file_name"] = filename
        recode["width"] = v["width"]
        recode["height"] = v["height"]
        annos = v["ann"]
        bbox = annos["bboxes"]
        objs = []

        num_bbox = len(bbox)

        for i in range(num_bbox):

            bbox = annos["bboxes"]
            x1 = bbox[i][0]
            y1 = bbox[i][1]
            x2 = bbox[i][2]
            y2 = bbox[i][3]

            lables = annos["labels"]
            lables_number = lables[i]
            # 4number ->
            """
                # BoxMode -> xyxy_ABS
                (x0, y0, x1, y1) in absolute floating points coordinates.
                The coordinates in range [0, width or height].
            """
            # one-stage background not / two-stage background 0 (def)
            # json 1 ~ 8 (two-stage 기반 이라서 .. 0 = background)
            obj = {
                "bbox" : [x1, y1 , x2 , y2],
                "bbox_mode" : BoxMode.XYXY_ABS,
                "category_id" : lables_number-1
            }
            objs.append(obj)

        recode["annotations"] = objs
        # print("recode" , recode)
        dataset_dicts.append(recode)

    print("Showing an example: ")
    # print(dataset_dicts)
    pprint.pprint(random.sample(dataset_dicts, 1))
    return dataset_dicts

