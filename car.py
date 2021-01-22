import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some commom library
import numpy as np
import os, json, cv2, random
import torch, torchvision
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,DatasetCatalog

import os
import cv2
import json
from detectron2.structures import BoxMode


train_dir = './data/cardata/dataset/car_train'

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
def get_car_dicts(img_dir):
    json_file = os.path.join(img_dir, "car_train.json")
    with open(json_file,encoding='UTF8') as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        record = {}

        filename = os.path.join(train_dir, v["filename"])
        height, width = v['height'], v['width']

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["ann"]
        objs = []
        for i, bbox in enumerate(annos['bboxes']):
            obj = {
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": annos['labels'][i]-1,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


for d in ['train','val']:
    DatasetCatalog.register("car_"+ d,lambda d=d:get_car_dicts('./data/cardata/dataset/car_'+d))
    MetadataCatalog.get("car_"+d).set(thing_classes=target_classes)
wheat_metadata = MetadataCatalog.get("car_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("car_train")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
cfg.OUTPUT_DIR = 'CAR_R_101_FPN'

from detectron2.engine import DefaultTrainer

os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()