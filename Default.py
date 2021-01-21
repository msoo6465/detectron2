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

from detectron2.structures import BoxMode
from pprint import pprint

import pandas as pd
import ast
import csv

import tqdm
import time

lsf = {'ff','ddd'}


def get_wheat_dicts(img_dir):
    csv_file = os.path.join(img_dir,'train.csv')
    imgs_anns = pd.read_csv(csv_file)
    imgs_anns['bbox'] = imgs_anns['bbox'].apply(lambda x: ast.literal_eval(x))
    dataset_dict = []

    grouped = imgs_anns['bbox'].groupby(imgs_anns['image_id']).sum()
    for i, (k, v) in enumerate(grouped.items()):
        filename = os.path.join(img_dir, k + '.jpg')

        bboxes = [v[i:i + 4] for i in range(0, len(v), 4)]
        objs = []
        for bbox in bboxes:
            x, y, w, h = bbox
            obj = {
                    "bbox": [x, y, x+w, y+h],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                }
            objs.append(obj)

        record = {}
        record['file_name'] = filename
        record['image_id'] = i
        record['height'] = 1024
        record['width'] = 1024
        record['annotations'] = objs
        dataset_dict.append(record)

    return dataset_dict


for d in ['train']:
    DatasetCatalog.register("wheat_"+ d,lambda d=d:get_wheat_dicts("./data/global-wheat-detection/"+d))
    MetadataCatalog.get("wheat_"+d).set(thing_classes=["wheat"])
wheat_metadata = MetadataCatalog.get("wheat_train")

# dataset_dicts = get_wheat_dicts("./data/global-wheat-detection/train")
#
# for d in random.sample(dataset_dicts,3):
#     img = cv2.imread(d['file_name'])
#     visualizer = Visualizer(img[:,:,::-1],metadata=wheat_metadata,scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow('show',out.get_image()[:,:,::-1])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("wheat_train")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)



test_dir = './data/global-wheat-detection/test'
img_files = os.listdir(test_dir)

from detectron2.utils.visualizer import ColorMode

img_pathes = [os.path.join(test_dir,img_file) for img_file in img_files]

final_list = []
for d in img_pathes:
    im = cv2.imread(d)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=wheat_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # print(d)
    # print(outputs['instances'].to('cpu').pred_boxes)
    for bbox in outputs['instances'].to('cpu').pred_boxes:
        final_dict = {}
        tl = bbox[:2]
        br = bbox[2:]
        # cv2.rectangle(im,tuple(tl),tuple(br),(255,0,0),3)
        final_dict['image_id'] = d.split('/')[-1]
        final_dict['PredictionString'] = np.array(bbox).astype('int')
        final_list.append(final_dict)

    cv2.imshow('final',out.get_image()[:, :, ::-1])
    # cv2.imshow('so',im)
    cv2.waitKey(0)
final = pd.DataFrame(final_list)
print(final)
final.to_csv('submission.csv',index=False)
post = pd.read_csv('submission.csv')
print(post)
post['PredictionString'] = post['PredictionString'].apply(lambda x : x[1:-1])
print(post)
post.to_csv('submission2.csv')