import os
import json
import cv2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from pprint import pprint

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
from detectron2.engine import default_argument_parser, launch

import os
import cv2
import json
from detectron2.structures import BoxMode

from pprint import pprint
target_classes = [
    "Vehicle",
    "LicensePlate",
    "Person"
]

img_dir = './data/dataset/car_train'
json_file = os.path.join(img_dir, "c_car_train.json")

if not os.path.isfile(json_file):
    json_file = os.path.join(img_dir, "car_val.json")


def get_car_dicts(img_dir):
    font = cv2.FONT_HERSHEY_SIMPLEX
    json_file = os.path.join(img_dir, "c_car_train.json")

    if not os.path.isfile(json_file):
        json_file = os.path.join(img_dir, "car_val.json")

    with open(json_file, encoding='UTF8') as f:
        imgs_anns = json.load(f)

        dataset_dicts = []
        for idx, v in enumerate(imgs_anns):
            record = {}

            filename = os.path.join("/".join(json_file.split('/')[:-1]), v["filename"])
            height, width = v['height'], v['width']
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = int(height)
            record["width"] = int(width)
            # temp = cv2.imread(filename)
            objs = []
            annos = v["ann"]
            annos['labels']=[1 if i > 0 and i < 6 else 2 if i == 6 else 3 if i == 7 else 1 if i == 8 else None for i in annos['labels']]
            # temp = cv2.imread(filename)
            for i, bbox in enumerate(annos['bboxes']):
                obj = {
                    "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": annos['labels'][i] - 1,
                }
                objs.append(obj)
                # temp = cv2.rectangle(temp, tuple(obj['bbox'][:2]), tuple(obj['bbox'][2:4]), (255, 0, 0), thickness=3)
                # temp = cv2.putText(temp, str(obj['category_id']) + target_classes[obj['category_id']], tuple(obj['bbox'][:2]), font, 2, (255, 255, 255), 2)
        #     temp = cv2.resize(temp, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        #     cv2.imshow('123', temp)
        #     cv2.waitKey(0)
        #
            record["annotations"] = objs
            dataset_dicts.append(record)
        # cv2.destroyAllWindows()

        return dataset_dicts


get_car_dicts('./data/dataset/car_train')
for d in ['train','val']:
    DatasetCatalog.register("car_"+ d,lambda d=d:get_car_dicts('./data/dataset/car_'+d))
    MetadataCatalog.get("car_"+d).set(thing_classes=target_classes)
car_metadata = MetadataCatalog.get("car_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("car_train")
cfg.DATASETS.TEST = ("car_val")
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.025
cfg.SOLVER.MAX_ITER = 40000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.OUTPUT_DIR = 'R_50_FPN_1x_result'
cfg.SOLVER.REFERENCE_WORLD_SIZE = 2
cfg.SOLVER.STEPS = (25000, 30000)
cfg.SOLVER.CHECKPOINT_PERIOD = 3000
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0073999.pth")  # path to the model we just trained

def main(train):
    print(cfg)
    exit()
    if train:
        from detectron2.engine import DefaultTrainer

        os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=True)
        try:
            trainer.train()
        except Exception as e:
            print('Error : ',e)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold

    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode

    dataset_dicts = get_car_dicts("./data/dataset/car_val")
    for d in random.sample(dataset_dicts, 10):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=car_metadata,
                       scale=0.7,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('hello',out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    evaluator = COCOEvaluator("car_val", None, True, output_dir="./output/car_03_10/")
    val_loader = build_detection_test_loader(cfg, "car_val")
    print('',val_loader)
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`
    return



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        1,
        num_machines=1,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(True,),
    )