import cv2
from train import get_car_dicts
import random
import os


from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

CLASS_NAMES = (
    'Vehicle',
    'LicensePlate',
    'Person'
     )
OUTPUT_PATH = './before_blur'


args = default_argument_parser().parse_args()
args.config_file = 'COCO-Detection/car_faster_rcnn_R_50_FPN_1x.yaml'
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
cfg.merge_from_list(args.opts)
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
cfg.freeze()
default_setup(
    cfg, args
)
print(cfg)
sample_path = "./data/blur_sample"

dataset_list = os.listdir(sample_path)
print(dataset_list)
dataset_list = [os.path.join(sample_path,i) for i in dataset_list]
predictor = DefaultPredictor(cfg)
MetadataCatalog.get("blur_sample").set(thing_classes=CLASS_NAMES)
for d in dataset_list:
    im = cv2.imread(d)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    for i in range(len(outputs['instances'])):
        label = int(outputs['instances'][i].get('pred_classes'))
        score = round(float(outputs['instances'][i].get('scores')),4)
        # if score <= 0.5:
        #     continue
        x1, y1, x2, y2 = outputs['instances'][i].get('pred_boxes').tensor[0]
        a_x1, a_y1, a_x2, a_y2 = int(x1), int(y1), int(x2), int(y2)
        a_area = abs(a_x1-a_x2)*abs(a_y1-a_y2)
        print(f'{i}번째 이후로 몇개가 더 있을까?',len(outputs['instances'][i+1:]))
        for idx in range(len(outputs['instances'][i+1:])):
            b_x1,b_y1,b_x2,b_y2 = outputs['instances'][i+idx+1].get('pred_boxes').tensor[0]
            b_x1, b_y1, b_x2, b_y2 = int(b_x1), int(b_y1), int(b_x2), int(b_y2)
            b_area = abs(b_x1-b_x2)*abs(b_y1-b_y2)


            intersaction_x = min(a_x2, b_x2) - max(a_x1, b_x1)
            print(f'intersaction_x : {min(a_x2, b_x2)} - {max(a_x1, b_x1)}={intersaction_x} ')
            intersaction_y = min(a_y2, b_y2) - max(a_y1, b_y1)
            print(f'intersaction_y : {min(a_y2, b_y2)} - {max(a_y1, b_y1)}={intersaction_y} ')

            intersaction_area = intersaction_y * intersaction_x

            union_area = a_area + b_area - intersaction_area

            iou = intersaction_area/union_area

            print(f"Onion area : {union_area}, intersaction area : {intersaction_area}")
            print(f'{i} and {i+idx+1} iou : ',iou)


        # print('a box :',int_x1,int_y1,'b box',(int_x2,int_y2))
        if label == 2:
            im = cv2.rectangle(im,(a_x1, a_y1),(a_x2, a_y2),(0,0,0),2)
        else:
            im = cv2.rectangle(im,(a_x1, a_y1),(a_x2, a_y2),(0,0,0), 2)

    for i in range(len(outputs['instances'])):
        label = int(outputs['instances'][i].get('pred_classes'))
        score = round(float(outputs['instances'][i].get('scores')), 4)
        # if score <= 0.7:
        #     continue
        x1, y1, x2, y2 = outputs['instances'][i].get('pred_boxes').tensor[0]
        int_x1, int_y1, int_x2, int_y2 = int(x1), int(y1), int(x2), int(y2)
        im = cv2.putText(im, text=f'[{i}] {CLASS_NAMES[label]} {(int_x1,int_x2,int_y1,int_y2)}', org=(int_x1, int_y1),
                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2)
        # if label == 2:
        #     im = cv2.putText(im, text=f'[{i}]'+CLASS_NAMES[label] + ' : ' + str(score)+"point : "+str(int_x1)+" , " +str(int_y1), org=(int_x1, int_y1),
        #                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2)
        # else:
        #     im = cv2.putText(im, text=f'[{i}]'+CLASS_NAMES[label] + ' : ' + str(score), org=(int_x1, int_y1),
        #                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2)

    cv2.imshow('hello', im)
    # cv2.imwrite(f'{OUTPUT_PATH}/blur_{d.split("/")[-1]}',im)
    cv2.waitKey(0)

cv2.destroyAllWindows()