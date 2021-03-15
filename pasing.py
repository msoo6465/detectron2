import os
import json
import cv2
from tqdm import tqdm
from pprint import pprint


def devide_dataset(save_path='./new_dataset',base_path='C:/dataset/iitp',show=True,save=True):
    img_path = os.path.join(save_path,'images')
    os.makedirs(img_path,exist_ok=True)
    json_path = f'{base_path}/190709_iitp_annotation_merge_vis_coco_nexys_filtered.json'
    base_train_path = os.path.join(base_path,'train')
    save_json = os.path.join(save_path, 'new_data.json')

    if os.path.isfile(save_json):
        is_json = True

    with open(json_path,encoding='utf8') as json_file:
        json_data = json.load(json_file)
        for how,data in enumerate(json_data):
            if how%5000==0:
                print('='*(how//5000)+f'  ({how} / {len(json_data)})')
            filename = data['filename']

            if save:
                save_filename = filename.split('/')[-1]
            data['filename'] = os.path.join(img_path,filename.split('/')[-1])


            if 'COCO' in filename:
                continue

            anns = data['ann']
            filename = os.path.join(base_train_path,filename)
            img = cv2.imread(filename)

            if not 1 in anns['labels'] and not 4 in anns['labels']:
                print(not 1 in anns['labels'])
                print(not 4 in anns['labels'])
                continue
            for idx in range(len(anns['bboxes'])):
                labels = anns['labels'][idx]
                bbox = anns['bboxes'][idx]
                if show:
                    im = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 0), 2)
                    im = cv2.putText(im, text=f'[{idx}] {labels}',
                                     org=(int(bbox[0]), int(bbox[1])),
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)

            cv2.imwrite(os.path.join(img_path,save_filename),img)

            if save:
                if is_json:
                    with open(save_json, encoding='utf8') as exist_json:
                        exist_json_data = json.load(exist_json)
                    exist_json_data.append(data)

                    with open(save_json, "w") as s_json_file:
                        json.dump(exist_json_data, s_json_file)
                else:
                    with open(save_json, "w") as s_json_file:
                        json.dump(data, s_json_file)

            if show:
                cv2.imshow('1',im)
                cv2.waitKey(0)
        cv2.destroyAllWindows()

devide_dataset(show=False)