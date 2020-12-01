import cv2

import sys, os
import json
from tqdm import tqdm

sys.path.insert(0, '.')
from configs import add_centernet_config
from detectron2.config import get_cfg
from inference.centernet import build_model
from detectron2.checkpoint import DetectionCheckpointer

def get_widerface_images():
    lines = open('datasets/wider_face_add_lm_10_10_add_mafa/ImageSets/Main/trainval_all.txt').readlines()
    root = 'datasets/wider_face_add_lm_10_10_add_mafa/JPEGImages/'
    images = [root + i.strip() + '.jpg' for i in lines]
    return images

def get_crowdhuman_images():
    json_file = 'datasets/crowd_human/Annotations/annotation_train.odgt'
    lines = open(json_file).readlines()
    root = 'datasets/crowd_human/JPEGImages/'
    images = [root + json.loads(line.strip('\n'))['ID']+ '.jpg' for line in lines]
    return images


if __name__ == "__main__":
    # cfg
    cfg = get_cfg()
    add_centernet_config(cfg)
    #cfg.merge_from_file("yamls/coco_det/centernet_r_50_C4_0.5x_coco_person.yaml")
    cfg.merge_from_file("yamls/person_face/face_res50.yaml")

    # model
    model = build_model(cfg)
    #DetectionCheckpointer(model).load("models/coco_det_crowd_R50_SGD.pth")
    DetectionCheckpointer(model).load("models/person_face_R50_face_adam.pth")
    model.eval()

    #txt
    #txt = open('datasets/wider_face_add_lm_10_10_add_mafa/widerface_pseudo_human.txt','w')
    txt = open('datasets/crowd_human/crowdhuman_pseudo_face.txt','w')
    class_name = 'face'

    # images
    images = get_crowdhuman_images()
    bs = 8
    for i in tqdm(range(0, len(images), bs)):
        images_rgb = [cv2.imread(j)[:,:,::-1] for j in images[i:i + bs]]
        img_names = [os.path.basename(j) for j in images[i:i + bs]]
        results = model.inference_on_images(images_rgb, K=100, max_size=640)
        for k,result in enumerate(results):
            cls = result['cls'].cpu().numpy()
            bbox = result['bbox'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            H,W,C = images_rgb[k].shape
            img = images_rgb[k][:,:,::-1]
            img_name = img_names[k]
            line = ';'.join([img_name,str(W),str(H)])
            for c,(x1,y1,x2,y2),s in zip(cls,bbox,scores):
                if c != 0.0 or s < 0.3:
                    continue
                x1 = str(max(0, int(x1)))
                y1 = str(max(0, int(y1)))
                x2 = str(min(W, int(x2)))
                y2 = str(min(H, int(y2)))
                s = str(round(float(s),3))
                line += ';'.join(['',class_name,x1,y1,x2,y2])
            txt.write(line+'\n')