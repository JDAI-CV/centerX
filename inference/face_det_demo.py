import cv2

import sys, os
from tqdm import tqdm

sys.path.insert(0, '.')
from configs import add_centernet_config
from detectron2.config import get_cfg
from inference.centernet import build_model
from detectron2.checkpoint import DetectionCheckpointer

if __name__ == "__main__":
    # cfg
    cfg = get_cfg()
    add_centernet_config(cfg)
    #cfg.merge_from_file("yamls/person_face/person_face_res18_multi_KD.yaml")
    cfg.merge_from_file("yamls/person_face/person_face_regnetx_400MF_multi_KD.yaml")

    # model
    model = build_model(cfg)
    DetectionCheckpointer(model).load("exp_results/person_face/person_face_regnetX_multiKD_sgd/model_final.pth")
    model.eval()

    #txt
    val_result_txt_save_root = '../Ultra-Light-Fast-Generic-Face-Detector-1MB/widerface_evaluate/centerx_pred/'
    val_image_root = '/home/chengpeng39/notespace/centerX/centerx/datasets/wider_face_add_lm_10_10_add_mafa/WIDER_val/'
    images = []
    for parent, dir_names, file_names in os.walk(val_image_root):
        for file_name in file_names:
            if not file_name.lower().endswith('jpg'):
                continue
            images.append(os.path.join(parent, file_name))
    # images
    bs = 8
    for i in tqdm(range(0, len(images), 8)):
        images_rgb = [cv2.imread(j)[:,:,::-1] for j in images[i:i + 8]]
        img_names = [j for j in images[i:i + 8]]
        results = model.inference_on_images(images_rgb, K=500, max_size=640)
        for k,result in enumerate(results):
            cls = result['cls'].cpu().numpy()
            bbox = result['bbox'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            H,W,C = images_rgb[k].shape

            # make dir and txt
            img_name = img_names[k]
            txt_name = os.path.basename(img_name).split('.')[0] + '.txt'
            event_name = img_name.split('/')[-2]
            if not os.path.exists(os.path.join(val_result_txt_save_root, event_name)):
                os.makedirs(os.path.join(val_result_txt_save_root, event_name))
            fout = open(os.path.join(val_result_txt_save_root, event_name, txt_name), 'w')
            fout.write(txt_name.split('.')[0] + '\n')
            fout.write(str(int(sum(cls))) + '\n')
            for c,(x1,y1,x2,y2),s in zip(cls,bbox,scores):
                if c != 1.0:
                    continue
                x1 = str(max(0, int(x1)))
                y1 = str(max(0, int(y1)))
                w = str(min(W, int(x2)) - int(x1))
                h = str(min(H, int(y2)) - int(y1))
                s = str(round(float(s),3))
                line = ' '.join([x1,y1,w,h,s])
                fout.write(line+'\n')
            fout.close()