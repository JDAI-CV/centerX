import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import time


class centerX_pt():
    def __init__(self, model_path, device="cuda:0"):
        self.model = torch.jit.load(model_path)
        self.model.to(device).eval()
        self.device = device

    def resize_and_pad(self, images, new_shape=(640, 384), stride=32):
        # images: a list of image,[[H,W,C],...,[H,W,C]]
        shape = np.array([image.shape[:2][::-1] for image in images])  # width, height
        if isinstance(new_shape, int):
            ratio = float(new_shape) / np.max(shape, axis=1, keepdims=True)
        else:
            new_shape = np.array(new_shape)
            ratio = np.min(new_shape / shape, axis=1, keepdims=True)  # ratio  = new / old

        new_unpads = shape * ratio
        new_unpads = new_unpads.astype(np.int)
        imgs = np.zeros(tuple([len(images)]) + tuple(new_shape[::-1]) + (3,))

        for k, image, new_unpad in zip(range(len(images)), images, new_unpads):
            new_unpad = tuple(new_unpad)
            w, h = new_unpad
            img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_AREA)  # resized, no border
            imgs[k, :h, :w, :] = img
        imgs = imgs.astype(np.float32)
        return imgs, ratio

    def preprocess(self, images, new_size=(640, 384), stride=32):
        # images: [ndarray(H,W,C),...] in bgr Mode
        images = [image[:, :, ::-1] for image in images]  # to rgb
        pad_images, ratio = self.resize_and_pad(images, new_size, stride)
        return pad_images, ratio

    def postprocess(self, result, ratios, thresh=0.3):
        clses, regs, whs = result
        clses, regs, whs  = clses.cpu().numpy(), regs.cpu().numpy(), whs.cpu().numpy()
        # clses: (b,c,h,w)
        # regs:  (b,2,h,w)
        bboxes = []

        for cls, reg, wh, ratio in zip(clses, regs, whs, ratios):
            index = np.where(cls >= thresh)
            ratio = 4 / ratio
            score = np.array(cls[index])
            cat = np.array(index[0])
            ctx, cty = index[-1], index[-2]
            w, h = wh[0, cty, ctx], wh[1, cty, ctx]
            off_x, off_y = reg[0, cty, ctx], reg[1, cty, ctx]
            ctx = np.array(ctx) + np.array(off_x)
            cty = np.array(cty) + np.array(off_y)
            x1, x2 = ctx - np.array(w) / 2, ctx + np.array(w) / 2
            y1, y2 = cty - np.array(h) / 2, cty + np.array(h) / 2
            x1, y1, x2, y2 = x1 * ratio, y1 * ratio, x2 * ratio, y2 * ratio
            bbox = np.stack((cat, score, x1, y1, x2, y2), axis=1).tolist()
            bbox = sorted(bbox, key=lambda x: x[1], reverse=True)
            bboxes.append(bbox)

        return bboxes

    def inferencen_on_images(self, images, new_size=(640, 384), thresh=0.30):
        # images: [ndarray(H,W,C),...] in bgr Mode

        torch.cuda.synchronize()
        t0 = time.time()
        pad_images, ratios = self.preprocess(images, new_size)
        pad_images = pad_images.transpose(0, 3, 1, 2)
        pad_images = torch.from_numpy(pad_images).to(self.device)
        torch.cuda.synchronize()
        t1 = time.time()
        print('preprocess cost', t1-t0, 's')

        with torch.no_grad():
            result = self.model(pad_images)
        torch.cuda.synchronize()
        t2 = time.time()
        print('forward cost', t2-t1, 's')
        bboxes = self.postprocess(result, ratios, thresh)
        torch.cuda.synchronize()
        t3 = time.time()
        print('postprocess cost',t3-t2,'s')
        return bboxes


if __name__ == '__main__':
    net = centerX_pt('models/person_face_regnetX400MF_sgd.pt')
    root = '/export/167/centerX/to_other/bjz_test/'
    save_path = '/export/167/centerX/results/'
    images_path = os.listdir('/export/167/centerX/to_other/bjz_test/')
    images = [cv2.imread(root+img) for img in images_path]
    print('------------')
    bboxes = net.inferencen_on_images(images, thresh=0.25)
    for img,img_bgr,bbox in zip(images_path, images, bboxes):
        for (cat, score, x1, y1, x2, y2) in bbox:
            # if cat != 1:
            #     continue
            color = (255,0,0)
            if cat == 1:
                color = (0,255,0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img_bgr,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
            cv2.putText(img_bgr, str(round(score,3)), (int(x1), int(y1)), font, 2, (0, 0, 255), 2)
        cv2.imwrite(save_path+img, img_bgr)
    # for img in tqdm(images):
    #     img_bgr = cv2.imread(root + img)
    #     bbox = net.inferencen_on_images([img_bgr])[0]
    #     for (_,_,x1,y1,x2,y2) in bbox:
    #        cv2.rectangle(img_bgr,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
    #     cv2.imwrite(save_path+img, img_bgr)