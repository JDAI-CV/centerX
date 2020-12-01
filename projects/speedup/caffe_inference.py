import os
import cv2
import numpy as np
import caffe
from tqdm import tqdm
import time

caffe.set_mode_gpu()

class centerX_caffe():
    def __init__(self, model_path, prt_path):
        self.net = caffe.Net(prt_path, model_path, caffe.TEST)
        self.net.blobs['blob1'].reshape(1, 3, 384, 640)
        self.mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
        self.std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])

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
        images = [(image[:, :, ::-1] - self.mean) / self.std for image in images]  # to rgb
        pad_images, ratio = self.resize_and_pad(images, new_size, stride)
        return pad_images, ratio

    def postprocess(self, result, ratios, thresh=0.3):
        print(result.keys())
        clses, max_pool,  = result['relu_blob77'], result['max_pool_blob1']
        regs, whs = result['conv_blob80'], result['conv_blob78']
        keep = (clses == max_pool).astype(np.float)
        clses = clses*keep
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

    def inferencen_on_images(self, images, new_size=(640, 384), thresh=0.2):
        # images: [ndarray(H,W,C),...] in bgr Mode
        # only support batch = 1, rely on centerX2onnx input , line: 142
        # you can modify your batch in centerX2onnx and generate new model
        assert len(images) == 1

        t0 = time.time()
        pad_images, ratios = self.preprocess(images, new_size)
        pad_images = pad_images.transpose(0, 3, 1, 2)
        t1 = time.time()
        print('preprocess cost', t1-t0, 's')

        self.net.blobs['blob1'].data[...] = pad_images
        result = self.net.forward()

        t2 = time.time()
        print('forward cost', t2-t1, 's')
        bboxes = self.postprocess(result, ratios, thresh)
        t3 = time.time()
        print('postprocess cost',t3-t2,'s')
        return bboxes


if __name__ == '__main__':
    ort_sess = centerX_caffe('models/person_face_regnetX400MF_sgd.caffemodel',
                             'models/person_face_regnetX400MF_sgd.prototxt')
    img = cv2.imread('datasets/crowd_human/JPEGImages/273271,1278b0001499303e.jpg')
    for i in tqdm(range(1)):
        bbox = ort_sess.inferencen_on_images([img])[0]
        for (cls,_,x1,y1,x2,y2) in bbox:
            print(cls)
            color = (0,255,0) if cls == 0 else (0,255,255)
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),color,4)
        cv2.imwrite('result.jpg', img)