import torch
import torch.nn as nn
from typing import List
import cv2
from .utils import batch_padding

from modeling.backbone import build_backbone
from modeling.layers import *

__all__ = ["CenterNet"]
class CenterNet(nn.Module):
    """
    Implement CenterNet (https://arxiv.org/abs/1904.07850).
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg

        # fmt: off
        self.num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        # fmt: on
        self.backbone = build_backbone(cfg)
        self.upsample = CenternetDeconv(cfg)
        self.head = CenternetHead(cfg)

        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    @torch.no_grad()
    def _forward(self, batch_images):

        features = self.backbone(batch_images)
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)

        return pred_dict

    @torch.no_grad()
    def inference_on_images(self, images: List, K=100, max_size=512):

        batch_images,params = self._preprocess(images, max_size)
        pred_dict = self._forward(batch_images)

        heat, wh, reg = pred_dict['cls'], pred_dict['wh'], pred_dict['reg']
        batch, cat, height, width = heat.size()
        bboxes,scores,clses = CenterNetDecoder.decode(heat, wh , reg, K=K)
        clses = clses.view(batch, K)
        scores = scores.view(batch, K)
        results = []
        for i,param in zip(range(batch),params):
            scale_x, scale_y = param['width'] / float(param['resized_width']), \
                               param['height'] / float(param['resized_height'])
            bboxes[i, :, 0::2] = bboxes[i, :, 0::2] * scale_x * self.cfg.MODEL.CENTERNET.DOWN_SCALE
            bboxes[i, :, 1::2] = bboxes[i, :, 1::2] * scale_y * self.cfg.MODEL.CENTERNET.DOWN_SCALE

            # import cv2
            # image = images[i]
            # for j,bbox in enumerate(bboxes[i]):
            #     if scores[i][j] > 0.1:
            #         cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
            # cv2.imwrite('result.jpg',image)
            # import pdb; pdb.set_trace()

            # result.pred_boxes = Boxes(bboxes[i])
            # result.scores = scores[i]
            # result.pred_classes = clses[i]
            # results.append({"instances": result})
            result = {'cls': clses[i],
                      'bbox': bboxes[i],
                      'scores': scores[i]}
            results.append(result)
        return results

    def _preprocess(self, images: List, max_size=512):
        """
        Normalize, pad and batch the input images.
        """
        batch_images = []
        params = []
        for image in images:
            old_size = image.shape[0:2]
            ratio = min(float(max_size) / (old_size[i]) for i in range(len(old_size)))
            new_size = tuple([int(i * ratio) for i in old_size])
            resize_image = cv2.resize(image, (new_size[1], new_size[0]))
            params.append({'width': old_size[1],
                           'height': old_size[0],
                           'resized_width': new_size[1],
                           'resized_height':new_size[0]
                           })
            batch_images.append(resize_image)
        batch_images = [torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) \
                        for img in batch_images]
        batch_images = [img.to(self.device) for img in batch_images]
        batch_images = [self.normalizer(img/255.) for img in batch_images]
        batch_images = batch_padding(batch_images, 32)
        return batch_images, params


def build_model(cfg):

    model = CenterNet(cfg)
    return model
