import math

import numpy as np
import torch
import torch.nn as nn

# from centernet.network.backbone import Backbone
from ..backbone import build_backbone
from ..losses import reg_l1_loss, modified_focal_loss, KdLoss, ignore_unlabel_focal_loss, mse_loss
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
from ..layers import *


__all__ = ["CenterNet"]


@META_ARCH_REGISTRY.register()
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
        # Loss parameters:
        # Inference parameters:
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on
        self.backbone = build_backbone(cfg)
        self.upsample = CenternetDeconv(cfg)
        self.head = CenternetHead(cfg)

        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        if cfg.MODEL.CENTERNET.KD.ENABLED:
            self.kd_loss = KdLoss(cfg)
            self.kd_without_label = cfg.MODEL.CENTERNET.KD.KD_WITHOUT_LABEL

        self.to(self.device)

    def forward(self, batched_inputs, model_ts=None):
        """
        Args:
            batched_inputs(list): batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
        """
        images = self.preprocess_image(batched_inputs)

        if not self.training:
            # return self.inference(images)
            return self.inference(images, batched_inputs)

        image_shape = images.tensor.shape[-2:]

        features = self.backbone(images.tensor)

        # features = features[self.cfg.MODEL.RESNETS.OUT_FEATURES[0]]
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)

        loss = {}
        # KD loss
        if model_ts is not None:
            kd_losses = {}
            for idx, model_t in enumerate(model_ts):
                with torch.no_grad():
                    teacher_output = model_t.backbone(images.tensor)
                    teacher_output = model_t.upsample(teacher_output)
                    teacher_output = model_t.head(teacher_output)
                kd_loss = self.kd_loss(pred_dict, teacher_output, idx) #, model_ts, images)
                kd_losses = {**kd_losses, **kd_loss}
            if self.kd_without_label:
                loss = kd_losses
                return loss
            else:
                loss = {**kd_losses, **loss}

        gt_dict = self.get_ground_truth(batched_inputs, image_shape)
        gt_loss = self.losses(pred_dict, gt_dict)

        loss = {**loss, **gt_loss}
        return loss

    @torch.no_grad()
    def inference(self, images, batched_inputs, K=100):

        features = self.backbone(images.tensor)
        # features = features[self.cfg.MODEL.RESNETS.OUT_FEATURES[0]]
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)

        heat, wh, reg = pred_dict['cls'], pred_dict['wh'], pred_dict['reg']
        batch, cat, height, width = heat.size()
        bboxes, scores, clses = CenterNetDecoder.decode(heat, wh, reg)
        clses = clses.view(batch, K)  # .float()
        scores = scores.view(batch, K)
        results = []
        for i in range(batch):
            scale_x, scale_y = batched_inputs[i]['width'] / float(images.image_sizes[i][1]), \
                               batched_inputs[i]['height'] / float(images.image_sizes[i][0])
            # print(batched_inputs[i], images.image_sizes[i])
            # print(scale_x,scale_y)
            result = Instances(images.image_sizes[i])
            bboxes[i, :, 0::2] = bboxes[i, :, 0::2] * scale_x * self.cfg.MODEL.CENTERNET.DOWN_SCALE
            bboxes[i, :, 1::2] = bboxes[i, :, 1::2] * scale_y * self.cfg.MODEL.CENTERNET.DOWN_SCALE

            # import cv2
            # image = cv2.imread(batched_inputs[i]['file_name'])
            # for j,bbox in enumerate(bboxes[i]):
            #     if scores[i][j] > 0.1:
            #         cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
            # cv2.imwrite('result.jpg',image)
            # import pdb; pdb.set_trace()

            result.pred_boxes = Boxes(bboxes[i])
            result.scores = scores[i]
            result.pred_classes = clses[i]
            results.append({"instances": result})

        return results

    def losses(self, pred_dict, gt_dict):
        r"""
        calculate losses of pred and gt

        Args:
            gt_dict(dict): a dict contains all information of gt
            gt_dict = {
                "score_map": gt scoremap,
                "wh": gt width and height of boxes,
                "reg": gt regression of box center point,
                "reg_mask": mask of regression,
                "index": gt index,
            }
            pred(dict): a dict contains all information of prediction
            pred = {
            "cls": predicted score map
            "reg": predcited regression
            "wh": predicted width and height of box
        }
        """
        # scoremap loss
        pred_score = pred_dict["cls"]
        cur_device = pred_score.device
        for k in gt_dict:
            gt_dict[k] = gt_dict[k].to(cur_device)

        if self.cfg.MODEL.CENTERNET.LOSS.IGNORE_UNLABEL:
            loss_cls = ignore_unlabel_focal_loss(pred_score, gt_dict["score_map"])
        elif self.cfg.MODEL.CENTERNET.LOSS.MSE:
            loss_cls = mse_loss(pred_score, gt_dict["score_map"])
        else:
            loss_cls = modified_focal_loss(pred_score, gt_dict["score_map"])

        mask = gt_dict["reg_mask"]
        index = gt_dict["index"]
        index = index.to(torch.long)
        # width and height loss, better version
        norm_wh =  self.cfg.MODEL.CENTERNET.LOSS.NORM_WH
        loss_wh = reg_l1_loss(pred_dict["wh"], mask, index, gt_dict["wh"], norm_wh)

        # regression loss
        loss_reg = reg_l1_loss(pred_dict["reg"], mask, index, gt_dict["reg"])

        loss_cls *= self.cfg.MODEL.CENTERNET.LOSS.CLS_WEIGHT
        loss_wh *= self.cfg.MODEL.CENTERNET.LOSS.WH_WEIGHT
        loss_reg *= self.cfg.MODEL.CENTERNET.LOSS.REG_WEIGHT

        loss = {"loss_cls": loss_cls, "loss_box_wh": loss_wh, "loss_off_reg": loss_reg}
        # print(loss)
        return loss

    @torch.no_grad()
    def get_ground_truth(self, batched_inputs, image_shape):
        return CenterNetGT.generate(self.cfg, batched_inputs, image_shape)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(img / 255.) for img in images]
        # images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        images = ImageList.from_tensors(images, 32)
        return images


def build_model(cfg):
    model = CenterNet(cfg)
    return model
