import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data import CenterAffine


def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()


    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


class CenterNetDecoder(object):
    @staticmethod
    def decode(fmap, wh, reg=None, cat_spec_wh=False, K=100):
        r"""
        decode output feature map to detection results

        Args:
            fmap(Tensor): output feature map
            wh(Tensor): tensor that represents predicted width-height
            reg(Tensor): tensor that represens regression of center points
            cat_spec_wh(bool): whether apply gather on tensor `wh` or not
            K(int): topk value
        """
        batch, channel, height, width = fmap.shape

        fmap = CenterNetDecoder.pseudo_nms(fmap)

        scores, index, clses, ys, xs = CenterNetDecoder.topk_score(fmap, K=K)
        if reg is not None:
            reg = gather_feature(reg, index, use_transform=True)
            reg = reg.reshape(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = gather_feature(wh, index, use_transform=True)

        if cat_spec_wh:
            wh = wh.view(batch, K, channel, 2)
            clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
            wh = wh.gather(2, clses_ind).reshape(batch, K, 2)
        else:
            wh = wh.reshape(batch, K, 2)

        clses = clses.reshape(batch, K, 1).float()
        scores = scores.reshape(batch, K, 1)

        half_w, half_h = wh[..., 0:1] / 2, wh[..., 1:2] / 2
        bboxes = torch.cat([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=2)

        detections = (bboxes, scores, clses)

        return detections

    @staticmethod
    def transform_boxes(boxes, img_info, scale=1):
        r"""
        transform predicted boxes to target boxes

        Args:
            boxes(Tensor): torch Tensor with (Batch, N, 4) shape
            img_info(dict): dict contains all information of original image
            scale(float): used for multiscale testing
        """
        boxes = boxes.cpu().numpy().reshape(-1, 4)

        center = img_info["center"]
        size = img_info["size"]
        output_size = (img_info["width"], img_info["height"])
        src, dst = CenterAffine.generate_src_and_dst(center, size, output_size)
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

        coords = boxes.reshape(-1, 2)
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        target_boxes = np.dot(aug_coords, trans.T).reshape(-1, 4)
        return target_boxes

    @staticmethod
    def pseudo_nms(fmap, pool_size=3):
        r"""
        apply max pooling to get the same effect of nms

        Args:
            fmap(Tensor): output tensor of previous step
            pool_size(int): size of max-pooling
        """
        pad = (pool_size - 1) // 2
        fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=pad)
        keep = (fmap_max == fmap).float()
        return fmap * keep

    @staticmethod
    def topk_score(scores, K=40):
        """
        get top K point in score map
        """
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = (index / K).int()
        topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
        topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
