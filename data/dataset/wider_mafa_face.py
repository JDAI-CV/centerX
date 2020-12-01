import os
import json
import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import pickle
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
from fvcore.common.file_io import PathManager
import logging

__all__ = ["load_face_instances", "register_face"]

# fmt: off
CLASS_NAMES = ("face",)


# fmt: on


def load_face_instances(txt, annotation_dirname, image_root, class_names):
    """
    Load crowdhuman detection annotations to Detectron2 format.
    """
    # Needs to read many small annotation files. Makes sense at local

    lines = open(txt).readlines()
    dicts = []
    for line in lines:
        fileid = line.strip()
        jpeg_file = os.path.join(image_root, fileid + ".jpg")
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)

    return dicts


def register_face(name, txt, annotation_dirname, image_root, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_face_instances(txt, annotation_dirname, image_root, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names)
    )
