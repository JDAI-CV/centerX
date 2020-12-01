import os
import json
import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import pickle
import logging

__all__ = ["load_crowd_instances", "register_crowd"]

# fmt: off
CLASS_NAMES = ("person",)


# fmt: on


def load_crowd_instances(json_file, image_root, class_names):
    """
    Load crowdhuman detection annotations to Detectron2 format.
    """
    # Needs to read many small annotation files. Makes sense at local
    cache_file = json_file + '.cache'
    if os.path.exists(json_file + '.cache'):
        logging.getLogger('detectron2').info('using cache ' + cache_file)
        return pickle.load(open(cache_file, 'rb'))

    lines = open(json_file).readlines()
    dicts = []
    for line in lines:
        anno = json.loads(line.strip('\n'))
        fileid = anno['ID']
        jpeg_file = os.path.join(image_root, fileid + ".jpg")

        h, w, c = cv2.imread(jpeg_file).shape

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": h,
            "width": w,
        }
        instances = []

        for obj in anno['gtboxes']:
            cls = obj['tag']
            if cls != "person":
                continue
            bbox = obj['vbox']
            bbox = [float(x) for x in bbox]
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            bbox[0] = max(bbox[0], 0.0)
            bbox[1] = max(bbox[1], 0.0)
            bbox[2] = min(bbox[2], float(w))
            bbox[3] = min(bbox[3], float(h))
            if bbox[2] - bbox[0] > 1.0 and bbox[3] - bbox[1] > 1.0:
                instances.append(
                    {"category_id": class_names.index(cls),
                     "bbox": bbox,
                     "bbox_mode": BoxMode.XYXY_ABS}
                )
        r["annotations"] = instances
        if len(instances) > 0:
            dicts.append(r)

    if not os.path.exists(json_file + '.cache'):
        f = open(json_file + '.cache', 'wb')
        pickle.dump(dicts, f)
        f.close()
    return dicts


def register_crowd(name, json_file, image_root, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_crowd_instances(json_file, image_root, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names)
    )
