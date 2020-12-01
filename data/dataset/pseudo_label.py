import os
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from .crowd_human import load_crowd_instances
from .wider_mafa_face import load_face_instances

__all__ = ["load_pseudo_instances", "register_pseudo"]

# fmt: off
CLASS_NAMES = ("person","face")


# fmt: on


def load_pseudo_instances(pseudo_txt, image_root, class_names):
    """
    Load crowdhuman detection annotations to Detectron2 format.
    """
    # Needs to read many small annotation files. Makes sense at local

    lines = open(pseudo_txt).readlines()
    dicts = []
    for line in lines:
        line = line.strip()
        fileid = os.path.basename(line.split(';')[0])
        jpeg_file = os.path.join(image_root, fileid)

        w, h = line.split(';')[1:3]

        r = {
            "file_name": jpeg_file,
            "image_id": fileid.strip('.jpg'),
            "height": int(h),
            "width": int(w),
        }
        instances = []
        anno = np.array(line.split(';')[3:]).reshape(-1, 5)

        for cls,x1,y1,x2,y2 in anno:
            bbox = []
            bbox.append(max(float(x1), 0.0))
            bbox.append(max(float(y1), 0.0))
            bbox.append(min(float(x2), float(w)))
            bbox.append(min(float(y2), float(h)))
            if bbox[2] - bbox[0] > 1.0 and bbox[3] - bbox[1] > 1.0:
                instances.append(
                    {"category_id": class_names.index(cls),
                     "bbox": bbox,
                     "bbox_mode": BoxMode.XYXY_ABS}
                )
        r["annotations"] = instances
        if len(instances) > 0:
            dicts.append(r)
    return dicts

def merge_lists(list_a, list_b):

    def to_dic(list_):
        dic = {}
        for r in list_:
            dic[r['file_name']] = r
        return dic
    dic_a , dic_b = to_dic(list_a), to_dic(list_b)
    dic_all = dic_b
    for key in dic_a:
        if key in dic_b:
            dic_all[key]["annotations"] = dic_all[key]["annotations"] + dic_a[key]["annotations"]
        else:
            dic_all[key] = dic_a[key]

    dic_all = [dic_all[key] for key in dic_all]
    return dic_all

def load_personface_instances(class_names):
    crowd_human_gt = load_crowd_instances(
        "datasets/crowd_human/Annotations/annotation_train.odgt",
        "datasets/crowd_human/JPEGImages/",
        class_names)
    face_gt = load_face_instances(
        "datasets/wider_face_add_lm_10_10_add_mafa/ImageSets/Main/trainval_all.txt",
        "datasets/wider_face_add_lm_10_10_add_mafa/Annotations/",
        "datasets/wider_face_add_lm_10_10_add_mafa/JPEGImages/",
        class_names)
    human_pseudo = load_pseudo_instances(
        "datasets/wider_face_add_lm_10_10_add_mafa/widerface_pseudo_human.txt",
        "datasets/wider_face_add_lm_10_10_add_mafa/JPEGImages/",
        class_names)
    face_pseudo = load_pseudo_instances(
        "datasets/crowd_human/crowdhuman_pseudo_face.txt",
        "datasets/crowd_human/JPEGImages/",
        class_names)
    crowd_human = merge_lists(crowd_human_gt,face_pseudo)
    face = merge_lists(face_gt, human_pseudo)
    person_face = merge_lists(crowd_human,face)
    return person_face


def register_pseudo(name, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_personface_instances(class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names)
    )
