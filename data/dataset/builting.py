from .crowd_human import register_crowd
from .coco_class import register_coco_class
from .wider_mafa_face import register_face
from .pseudo_label import register_pseudo
from detectron2.data import MetadataCatalog
import os

def register_all_crowd(root):
    SPLITS = [
        ("crowd_human_train",
         "datasets/crowd_human/Annotations/annotation_train.odgt",
         "datasets/crowd_human/JPEGImages/",
         ["person",]),
        ("crowd_human_val",
         "datasets/crowd_human/Annotations/annotation_val.odgt",
         "datasets/crowd_human/JPEGImages/",
         ["person",]),
        ("crowd_human_face_train",
         "datasets/crowd_human/Annotations/annotation_train.odgt",
         "datasets/crowd_human/JPEGImages/",
         ["person","face"]),
        ("crowd_human_face_val",
         "datasets/crowd_human/Annotations/annotation_val.odgt",
         "datasets/crowd_human/JPEGImages/",
         ["person", "face"]),
    ]
    for name, json_file, image_root, class_names in SPLITS:
        register_crowd(name, json_file, image_root, class_names)
        MetadataCatalog.get(name).evaluator_type = "crowd_human"


def register_all_face(root):
    SPLITS = [
        ("face_train",
         "datasets/wider_face_add_lm_10_10_add_mafa/ImageSets/Main/trainval_all.txt",
         "datasets/wider_face_add_lm_10_10_add_mafa/Annotations/",
         "datasets/wider_face_add_lm_10_10_add_mafa/JPEGImages/",
         ["face",]),
        ("face_test",
         "datasets/wider_face_add_lm_10_10_add_mafa/ImageSets/Main/test.txt",
         "datasets/wider_face_add_lm_10_10_add_mafa/Annotations/",
         "datasets/wider_face_add_lm_10_10_add_mafa/JPEGImages/",
         ["face",]),
        ("widerface_crowdhuman_train",
         "datasets/wider_face_add_lm_10_10_add_mafa/ImageSets/Main/trainval_all.txt",
         "datasets/wider_face_add_lm_10_10_add_mafa/Annotations/",
         "datasets/wider_face_add_lm_10_10_add_mafa/JPEGImages/",
         ["person","face"]),
        ("widerface_crowdhuman_test",
         "datasets/wider_face_add_lm_10_10_add_mafa/ImageSets/Main/test.txt",
         "datasets/wider_face_add_lm_10_10_add_mafa/Annotations/",
         "datasets/wider_face_add_lm_10_10_add_mafa/JPEGImages/",
         ["person", "face"]),
    ]
    for name, txt, xml_root, image_root, class_names in SPLITS:
        register_face(name, txt, xml_root, image_root, class_names)
        MetadataCatalog.get(name).evaluator_type = "face"

def register_all_coco_class(root):
    SPLITS = [
        ("coco_person_train",
         "datasets/coco/annotations/instances_train2017.json",
         "datasets/coco/train2017/",
         ["person",]),
        ("coco_person_val",
         "datasets/coco/annotations/instances_val2017.json",
         "datasets/coco/val2017/",
         ["person", ]),
        ("coco_car_train",
         "datasets/coco/annotations/instances_train2017.json",
         "datasets/coco/train2017/",
         ["car",]),
        ("coco_car_val",
         "datasets/coco/annotations/instances_val2017.json",
         "datasets/coco/val2017/",
         ["", "car", ]),
        ("coco_person_car_val",
         "datasets/coco/annotations/instances_val2017.json",
         "datasets/coco/val2017/",
         ["person", "car"]),
    ]
    for name, json_file, image_root, class_names in SPLITS:
        register_coco_class(name, json_file, image_root, class_names)
        MetadataCatalog.get(name).evaluator_type = "coco_class"

def register_all_pseudo(root):
    SPLITS = [
        ("pseudo_person_face_train",
        ["person", "face"]),
    ]
    for name, class_names in SPLITS:
        register_pseudo(name, class_names)
        MetadataCatalog.get(name).evaluator_type = "pseudo_person_face"

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_crowd(_root)
register_all_coco_class(_root)
register_all_face(_root)
register_all_pseudo(_root)