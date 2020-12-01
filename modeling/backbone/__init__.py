from .build import build_backbone, BACKBONE_REGISTRY, get_norm

from .resnet import build_resnet
from .regnet import build_regnet
from .ori_resnet import build_torch_backbone