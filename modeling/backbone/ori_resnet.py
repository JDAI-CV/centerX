from .build import BACKBONE_REGISTRY
from detectron2.modeling import Backbone
from detectron2.layers import ShapeSpec
import torch.nn as nn

import torchvision.models.resnet as resnet
_resnet_mapper = {18: resnet.resnet18, 50: resnet.resnet50, 101: resnet.resnet101}

class ResnetBackbone(Backbone):
    def __init__(self, cfg, input_shape=None, pretrained=True):
        super().__init__()
        depth = cfg.MODEL.RESNETS.DEPTH
        backbone = _resnet_mapper[depth](pretrained=pretrained)
        self.stage0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

@BACKBONE_REGISTRY.register()
def build_torch_backbone(cfg, input_shape=None):
    """
    Build a backbone.
    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = ResnetBackbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone
