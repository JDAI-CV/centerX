from types import MethodType
import torchvision
import torch
import argparse
import io
import sys
import torch.nn as nn

sys.path.insert(0, '.')
from configs import add_centernet_config
from detectron2.config import get_cfg
from inference.centernet import build_model
from detectron2.checkpoint import DetectionCheckpointer
from fvcore.common.file_io import PathManager

def centerX_forward(self, x):
    x = self.normalizer(x / 255.)
    y = self._forward(x)
    fmap_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(y['cls'])
    keep = (y['cls'] - fmap_max).float() + 1e-9
    keep = nn.ReLU()(keep)
    keep = keep * 1e9
    result = y['cls'] * keep
    ret = [result,y['reg'],y['wh']]  ## change dict to list
    return ret

def load_model(config_file,model_path):
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(config_file)
    forward = {'centerX': centerX_forward}

    # model
    model = build_model(cfg)
    model.forward = MethodType(forward['centerX'], model)
    DetectionCheckpointer(model).load(model_path)
    model.eval()
    model.cuda()
    return model

def get_parser():
    parser = argparse.ArgumentParser(description="Convert Pytorch to ONNX model")

    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--model-path",
        metavar="FILE",
        help="path to model",
    )
    parser.add_argument(
        "--name",
        default="baseline",
        help="name for converted model"
    )
    parser.add_argument(
        "--output",
        default='pt_model',
        help='path to save converted onnx model'
    )
    parser.add_argument(
        "--input_w",
        default=640,
        type=int,
        help='image_width'
    )
    parser.add_argument(
        "--input_h",
        default=384,
        type=int,
        help='image_height'
    )
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    model = load_model(args.config_file, args.model_path)

    inputs = torch.randn(1, 3, args.input_h, args.input_w).cuda()

    traced_script_module = torch.jit.trace(model, inputs)

    PathManager.mkdirs(args.output)
    traced_script_module.save(f"{args.output}/{args.name}.pt")

    print(f"Export pt model in {args.output} successfully!")
