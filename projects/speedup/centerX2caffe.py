from types import MethodType
import torch

import argparse
import sys
import torch.nn as nn

sys.path.insert(0, '.')
from configs import add_centernet_config
from detectron2.config import get_cfg
from inference.centernet import build_model
from detectron2.checkpoint import DetectionCheckpointer
from fvcore.common.file_io import PathManager

sys.path.append('speedup/Caffe')
import pytorch_to_caffe

def centerX_forward(self, x):
    #x = ((x / 255.) - self.mean) / self.std
    y = self._forward(x)
    fmap_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(y['cls'])
    relu = nn.ReLU()(y['cls'])
    #keep = (y['cls'] - fmap_max) + 1e-9
    #keep = nn.ReLU()(keep)
    #keep = keep * 1e9
    #result = y['cls'] * keep
    ret = [relu, fmap_max, y['reg'], y['wh']]  ## change dict to list
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
        default='onnx_model',
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


def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model



if __name__ == '__main__':
    args = get_parser().parse_args()
    model = load_model(args.config_file, args.model_path)

    inputs = torch.randn(1, 3, args.input_h, args.input_w).cuda()

    PathManager.mkdirs(args.output)

    pytorch_to_caffe.trans_net(model, inputs, args.name)
    pytorch_to_caffe.save_prototxt(f"{args.output}/{args.name}.prototxt")
    pytorch_to_caffe.save_caffemodel(f"{args.output}/{args.name}.caffemodel")

    print(f"Export caffe model in {args.output} successfully!")
