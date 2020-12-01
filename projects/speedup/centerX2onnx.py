from types import MethodType
import onnx
import torch
from torch.onnx import OperatorExportTypes
from onnxsim import simplify
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


def export_onnx_model(model, inputs):
    """
    Trace and export a model to onnx format.
    Args:
        model (nn.Module):
        inputs (torch.Tensor): the model will be called by `model(*inputs)`
    Returns:
        an onnx model
    """
    assert isinstance(model, torch.nn.Module)

    # make sure all modules are in eval mode, onnx may change the training state
    # of the module if the states are not consistent
    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)

    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(
                model,
                inputs,
                f,
                operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                # verbose=True,  # NOTE: uncomment this for debugging
                # export_params=True,
            )
            onnx_model = onnx.load_from_string(f.getvalue())

    # Apply ONNX's Optimization
    all_passes = onnx.optimizer.get_available_passes()
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer", "fuse_bn_into_conv"]
    assert all(p in all_passes for p in passes)
    onnx_model = onnx.optimizer.optimize(onnx_model, passes)
    return onnx_model

if __name__ == '__main__':
    args = get_parser().parse_args()
    model = load_model(args.config_file, args.model_path)

    inputs = torch.randn(1, 3, args.input_h, args.input_w).cuda()
    onnx_model = export_onnx_model(model, inputs)

    model_simp, check = simplify(onnx_model)

    model_simp = remove_initializer_from_input(model_simp)

    assert check, "Simplified ONNX model could not be validated"

    PathManager.mkdirs(args.output)

    onnx.save_model(model_simp, f"{args.output}/{args.name}.onnx")

    print(f"Export onnx model in {args.output} successfully!")
