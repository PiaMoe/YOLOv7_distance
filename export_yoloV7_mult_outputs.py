import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
import models
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.activations import Hardswish, SiLU

class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x[:, :, :4]
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=boxes.dtype, device=boxes.device)
        boxes @= convert_matrix

        objectness = x[:, :, 4:5]
        class_scores = x[:, :, 5:-3]
        scores, labels = torch.max(class_scores, dim=2, keepdim=True)
        scores = (scores * objectness).float()
        scores = torch.round(scores * 100) / 100  # Truncate to two decimal places

        distances = x[:, :, -3].round().to(torch.int32).unsqueeze(-1)  # (B, N, 1)
        sinp = x[:, :, -2]
        cosp = x[:, :, -1]
        heading_deg = (torch.atan2(sinp, cosp) * 180.0 / torch.pi) % 360
        heading_deg = heading_deg.round().to(torch.int32).unsqueeze(-1)  # (B, N, 1)

        # encode score, distance into a single float value
        encoded = scores + distances / 1e6 + heading_deg / 1e9
        encoded = encoded.float()

        return torch.cat([boxes, encoded, labels.to(boxes.dtype)], dim=-1)


def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def yolov7_export(weights, device):
    model = attempt_load(weights, map_location=device)
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()
        if isinstance(m, models.common.Conv):
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    model.model[-1].export = False
    model.model[-1].concat = True
    model.eval()
    return model


def main(args):
    suppress_warnings()
    device = select_device('cpu')
    model = yolov7_export(args.weights, device)
    model = nn.Sequential(model, DeepStreamOutput())

    img_size = args.size * 2 if len(args.size) == 1 else args.size
    onnx_input = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = os.path.basename(args.weights).split('.pt')[0] + '.onnx'

    # --- dynamic axes only for input/output tensor
    dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}

    torch.onnx.export(
        model,
        onnx_input,
        onnx_output_file,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes if args.dynamic else None
    )

    if args.simplify:
        import onnx
        import onnxsim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx, _ = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f"Done: {onnx_output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='DeepStream YOLOv7 conversion')
    parser.add_argument('-w', '--weights', default='../runs/train/finalDataset/B3_singleHead/weights/best.pt', help='Input .pt file')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[608,1088], help='Inference size')
    parser.add_argument('--p6', action='store_true')
    parser.add_argument('--opset', type=int, default=12)
    parser.add_argument('--simplify', action='store_true')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--batch', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
