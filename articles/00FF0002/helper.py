import onnx
import torch
from torch import nn


class InputWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor):
        data = data.unsqueeze(0).permute(0, 3, 1, 2)
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        std = torch.tensor([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        return (data - mean) / std


class OutputWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor):
        data = (data - data.min()) / (data.max() - data.min()) * 255
        return data.type(torch.uint8)


def create():

    model1 = InputWrapper()
    fake1 = torch.rand((518, 518, 3))
    torch.onnx.export(
        model1, fake1, 'models/start_wrapper.onnx',
        input_names=['img_in'],
        output_names=['torch_ready'],
        opset_version=18
    )

    model2 = OutputWrapper()
    fake2 = torch.rand((1, 518, 518))
    torch.onnx.export(
        model2, fake2, 'models/end_wrapper.onnx',
        input_names=['torch_result'],
        output_names=['img_out'],
        opset_version=18
    )


def compose():
    p1 = onnx.load_model('models/start_wrapper.onnx')
    p2 = onnx.load_model('models/depth-anything-q2.onnx')
    p3 = onnx.load_model('models/end_wrapper.onnx')
    m1 = onnx.compose.add_prefix(p1, 'm1.')
    m2 = onnx.compose.add_prefix(p2, 'm2.')
    m3 = onnx.compose.add_prefix(p3, 'm3.')

    c1 = onnx.compose.merge_models(m1, m2, [('m1.torch_ready', 'm2.l_x_')])
    c2 = onnx.compose.merge_models(c1, m3, [('m2.select_36', 'm3.torch_result')])
    onnx.save_model(c2, 'models/depth-anything-web-q2.onnx')


create()
compose()
