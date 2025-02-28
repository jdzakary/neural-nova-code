import os

import cv2
import numpy as np
import onnx
import torch
from onnxruntime.quantization.quantize import quantize_dynamic, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.preprocess import quant_pre_process


class DataReader(CalibrationDataReader):
    def __init__(self, folder: str):
        files = [
            os.path.join(folder, x) for x in os.listdir(folder)
        ]
        self.__files = [
            x for x in files
            if not os.path.isdir(x)
            and x.endswith(('.jpg', 'jpeg', 'png'))
        ]

    def get_next(self) -> dict | None:
        if len(self.__files):
            return {'l_x_': self.load_image(self.__files.pop())}
        return None

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255
        img = cv2.resize(img, (518, 518))
        print(img.dtype)
        data = np.transpose(img, (2, 0, 1))
        data = np.expand_dims(data, 0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((3, 1, 1))
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((3, 1, 1))
        return (data - mean) / std


def dynamic():
    print('Pre Processing')
    quant_pre_process(
        input_model='models/depth-anything.onnx',
        output_model_path='models/temp.onnx',
        skip_symbolic_shape=True,
    )
    print('Quantize')
    quantize_dynamic(
        model_input='models/temp.onnx',
        model_output='models/depth-anything-q1.onnx',
    )


def static():
    print('Pre Processing')
    quant_pre_process(
        input_model='models/depth-anything.onnx',
        output_model_path='models/temp.onnx',
        skip_symbolic_shape=True,
    )
    print('Quantize')
    quantize_static(
        model_input='models/temp.onnx',
        model_output='models/depth-anything-q2.onnx',
        calibration_data_reader=DataReader(r'C:\DataFiles\Media\Pictures\Me'),
    )


if __name__ == "__main__":
    static()
