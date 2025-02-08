import onnx
from onnxruntime.quantization.quantize import quantize_dynamic
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.preprocess import quant_pre_process
from onnx.shape_inference import infer_shapes


class DataReader(CalibrationDataReader):

    def get_next(self) -> dict:
        pass


def main():
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


if __name__ == "__main__":
    main()
