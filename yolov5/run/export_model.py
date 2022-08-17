import argparse
import os
from models.common import DetectMultiBackend
from models.yolo import Model
import yaml
from utils.load_images import LoadImages
from utils.model_conversion import CalibrationDataPipeline, model_to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--quant_img_count", type=int, default=500)
    args = parser.parse_args()

    assert len(args.name) == len(args.weights)

    for model_name, weights in zip(args.name, args.weights):
        # model_name = args.name #  "yolov5m"
        config = f"models/{model_name}.yaml"  # "runs/train/m_bdd/weights/best.pt"
        input_size = (args.size, args.size)
        batch_size = 1
        model_path = "out/export"
        devices = ["onnx", "onnx_i8"]

        class_names = [
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motor",
            "bike",
            "traffic light",
            "traffic sign",
        ]

        Model(config)  # to load global config
        model = DetectMultiBackend(weights, device="cpu", dnn=False, data=None)
        model.eval()

        model = model.model
        det_layer = model.model[-1]

        det_layer.set_mode("export")

        convert_params = {
            "input_shape": (3, input_size[1], input_size[0]),
            "output_names": ("feat1", "feat2", "feat3"),
            "calib_data": CalibrationDataPipeline([
                LoadImages("datasets/bdd100k/images/small/train", output_dict=False, resize=(input_size[0], input_size[1]), as_tensor=False, img_count=args.quant_img_count), 
            ], model_in_name="input", batch_size=batch_size), 
            "model_path": model_path,
            "model_name": model_name,    
            "batch_size": batch_size,
            "quant": "dfg",
            # "input_type": input_type,
            # "input_format": input_format
            # "opset": 12
        }

        info = dict(
            anchors=det_layer.anchors.numpy().tolist(),
            class_names=class_names
        )

        info_file = os.path.join(model_path, model_name + ".yaml")

        for device in devices:
            model_to_device(model, device, **convert_params)

        with open(info_file, "w") as f:
            yaml.dump(info, f)


if __name__ == "__main__":
    main()
