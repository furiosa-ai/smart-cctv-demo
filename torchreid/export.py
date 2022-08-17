import torchreid
import torch


def main():
    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources="market1501",
        targets="market1501",
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"]
    )

    model_names = [
        "resnet50",
        # "osnet_x1_0",
        # "mobilenetv2_x1_4"
    ]

    batch_sizes = [1, 64, 128, 192, 256]

    for model_name in model_names:
        model = torchreid.models.build_model(
            name=model_name,
            num_classes=datamanager.num_train_pids,
            loss="softmax",
            pretrained=True
        ).eval()

        for batch_size in batch_sizes:
            x = torch.zeros(batch_size, 3, 256, 128)

            onnx_file = f"onnx/{model_name}-b{batch_size}.onnx"

            torch.onnx.export(model, x, onnx_file, opset_version=12)

    pass


if __name__ == "__main__":
    main()
