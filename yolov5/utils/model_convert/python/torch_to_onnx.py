from .util.util import check_onnx_model


def torch_to_onnx(pytorch_model, output_file, input_shape, opset=None, check_model=True,
                  input_names=("input",), output_names=("output",), batch_size=1, dyn_batch_size=False,
                  runtime_test=False, dry_run=True,
                  **kwargs):
    import onnx
    import torch

    if isinstance(input_shape[0], (list, tuple)):
        dummy_input = [torch.zeros((batch_size, *s)) for s in input_shape]
    else:
        dummy_input = torch.zeros((batch_size, *input_shape))

    dynamic_axes = {node: {0: 'batch_size'} for node in input_names + output_names} if dyn_batch_size else None

    if dry_run:
        _ = pytorch_model(dummy_input)

    print("Converting to ONNX")
    torch.onnx.export(
        pytorch_model, dummy_input, output_file,
        input_names=input_names, output_names=output_names,
        opset_version=opset, verbose=True, dynamic_axes=dynamic_axes,
        **kwargs)
    print("Converting done")

    if check_model:
        # Validate exported model
        model = onnx.load(output_file)  # Load the ONNX model
        print("Checking onnx export")
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print("ONNX Graph")
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        print("Checking successfull")

    if runtime_test:
        import numpy as np
        import onnxruntime

        x = torch.rand((batch_size, *input_shape))

        pytorch_model.eval()
        torch_out = pytorch_model(x)[0]

        ort_session = onnxruntime.InferenceSession(output_file)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    print("Done")
