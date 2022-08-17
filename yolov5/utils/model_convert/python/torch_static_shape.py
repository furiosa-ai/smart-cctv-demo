

def _static_tensor_size(tensor, i=None):
    import torch
    s = [int(x) for x in tensor.shape]
    # s = torch.tensor(s, dtype=torch.int64)

    if i is not None:
        # s = s[i]
        s = torch.tensor(s[i])

    return s


class TorchStaticShape:
    def __init__(self):
        self.old_shape_func = None
        self.old_size_func = None

    def __enter__(self):
        import torch
        self.old_size_func = torch.Tensor.size
        # self.old_shape_func = torch.Tensor.shape

        torch.Tensor.size = _static_tensor_size
        # torch.Tensor.shape = _static_tensor_size

    def __exit__(self, exc_type, exc_val, exc_tb):
        import torch
        torch.Tensor.size = self.old_size_func
        # torch.Tensor.shape = self.old_shape_func
