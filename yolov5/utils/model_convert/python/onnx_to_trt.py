import os


def onnx_to_trt(onnx_file_path, output_file, batch_stream=None, precision=None, force_create=True,
                input_shape=None, max_batch_size=1):
    import tensorrt as trt

    import pycuda.driver as cuda
    import pycuda.autoinit  # need this line

    trt.init_libnvinfer_plugins(None, '')

    TRT_LOGGER = trt.Logger()
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, stream, cache_file):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.stream = stream

            self.cache_file = cache_file

            self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
            stream.reset()

        def get_batch_size(self):
            return self.stream.batch_size

        def get_batch(self, names, p_str=None):
            batch = self.stream.next_batch()
            if not batch.size:
                return None

            cuda.memcpy_htod(self.d_input, batch)

            return [self.d_input]

        def read_calibration_cache(self):
            if self.cache_file is None:
                return None

            # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()

        def write_calibration_cache(self, cache):
            if self.cache_file is not None:
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = max_batch_size if max_batch_size is not None else batch_stream.batch_size
            builder.max_workspace_size = builder.max_batch_size << 28 # 256MiB

            if precision == "fp16":
                builder.fp16_mode = True
                builder.strict_type_constraints = True
                print("Setting FP16 precision")
            elif precision == "int8":
                builder.int8_mode = True
                builder.strict_type_constraints = True

                calibration_cache = None

                Int8_calibrator = PythonEntropyCalibrator(batch_stream, cache_file=calibration_cache)

                builder.int8_calibrator = Int8_calibrator

                print("Setting INT8 precision")

            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None

            network.get_input(0).shape = [max_batch_size, *(input_shape if input_shape is not None else batch_stream.input_shape)]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

            with open(output_file, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(output_file) and not force_create:
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(output_file))
        with open(output_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
