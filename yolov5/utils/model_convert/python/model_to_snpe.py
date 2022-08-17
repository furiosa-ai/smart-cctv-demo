import os


def model_to_snpe(model_type, input_file, output_file, input_names=(), input_shapes=(), output_names=(),
                  check_snpe_model=False, allow_unconsumed_nodes=False, strict=False):
    print("Converting to SNPE")

    if model_type == "onnx":
        snpe_cmd = "snpe-onnx-to-dlc"
        snpe_lib = "-o ~/miniconda3/envs/qc/lib/python3.5/site-packages/onnx/"
    elif model_type == "tf":
        snpe_cmd = "snpe-tensorflow-to-dlc"
        snpe_lib = "-t ~/miniconda3/envs/qc/lib/python3.5/site-packages/tensorflow/"
    else:
        raise Exception(model_type)

    snpe_args = []
    snpe_args += ["--input_network", input_file]
    snpe_args += ["-o", output_file]

    for n, s in zip(input_names, input_shapes):
        snpe_args += ["--input_dim", f"\"{n}\" {','.join([str(x) for x in s])}"]

    for n in output_names:
        snpe_args += ["--out_node", f"\"{n}\""]

    if allow_unconsumed_nodes:
        snpe_args += ["--allow_unconsumed_nodes"]

    if strict:
        snpe_args += ["--strict"]

    cmd = f"bash -c '\n" \
          f". ~/miniconda3/etc/profile.d/conda.sh\n" \
          f"conda activate qc\n" \
          f". ~/snpe-1.40.0.2130/bin/envsetup.sh {snpe_lib}\n" \
          f"{snpe_cmd} {' '.join(snpe_args)}\n" \
          f"'"

    print(cmd)
    os.system(cmd)

    if check_snpe_model:
        assert len(input_shapes) == 1

        in_shape_arg = " ".join([str(x) for x in (input_shapes[0][1:] + input_shapes[0][:1])])
        output_file_abs = os.path.abspath(output_file)

        cmd = f"bash -c '\n" \
              f". ~/snpe-1.40.0.2130/bin/envsetup.sh -o ~/miniconda3/envs/qc/lib/python3.5/site-packages/onnx/\n" \
              f"cd ~/Documents/projects/SnpeInference/python\n" \
              f"python3 perf_test.py --model {output_file_abs} --input_shape {in_shape_arg} --buffer itensor --arch cpu\n" \
              f"'"

        print(cmd)
        os.system(cmd)