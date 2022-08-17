import os


# takes frozen graph
# output_path is a folder
def model_to_openvino(input_file, output_path, input_shape=None, fp16=False):
    in_path = os.path.abspath(input_file)
    out_name = os.path.abspath(output_path)
    mo_dir = "/opt/intel/openvino/deployment_tools/model_optimizer"
    mo_script = "mo.py"

    mo_args = ["--input_model", in_path, "--output_dir", out_name]

    if input_shape is not None:
        mo_args += ["--input_shape", f"[{','.join([str(v) for v in (1, *input_shape)])}]"]

    if fp16:
        mo_args += ["--data_type", "FP16"]

    # mo_args += ["--log_level=DEBUG"]

    """
    cmd = ""

    cmd += f"cd {mo_dir};\n"
    cmd += f"./{mo_script} {' '.join(mo_args)};\n"
    """

    cmd = f"bash -c '\n" \
          f". ~/miniconda3/etc/profile.d/conda.sh\n" \
          f"conda activate openvino\n" \
          f"cd {mo_dir};\n" \
          f"./{mo_script} {' '.join(mo_args)}\n" \
          f"'"

    print(cmd)
    os.system(cmd)
