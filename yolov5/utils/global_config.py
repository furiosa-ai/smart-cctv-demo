

global_config_dft = {
    "export_skip_postproc": False,
    "force_act": None,
    "use_add": False,
    "allow_add_pad": False,
    "no_ln": False
}

global_config = {**global_config_dft}


def reset_global_cfg():
    for k in global_config:
        global_config[k] = global_config_dft[k]
