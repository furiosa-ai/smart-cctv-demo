import importlib
import os
import os.path as osp
from ai_util.imp_env import ImpEnv

import configs.base
import sys
from pathlib import Path


def get_config(config_file, args):
    with ImpEnv(Path(__file__).parent.parent):
        assert config_file.startswith('configs/'), 'config file setting must start with configs/'
        temp_config_name = osp.basename(config_file)
        temp_module_name = osp.splitext(temp_config_name)[0]
        config = importlib.import_module("configs.base")
        cfg = config.config
        config = importlib.import_module("configs.%s" % temp_module_name)
        job_cfg = config.config
        cfg.update(job_cfg)
        # if cfg.output is None:
        #     cfg.output = osp.join('work_dirs', temp_module_name)
        if hasattr(args, "name"):
            assert cfg.output is None
            cfg.output = osp.join('runs', args.name)
            assert not osp.exists(cfg.output)
        
        return cfg