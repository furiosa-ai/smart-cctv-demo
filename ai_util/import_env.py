import importlib
import sys
import inspect

_dft_modules = {**sys.modules}


def _get_module_path(mod):
    path = mod.__file__

    if path is None:
        path = mod.__path__._path[0]
        
    assert path is not None

    return path

class ImportEnv:
    local_modules = []

    def __init__(self) -> None:
        self.new_path_count = 0
        self.modules_keep = None
        self.prefix = "/Users/kevin/Documents/projects/smart-cctv-demo"

    def add_path(self, path):
        sys.path.insert(0, str(path))
        self.new_path_count += 1
        return self

    def unload_module(self, name):
        pass

    def __enter__(self):
        self.modules_keep = {}

        for name, mod in list(sys.modules.items()):
            if (self.prefix in str(mod) or name == "utils") and not name.startswith("ai_util") and not name.startswith("insightface"):
            # if name.startswith("utils"):
                # del sys.modules[name]
                # self.modules_keep[name] = sys.modules.pop(name)
                importlib.reload(sys.modules[name])
                # importlib.import_module()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return
        for _ in range(self.new_path_count):
            del sys.path[0]

        return
        for name, mod in self.modules_keep.items():
            if name in sys.modules:
                if _get_module_path(mod) != _get_module_path(sys.modules[name]):
                    del sys.modules[name]
                    importlib.import_module(name)
                    # sys.modules[name] = mod
                    # del sys.modules[name]

        # for name, mod in self.modules_keep.items():
            # importlib.import_module(name)
            # sys.modules[name] = mod
            # importlib.reload(mod)
        # self.modules.update(sys.modules)
        # sys.modules = self.modules
        # sys.modules.update(self.modules)

        # sys.modules = self.modules
        # pass


class ImportEnv2:
    def __init__(self) -> None:
        self.new_path_count = 0
        self.mod_filter = []
        self.modules_unloaded = {}

    def add_path(self, path):
        sys.path.insert(0, str(path))
        self.new_path_count += 1
        return self

    def unload_module(self, name):
        self.mod_filter.append(name)

        for mod_name in list(sys.modules):
            # if mod_name.startswith(name):
            if mod_name == name:
                self.modules_unloaded[mod_name] = sys.modules[mod_name]  # if name in sys.modules else None
                del sys.modules[mod_name]
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for _ in range(self.new_path_count):
            del sys.path[0]

        # for m in list(sys.modules):
        #     if any(m.startswith(f) for f in self.mod_filter):
        #         del sys.modules[m]

        # sys.modules = {m: mod for m, mod in sys.modules.items() if not m in self.mod_filter}

        for name, module in self.modules_unloaded.items():
            if module is None and name in sys.modules:
                del sys.modules[name]
            else:
                sys.modules[name] = module

