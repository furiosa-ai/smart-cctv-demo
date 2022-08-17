import sys
import importlib


class ImpEnv:
    def __init__(self, paths=None, imports=None, imports_keep=None) -> None:
        if paths is None:
            paths = []

        if not isinstance(paths, (list, tuple)):
            paths = [paths]

        imports = imports if imports is not None else []

        imps = []
        for imp in imports:
            parts = imp.split(".")
            cur_mod = parts[0]
            imps.append(cur_mod)
            for part in parts[1:]:
                cur_mod += "." + part
                imps.append(cur_mod)

        self.new_paths = paths
        self.path = [*sys.path]      
        self.modules = None
        self.imports = imps
        self.imports_keep = imports_keep if imports_keep is not None else []

    def __enter__(self):
        self.modules = {**sys.modules}

        self.path = [*sys.path]      
        for p in self.new_paths:
            sys.path.insert(0, str(p))

        for imp in self.imports:
            importlib.reload(importlib.import_module(imp))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.path = self.path

        for name in list(sys.modules):
            if name in self.imports_keep:
                pass
            elif name not in self.modules:
                del sys.modules[name]
            elif name in self.imports:
                # reset to old module
                sys.modules[name] = self.modules[name]
