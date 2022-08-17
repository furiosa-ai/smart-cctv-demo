import importlib
import sys
import pytest
import subprocess

sys.path.append(".")
# list of Cython modules containing tests
cython_test_modules = ["utils.mot.mot_tests"]

"""
@pytest.fixture(autouse = True)
def wrapper(request):
    print('\nbefore: {}'.format(request.node.name))
    yield
    print('\nafter: {}'.format(request.node.name))
"""

"""
@pytest.fixture(scope='session', autouse=True)
def setup():
    command = "source /Users/kevin/anaconda3/etc/profile.d/conda.sh && conda activate torch && python setup.py build_ext --inplace"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    
    assert process.returncode == 0
"""


for mod in cython_test_modules:
    try:
        # For each callable in `mod` with name `test_*`,
        # set the result as an attribute of this module.
        mod = importlib.import_module(mod)
        for name in dir(mod):
            item = getattr(mod, name)
            if callable(item) and name.startswith("test_"):
                setattr(sys.modules[__name__], name, item)
    except ImportError:
        pass
