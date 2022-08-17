import ctypes

try:
    lib = ctypes.cdll.LoadLibrary(None)
    prctl = lib.prctl
    prctl.restype = ctypes.c_int
    prctl.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_ulong,
                    ctypes.c_ulong, ctypes.c_ulong]
except:
    prctl = None

def set_proctitle(new_title):
    if prctl is not None:
        result = prctl(15, new_title, 0, 0, 0)
    # if result != 0:
    #     raise OSError("prctl result: %d" % result)