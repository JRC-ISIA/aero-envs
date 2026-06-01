import ctypes as ct
import platform
from pathlib import Path


def get_model_path() -> str:
    """
    Get the path to the model file.
    :return: The path to the model file.
    """
    platforms = {
        "windows": "dll",
        "linux": "so",
    }
    system = platform.system().lower()
    if system not in platforms.keys():
        raise NotImplementedError(f"Platform {system} is not supported.")

    package_root = Path(__file__).parent.parent
    model_path = package_root / "resources" / system / f"model.{platforms[system]}"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    return str(model_path)


class AeroInput(ct.Structure):
    _fields_ = [
        ("v0", ct.c_double),
        ("v1", ct.c_double),
        ("phi0", ct.c_double),
    ]


class AeroOutput(ct.Structure):
    _fields_ = [
        ("pitch", ct.c_double),
        ("velocity", ct.c_double),
        ("i0", ct.c_double),
        ("i1", ct.c_double),
        ("w0", ct.c_double),
        ("w1", ct.c_double),
    ]
