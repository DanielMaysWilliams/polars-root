from importlib.metadata import version

from polars_root.functions import (
    read_root,
    scan_root,
)

__version__ = version(__package__)

__all__ = [
    "read_root",
    "scan_root",
]
