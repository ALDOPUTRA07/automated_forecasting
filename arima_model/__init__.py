import arima_model
from pathlib import Path

# Project Directories
PACKAGE_ROOT = Path(arima_model.__file__).resolve().parent

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()