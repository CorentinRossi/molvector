"""Molvector — Interactive 3D Molecule Viewer and Builder."""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("molvector")   # single source of truth: pyproject.toml
except PackageNotFoundError:              # running from a checkout that was never installed
    __version__ = "unknown"

from molvector.gui import main
