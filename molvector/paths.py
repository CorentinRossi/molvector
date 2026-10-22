"""
Where Molvector keeps its per-user files (settings, session, logs, caches).

One folder per user, in the platform's usual place:

  Windows   %APPDATA%\\Molvector
  macOS     ~/Library/Application Support/Molvector
  Linux     $XDG_CONFIG_HOME/molvector  (default ~/.config/molvector)

``MOLVECTOR_CONFIG_DIR`` overrides it.  Earlier versions wrote settings into
the installed package folder (read-only for system / Program Files installs,
and wiped by every ``pip install --upgrade``); :func:`config_path` copies such
a legacy file over the first time it is asked for one that does not exist yet.

No Qt here: native_guard uses this before a QApplication exists.
"""

import json
import os
import shutil
import sys
import tempfile

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# file name -> earlier locations, most relevant first
_LEGACY = {
    "molvector_config.json": [os.path.join(_PKG_DIR, "molvector_config.json")],
    "last_molecule.json": [os.path.join(_PKG_DIR, "molvector_last_molecule.txt")],
    "scp_servers.json": [os.path.join(os.path.dirname(_PKG_DIR), ".molvector_scp_servers.json")],
    "color_table.json": [os.path.join(os.path.expanduser("~"), ".config", "molvector", "color_table.json")],
}


def user_dir() -> str:
    """The per-user Molvector folder (not created here)."""
    override = os.environ.get("MOLVECTOR_CONFIG_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
        return os.path.join(base, "Molvector")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support", "Molvector")
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    return os.path.join(base, "molvector")


def config_path(name: str) -> str:
    """Path of per-user file *name*, migrating it from a legacy location once."""
    path = os.path.join(user_dir(), name)
    if not os.path.exists(path):
        for old in _LEGACY.get(name, ()):
            if os.path.isfile(old) and os.path.abspath(old) != os.path.abspath(path):
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    shutil.copy2(old, path)
                except OSError:
                    pass
                break
    return path


def read_json(path: str, default=None):
    """Parsed JSON from *path*, or *default* if it is missing or unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path: str, data, **dump_kwargs) -> None:
    """Write *data* to *path* atomically (a crash never leaves half a file).

    Raises OSError if the file cannot be written.
    """
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, **dump_kwargs)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
