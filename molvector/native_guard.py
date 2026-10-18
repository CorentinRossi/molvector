"""
Crash-safe loading of optional native libraries (OpenBabel, RDKit, cclib).

Why this exists
---------------
Wheels such as ``openbabel-wheel`` and ``rdkit`` ship their own copies of
native libraries (InChI, cairo, freetype, libxml2, boost, ...).  On Linux,
loading one of them into a process that already holds PyQt6 / SciPy / another
chemistry toolkit can SEGFAULT.  A segfault is a signal, not an exception:
``try/except`` cannot catch it and the whole GUI dies.

Strategy
--------
Before a library is imported into *this* process we import it in a
throw-away child process that reproduces the parent's native state as
faithfully as possible:

1. every compiled extension module already loaded in the parent is imported
   again, in the same order (PyQt6, numpy, scipy, rdkit, ...);
2. on Linux, if the parent runs a ``QApplication``, the child creates one too,
   on the same platform plugin (xcb / wayland), so Qt's system libraries are
   loaded as well;
3. the target is imported and a smoke test exercises the code paths
   Molvector uses (force fields, charge models, InChI, ...).

Only if the child exits cleanly is the library imported in the parent.
Results are cached in ``~/.molvector/native_probe.json`` and keyed by a
fingerprint of the environment (interpreter, library files, loaded
extensions), so the probe only runs again when something changes.

Rule for the rest of the code base: never import these libraries directly.
Go through :func:`load` (or :func:`ensure` followed by an import under
:func:`blocked`).

Environment variables
---------------------
``MOLVECTOR_DISABLE=openbabel,rdkit,cclib``  never load the listed libraries.
``MOLVECTOR_REPROBE=1``                      ignore the on-disk cache.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".molvector")
_CACHE_FILE = os.path.join(_CACHE_DIR, "native_probe.json")
_CACHE_MAX_ENTRIES = 64
_PROBE_TIMEOUT_S = 90
_CACHE_VERSION = 1


@dataclass(frozen=True)
class _Lib:
    package: str          # top-level package, used for "is it installed?"
    module: str           # module returned by load()
    import_code: str      # statement(s) that import the library
    smoke_code: str = ""  # exercises native code paths; Python errors are tolerated
    block: Tuple[str, ...] = ()  # packages hidden while importing (see blocked())


_METHANE_XYZ = (
    "5\\n\\nC 0 0 0\\nH 0.63 0.63 0.63\\nH -0.63 -0.63 0.63\\n"
    "H -0.63 0.63 -0.63\\nH 0.63 -0.63 -0.63\\n"
)

LIBS: Dict[str, _Lib] = {
    "openbabel": _Lib(
        package="openbabel",
        module="openbabel.openbabel",
        import_code="from openbabel import openbabel as ob",
        smoke_code=(
            "m = ob.OBMol()\n"
            "c = ob.OBConversion()\n"
            "c.SetInFormat('xyz')\n"
            f"c.ReadString(m, '{_METHANE_XYZ}')\n"
            "for name in ('MMFF94s', 'UFF'):\n"
            "    ff = ob.OBForceField.FindForceField(name)\n"
            "    if ff is not None and ff.Setup(m):\n"
            "        ff.ConjugateGradients(5)\n"
            "cm = ob.OBChargeModel.FindType('eem')\n"
            "if cm is not None:\n"
            "    cm.ComputeCharges(m)\n"
            "c.SetOutFormat('inchi')\n"
            "c.WriteString(m)\n"
        ),
    ),
    "rdkit": _Lib(
        package="rdkit",
        module="rdkit.Chem",
        import_code="from rdkit import Chem",
        smoke_code=(
            "m = Chem.MolFromSmiles('c1ccccc1O')\n"
            "Chem.MolToInchi(m)\n"
            "from rdkit.Chem import AllChem\n"
            "AllChem.Compute2DCoords(m)\n"
            "from rdkit.Chem import rdCoordGen\n"
            "rdCoordGen.AddCoords(Chem.RWMol(m))\n"
        ),
    ),
    # cclib eagerly imports optional "bridges" (openbabel, pyscf, ase, ...)
    # at import time.  We only need its parsers, so those are hidden.
    "cclib": _Lib(
        package="cclib",
        module="cclib",
        import_code="import cclib",
        smoke_code="import cclib.io",
        block=("openbabel", "pyscf", "ase", "Bio", "iodata", "psi4", "PyQuante"),
    ),
}

# name -> (ok, reason) decided during this session
_session: Dict[str, Tuple[bool, str]] = {}


# ── public API ───────────────────────────────────────────────────────────────

def is_installed(name: str) -> bool:
    """True if the library's package is on sys.path (does not import it)."""
    try:
        spec = importlib.util.find_spec(LIBS[name].package)
    except (ImportError, ValueError):
        return False
    return spec is not None


def ensure(name: str) -> bool:
    """Return True if library *name* can be imported here without crashing.

    Runs (or reuses) the out-of-process probe.  The verdict is remembered for
    the rest of the session.
    """
    if name in _session:
        return _session[name][0]
    ok, reason = _decide(name)
    _session[name] = (ok, reason)
    if not ok and reason.startswith("crashed"):
        print(f"[molvector] {name} disabled: {reason}", file=sys.stderr)
    return ok


def load(name: str):
    """Import and return the library's main module, or None if unsafe/missing."""
    if not ensure(name):
        return None
    lib = LIBS[name]
    try:
        with blocked(*lib.block):
            return importlib.import_module(lib.module)
    except Exception as exc:  # broken install that slipped past the probe
        _session[name] = (False, f"import failed: {exc!r}")
        return None


def failure_reason(name: str) -> Optional[str]:
    """Why *name* is unavailable (None if it loaded or was never requested)."""
    ok, reason = _session.get(name, (True, ""))
    return None if ok else reason


def crashed(name: str) -> bool:
    """True if *name* is installed but was disabled because it would crash."""
    return (failure_reason(name) or "").startswith("crashed")


@contextmanager
def blocked(*packages: str):
    """Hide *packages* from the import system for the duration of the block.

    ``sys.modules[name] = None`` makes ``import name`` raise ImportError and
    ``importlib.util.find_spec(name)`` return None.  Packages that are already
    imported are left untouched.
    """
    added = [p for p in packages if p not in sys.modules]
    for p in added:
        sys.modules[p] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        for p in added:
            if sys.modules.get(p, 0) is None:
                del sys.modules[p]


def clear_cache() -> None:
    """Forget every stored probe verdict (call after (re)installing a library)."""
    _session.clear()
    try:
        os.remove(_CACHE_FILE)
    except OSError:
        pass


def enable_crash_traceback() -> None:
    """Make a segfault print the Python stack instead of dying silently.

    Writes to stderr when there is one, else to ~/.molvector/crash.log
    (pythonw / desktop launchers have no usable stderr).
    """
    import faulthandler
    try:
        if sys.stderr is not None and sys.stderr.fileno() >= 0:
            faulthandler.enable(all_threads=True)
            return
    except Exception:
        pass
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        # Kept open for the life of the process: faulthandler writes to the fd.
        global _crash_log
        _crash_log = open(os.path.join(_CACHE_DIR, "crash.log"), "a")
        faulthandler.enable(file=_crash_log, all_threads=True)
    except Exception:
        pass


_crash_log = None


# ── decision logic ───────────────────────────────────────────────────────────

def _disabled_by_env(name: str) -> bool:
    raw = os.environ.get("MOLVECTOR_DISABLE", "")
    return name in {s.strip().lower() for s in raw.split(",") if s.strip()}


def _decide(name: str) -> Tuple[bool, str]:
    lib = LIBS[name]
    if _disabled_by_env(name):
        return False, "disabled by MOLVECTOR_DISABLE"
    if lib.module in sys.modules:
        return True, "already loaded"
    if not is_installed(name):
        return False, "not installed"
    if getattr(sys, "frozen", False):
        # Bundled app: sys.executable is not a Python interpreter, so no probe
        # is possible.  The bundle was tested as a whole when it was built.
        return True, "frozen build"

    preload = _loaded_extension_modules()
    qt_platform = _qt_platform()
    fingerprint = _fingerprint(name, preload, qt_platform)

    if os.environ.get("MOLVECTOR_REPROBE") != "1":
        cached = _cache_get(fingerprint)
        if cached is not None:
            return cached

    ok, reason, cacheable = _run_probe(lib, preload, qt_platform)
    if cacheable:
        _cache_put(fingerprint, name, ok, reason)
    return ok, reason


def _loaded_extension_modules() -> list:
    """Names of compiled extension modules loaded in this process, in load order."""
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    names = []
    for mod_name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if isinstance(f, str) and f.endswith(suffixes):
            names.append(mod_name)
    return names


def _qt_platform() -> Optional[str]:
    """Qt platform plugin in use (Linux only), or None.

    Creating a QApplication is what pulls in xcb / wayland / fontconfig / GL,
    so on Linux the child replays it.  On macOS a child QApplication would
    flash a Dock icon, and the Linux-style library clashes don't occur there.
    """
    if not sys.platform.startswith("linux"):
        return None
    qtw = sys.modules.get("PyQt6.QtWidgets")
    if qtw is None:
        return None
    try:
        app = qtw.QApplication.instance()
        return app.platformName() if app is not None else None
    except Exception:
        return None


def _file_stamp(path: Optional[str]) -> list:
    try:
        st = os.stat(path)  # type: ignore[arg-type]
        return [path, st.st_size, int(st.st_mtime)]
    except (OSError, TypeError):
        return [path]


def _fingerprint(name: str, preload: list, qt_platform: Optional[str]) -> str:
    lib = LIBS[name]
    try:
        origin = importlib.util.find_spec(lib.package).origin  # type: ignore[union-attr]
    except Exception:
        origin = None
    loaded = [_file_stamp(getattr(sys.modules.get(m), "__file__", None)) for m in preload]
    blob = json.dumps([
        _CACHE_VERSION, name, lib.import_code, lib.smoke_code, list(lib.block),
        sys.executable, sys.version, sys.platform, qt_platform,
        _file_stamp(origin), loaded,
    ], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# ── the child process ────────────────────────────────────────────────────────

_CHILD_TEMPLATE = r'''
import faulthandler, importlib, json, sys
faulthandler.enable()
cfg = json.loads(sys.argv[1])
for _name in cfg["preload"]:
    try:
        importlib.import_module(_name)
    except BaseException:
        pass
if cfg["qt_platform"]:
    from PyQt6.QtWidgets import QApplication
    _app = QApplication(["molvector-probe", "-platform", cfg["qt_platform"]])
for _name in cfg["block"]:
    if _name not in sys.modules:
        sys.modules[_name] = None
try:
    exec(cfg["import_code"])
except ImportError as _exc:
    print("MOLVECTOR_PROBE_IMPORT_ERROR", repr(_exc))
    sys.stdout.flush()
    sys.exit(3)
try:
    exec(cfg["smoke_code"])
except Exception as _exc:
    print("MOLVECTOR_PROBE_SMOKE_ERROR", repr(_exc))
print("MOLVECTOR_PROBE_OK")
sys.stdout.flush()
'''


def _run_probe(lib: _Lib, preload: list, qt_platform: Optional[str]) -> Tuple[bool, str, bool]:
    """Return (ok, reason, cacheable)."""
    cfg = json.dumps({
        "preload": preload,
        "qt_platform": qt_platform,
        "block": list(lib.block),
        "import_code": lib.import_code,
        "smoke_code": lib.smoke_code,
    })
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", _CHILD_TEMPLATE, cfg],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
            stdin=subprocess.DEVNULL, **kwargs,
        )
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {_PROBE_TIMEOUT_S}s", False
    except Exception as exc:
        return False, f"probe could not run: {exc!r}", False

    out = r.stdout or ""
    if r.returncode == 0 and "MOLVECTOR_PROBE_OK" in out:
        return True, "probe passed", True
    if "MOLVECTOR_PROBE_IMPORT_ERROR" in out:
        line = out.split("MOLVECTOR_PROBE_IMPORT_ERROR", 1)[1].strip().splitlines()[0]
        return False, f"import failed: {line}", False
    if r.returncode < 0:
        detail = f"signal {-r.returncode}"
    else:
        detail = f"exit code {r.returncode}"
    tail = (r.stderr or "").strip().splitlines()[-12:]
    reason = f"crashed in probe ({detail})"
    if tail:
        reason += "\n" + "\n".join(tail)
    return False, reason, True


# ── on-disk cache ────────────────────────────────────────────────────────────

def _cache_read() -> dict:
    try:
        with open(_CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _cache_get(fingerprint: str) -> Optional[Tuple[bool, str]]:
    entry = _cache_read().get(fingerprint)
    if not isinstance(entry, dict) or "ok" not in entry:
        return None
    return bool(entry["ok"]), str(entry.get("reason", "")) + " (cached)"


def _cache_put(fingerprint: str, name: str, ok: bool, reason: str) -> None:
    data = _cache_read()
    data[fingerprint] = {"name": name, "ok": ok, "reason": reason}
    if len(data) > _CACHE_MAX_ENTRIES:
        data = dict(list(data.items())[-_CACHE_MAX_ENTRIES:])
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, _CACHE_FILE)
    except Exception:
        pass
