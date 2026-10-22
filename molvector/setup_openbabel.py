#!/usr/bin/env python3
"""
Molvector OpenBabel setup helper
=================================
Detects a broken or missing OpenBabel installation and attempts to fix it
automatically.  Designed to be run once, either from the command line::

    molvector-setup

or directly::

    python -m molvector.setup_openbabel
"""

import os
import shutil
import subprocess
import sys
import textwrap


# ── helpers ──────────────────────────────────────────────────────────────────

def _run(cmd, **kw):
    """Run *cmd* silently; return (returncode, stdout+stderr)."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, **kw,
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except FileNotFoundError:
        return -1, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -2, "timed out"


def _probe():
    """Return True if ``from openbabel import openbabel`` + OBMol works."""
    code = (
        "from openbabel import openbabel as ob; "
        "m = ob.OBMol(); m.NewAtom(); print('OK')"
    )
    rc, out = _run([sys.executable, "-c", code])
    return rc == 0 and "OK" in out


def _detect_pkg_manager():
    """Return ('manager', [install_cmd_prefix]) or None."""
    if shutil.which("apt"):
        return ("apt", ["sudo", "apt-get", "install", "-y", "python3-openbabel"])
    if shutil.which("dnf"):
        return ("dnf", ["sudo", "dnf", "install", "-y", "python3-openbabel"])
    if shutil.which("yum"):
        return ("yum", ["sudo", "yum", "install", "-y", "python3-openbabel"])
    if shutil.which("pacman"):
        return ("pacman", ["sudo", "pacman", "-S", "--noconfirm", "openbabel", "swig"])
    if shutil.which("zypper"):
        return ("zypper", ["sudo", "zypper", "install", "-y", "python3-openbabel"])
    return None


def _pip_install():
    """Try (re)installing openbabel-wheel via pip."""
    print("  Trying: pip install --force-reinstall openbabel-wheel")
    rc, out = _run([sys.executable, "-m", "pip", "install",
                     "--force-reinstall", "openbabel-wheel"])
    if rc != 0:
        print(f"    pip failed (rc={rc})")
        return False
    return _probe()


def _system_install():
    """Try installing via the system package manager."""
    info = _detect_pkg_manager()
    if info is None:
        print("  No supported system package manager found (apt/dnf/pacman/zypper).")
        return False
    mgr, cmd = info
    print(f"  Trying: {' '.join(cmd)}")
    rc, out = _run(cmd)
    if rc != 0:
        print(f"    {mgr} failed (rc={rc})")
        if "permission" in out.lower() or "denied" in out.lower():
            print("    → You may need to run this script with sudo.")
        return False
    return _probe()


# ── main ─────────────────────────────────────────────────────────────────────

def _forget_probe_verdicts():
    """Molvector caches whether OpenBabel is safe to load; reset after changes."""
    try:
        from molvector import native_guard
        native_guard.clear_cache()
    except Exception:
        pass


def _manual_hints():
    """Install commands that fit this platform."""
    hints = ["pip install openbabel-wheel",
             "conda install -c conda-forge openbabel   (in a conda environment)"]
    if sys.platform.startswith("linux"):
        hints += ["sudo apt install python3-openbabel   (Debian/Ubuntu)",
                  "sudo dnf install python3-openbabel   (Fedora)",
                  "sudo pacman -S openbabel              (Arch)"]
    return hints


def setup():
    # Windows consoles with a legacy code page cannot print ✓ / — / …
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    _forget_probe_verdicts()
    print("=" * 60)
    print("  Molvector — OpenBabel setup")
    print("=" * 60)

    # 1. Already working?
    if _probe():
        print("\n  OpenBabel is already working. Nothing to do.")
        return True

    print("\n  OpenBabel is NOT working (missing or broken).")

    # 2. Try pip reinstall
    steps = 2 if sys.platform.startswith("linux") else 1
    print(f"\n[1/{steps}] Attempting pip install …")
    if _pip_install():
        print("\n  ✓ openbabel-wheel installed and verified.")
        return True

    # 3. Try system package (Linux only: macOS / Windows have no system
    #    package that provides the Python bindings)
    if sys.platform.startswith("linux"):
        print("\n[2/2] Attempting system package install …")
        if _system_install():
            print("\n  ✓ System OpenBabel installed and verified.")
            return True

    # 4. Give up gracefully
    print("\n" + "─" * 60)
    print("  Could not install OpenBabel automatically.")
    print()
    print("  Molvector will still work — just without OpenBabel features")
    print("  (geometry optimisation, dipole moments, InChI generation).")
    print()
    print("  To fix manually, try one of:")
    for hint in _manual_hints():
        print(f"    • {hint}")
    print("─" * 60)
    return False


if __name__ == "__main__":
    ok = setup()
    raise SystemExit(0 if ok else 1)
