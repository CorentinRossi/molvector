"""
molvector_render.py — Renderer + Parsers + Force Field library
==============================================================
Ball-and-stick SVG renderer for molecules.

Parsers:
  parse_xyz(text, name)       — standard XYZ format
  parse_gaussian(text)        — Gaussian .gjf / .com input
  parse_gaussian_log(text)    — Gaussian .log / .out output (cclib-backed, legacy fallback)
  parse_pdb(text)             — standard PDB format
  parse_mol(text, name)       — MDL Molfile V2000 / V3000 format
  infer_bonds(mol)            — distance-threshold bond detection

Renderer:
  render_molecule(mol, ...)   — produces an SVG file

Writers:
  save_xyz(mol)               — returns XYZ string
  save_gaussian_input(mol)    — returns GJF string
  save_pdb(mol)               — returns PDB string
  save_mol(mol)               — returns MDL Molfile V2000 string

Force Field:
  optimize_geometry(mol, ...) — OpenBabel-backed MMFF94s/UFF geometry optimization

Properties:
  calculate_dipole_moment(mol, ...) — Gasteiger-charge dipole moment estimation
"""

import math, random, string, os, sys, tempfile
import numpy as np
import svgwrite
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import mol_strudel as strudel

try:
    import cclib
    _HAS_CCLIB = True
except ImportError:
    _HAS_CCLIB = False

# ── OpenBabel data directory setup (cross-platform) ───────────────────────────
# A broken openbabel-wheel installation (missing .so deps, incompatible
# glibc, etc.) can SEGFAULT during `import openbabel`.  Because segfaults
# are OS-level *signals*, not Python exceptions, a bare try/except cannot
# catch them — the interpreter is killed before the except clause runs.
#
# We probe the actual SWIG extension import in a short-lived subprocess first.
# If the child exits abnormally (signal / non-zero code) we treat OpenBabel
# as unavailable and carry on without it.
_HAS_OPENBABEL_PKG = False   # True if the openbabel *package* can be imported
_HAS_OPENBABEL_DATA = False  # True if UFF.prm was located
_OB_DATA_DIR = None          # path to the directory containing UFF.prm
openbabel = None             # type: ignore[assignment]

_OB_PROBE_CODE = (
    "import openbabel; from openbabel import openbabel;"
    " ob = openbabel.OBMol(); ob.NewAtom(); ob.DeleteAtom(ob.GetAtom(1))"
)


def _probe_openbabel() -> bool:
    """Return True if the OpenBabel SWIG extension loads without crashing.

    Tests ``from openbabel import openbabel`` and a minimal OBMol round-trip,
    which is the exact code path used by Molvector.  A broken shared library
    will SEGFAULT here (child process), not in the main process.
    """
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-c", _OB_PROBE_CODE],
            timeout=10,
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _try_setup_openbabel() -> bool:
    """Attempt to install / repair OpenBabel automatically.

    Runs ``molvector.setup_openabel`` in a subprocess so that any crash
    or ``sudo`` prompt stays isolated from the main process.  Returns
    True if the probe passes after the setup attempt.
    """
    import subprocess
    setup_mod = os.path.join(os.path.dirname(__file__), "setup_openbabel.py")
    if not os.path.isfile(setup_mod):
        return False
    try:
        subprocess.run(
            [sys.executable, setup_mod],
            timeout=180,
        )
    except Exception:
        pass
    return _probe_openbabel()


def _setup_openbabel_data() -> None:
    """Locate OpenBabel data files (UFF.prm etc.) on disk."""
    global _OB_DATA_DIR, _HAS_OPENBABEL_DATA
    _IS_WIN = sys.platform == 'win32'
    _OB_VERSIONS = ('3.1.1', '3.1.0', '3.0.0')

    _OB_CANDIDATES = []

    _ob_file = getattr(openbabel, '__file__', None) or ''
    _OB_PKG_DIR = os.path.dirname(_ob_file)
    _is_single_module = os.path.isfile(_ob_file) and not os.path.isdir(
        os.path.join(_OB_PKG_DIR, os.path.splitext(os.path.basename(_ob_file))[0])
    )

    _ENV_DATADIR = os.environ.get('BABEL_DATADIR')
    if _ENV_DATADIR and os.path.isfile(os.path.join(_ENV_DATADIR, 'UFF.prm')):
        _OB_CANDIDATES.append(_ENV_DATADIR)

    for ver in _OB_VERSIONS:
        _OB_CANDIDATES.append(os.path.join(_OB_PKG_DIR, 'share', 'openbabel', ver))
    _OB_CANDIDATES += [
        os.path.join(_OB_PKG_DIR, 'bin', 'data'),
        os.path.join(_OB_PKG_DIR, 'data'),
    ]
    if _is_single_module:
        _mod_stem = os.path.splitext(os.path.basename(_ob_file))[0]
        _mod_dir = os.path.join(_OB_PKG_DIR, _mod_stem)
        for ver in _OB_VERSIONS:
            _OB_CANDIDATES.append(os.path.join(_mod_dir, 'share', 'openbabel', ver))
        _OB_CANDIDATES += [
            os.path.join(_mod_dir, 'bin', 'data'),
            os.path.join(_mod_dir, 'data'),
        ]

    if _IS_WIN:
        _p = _OB_PKG_DIR
        for _ in range(3):
            _p = os.path.dirname(_p)
            _OB_CANDIDATES.append(os.path.join(_p, 'openbabel', 'data'))
            _OB_CANDIDATES.append(os.path.join(_p, 'data'))

    for _prefix in dict.fromkeys([sys.prefix, getattr(sys, 'base_prefix', sys.prefix)]):
        for ver in _OB_VERSIONS:
            _OB_CANDIDATES.append(os.path.join(_prefix, 'share', 'openbabel', ver))
            if _IS_WIN:
                _OB_CANDIDATES.append(os.path.join(_prefix, 'Library', 'share', 'openbabel', ver))
        _OB_CANDIDATES.append(os.path.join(_prefix, 'data'))

    _OB_PARENT = _OB_PKG_DIR
    for _ in range(6):
        _OB_PARENT = os.path.dirname(_OB_PARENT)
        for ver in _OB_VERSIONS:
            _OB_CANDIDATES.append(os.path.join(_OB_PARENT, 'share', 'openbabel', ver))
            if _IS_WIN:
                _OB_CANDIDATES.append(os.path.join(_OB_PARENT, 'Library', 'share', 'openbabel', ver))

    if _IS_WIN:
        for pf_var in ('PROGRAMFILES', 'PROGRAMFILES(X86)'):
            pf = os.environ.get(pf_var, '')
            if pf:
                for ver in _OB_VERSIONS:
                    _OB_CANDIDATES.append(os.path.join(pf, 'OpenBabel', ver, 'data'))
                    _OB_CANDIDATES.append(os.path.join(pf, 'OpenBabel', 'share', 'openbabel', ver))
        for data_var in ('APPDATA', 'PROGRAMDATA', 'LOCALAPPDATA'):
            d = os.environ.get(data_var, '')
            if d:
                for ver in _OB_VERSIONS:
                    _OB_CANDIDATES.append(os.path.join(d, 'openbabel', ver))
                    _OB_CANDIDATES.append(os.path.join(d, 'openbabel', 'share', 'openbabel', ver))
    else:
        for prefix in ('/opt/homebrew', '/usr/local', '/usr'):
            for ver in _OB_VERSIONS:
                _OB_CANDIDATES.append(os.path.join(prefix, 'share', 'openbabel', ver))

    for d in _OB_CANDIDATES:
        if d and os.path.isfile(os.path.join(d, 'UFF.prm')):
            _OB_DATA_DIR = d
            break
    if _OB_DATA_DIR:
        os.environ['BABEL_DATADIR'] = _OB_DATA_DIR
    _HAS_OPENBABEL_DATA = _OB_DATA_DIR is not None


if _probe_openbabel():
    try:
        import openbabel as _ob_pkg
        from openbabel import openbabel as _ob_swig
        # Touch the SWIG module so it's fully loaded before we proceed.
        _ob_swig.OBMol()
        openbabel = _ob_pkg
        _HAS_OPENBABEL_PKG = True
    except Exception:
        openbabel = None
else:
    # First probe failed — try automatic repair, then re-probe.
    if _try_setup_openbabel():
        try:
            import openbabel as _ob_pkg
            from openbabel import openbabel as _ob_swig
            _ob_swig.OBMol()
            openbabel = _ob_pkg
            _HAS_OPENBABEL_PKG = True
        except Exception:
            openbabel = None
    else:
        print(
            "[molvector] OpenBabel is not available.\n"
            "            Features requiring it (optimise, dipole, InChI) are disabled.\n"
            "            To enable, run:  molvector-setup",
            file=sys.stderr,
        )

if _HAS_OPENBABEL_PKG:
    try:
        _setup_openbabel_data()
    except Exception:
        _HAS_OPENBABEL_PKG = False
        _HAS_OPENBABEL_DATA = False
        openbabel = None

# Convenient aliases for the rest of the codebase
HAS_OPENBABEL = _HAS_OPENBABEL_PKG and _HAS_OPENBABEL_DATA


# ── Suppress OpenBabel stderr (incl. C-level fd 2 on Windows) ────────────────
from contextlib import contextmanager

@contextmanager
def _suppress_ob_stderr():
    """Context manager that silences both Python-level and C-level stderr.

    OpenBabel's SWIG bindings may write error messages directly to file
    descriptor 2, bypassing Python's ``sys.stderr``.  On Windows this is
    common.  We save/redirect the real fd 2 to ``os.devnull`` while also
    replacing ``sys.stderr``.
    """
    import os, sys, io
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    fd2 = None
    try:
        fd2 = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
    except Exception:
        fd2 = None
    try:
        yield
    finally:
        sys.stderr = old_stderr
        if fd2 is not None:
            try:
                os.dup2(fd2, 2)
                os.close(fd2)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Atom:
    element: str
    x: float
    y: float
    z: float
    color: Optional[str] = None

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Bond:
    i: int
    j: int
    order: int = 1


@dataclass
class ExcitedState:
    index: int
    energy_ev: float
    wavelength_nm: float
    oscillator_strength: float
    symmetry: str


@dataclass
class VibrationalMode:
    index: int
    frequency: float  # cm^-1
    intensity: float = 0.0 # IR Intensity
    displacements: np.ndarray = None # (N, 3)



@dataclass
class Molecule:
    name: str
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    charge: int = 0
    excited_states: List[ExcitedState] = field(default_factory=list)
    vibrational_modes: List[VibrationalMode] = field(default_factory=list)
    g16_nproc: Optional[int] = None
    g16_mem: Optional[str] = None
    g16_route: Optional[str] = None
    g16_chk: Optional[str] = None
    g16_dipole: Optional[Tuple[float, float, float, float]] = None
    g16_rotconst: Optional[Tuple[float, float, float]] = None
    g16_scf_energy: Optional[float] = None
    g16_opt_energy: Optional[float] = None
    g16_point_group: Optional[str] = None
    g16_zpve: Optional[float] = None
    g16_enthalpy: Optional[float] = None
    g16_entropy: Optional[float] = None
    g16_freeenergy: Optional[float] = None
    g16_temperature: Optional[float] = None



# ─────────────────────────────────────────────────────────────────────────────
# CHEMISTRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Standard atomic weights (g/mol), most common isotope / IUPAC 2021
ATOMIC_MASSES: Dict[str, float] = {
    "H":1.008,  "D":2.014,  "T":3.016,  "He":4.003, "Li":6.941, "Be":9.012, "B":10.811,
    "C":12.011, "13C":13.003, "14C":14.003, "N":14.007, "15N":15.000, "O":15.999, "F":18.998, "Ne":20.180,
    "Na":22.990,"Mg":24.305,"Al":26.982,"Si":28.086,"P":30.974,
    "S":32.065, "Cl":35.453,"Ar":39.948,"K":39.098, "Ca":40.078,
    "Fe":55.845,"Ni":58.693,"Cu":63.546,"Zn":65.38, "Br":79.904,
    "I":126.904,"Au":196.967,"Hg":200.592,
}

# Atomic number to element symbol (full periodic table, Z=1..118)
Z_TO_SYM: Dict[int, str] = {
     1:"H",  2:"He",  3:"Li",  4:"Be",  5:"B",  6:"C",  7:"N",  8:"O",
     9:"F", 10:"Ne", 11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P", 16:"S",
    17:"Cl",18:"Ar", 19:"K", 20:"Ca", 21:"Sc", 22:"Ti", 23:"V", 24:"Cr",
    25:"Mn",26:"Fe", 27:"Co", 28:"Ni", 29:"Cu", 30:"Zn", 31:"Ga", 32:"Ge",
    33:"As",34:"Se", 35:"Br", 36:"Kr", 37:"Rb", 38:"Sr", 39:"Y", 40:"Zr",
    41:"Nb",42:"Mo", 43:"Tc", 44:"Ru", 45:"Rh", 46:"Pd", 47:"Ag", 48:"Cd",
    49:"In",50:"Sn", 51:"Sb", 52:"Te", 53:"I", 54:"Xe", 55:"Cs", 56:"Ba",
    57:"La",58:"Ce", 59:"Pr", 60:"Nd", 61:"Pm", 62:"Sm", 63:"Eu", 64:"Gd",
    65:"Tb",66:"Dy", 67:"Ho", 68:"Er", 69:"Tm", 70:"Yb", 71:"Lu", 72:"Hf",
    73:"Ta",74:"W",  75:"Re", 76:"Os", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg",
    81:"Tl",82:"Pb", 83:"Bi", 84:"Po", 85:"At", 86:"Rn", 87:"Fr", 88:"Ra",
    89:"Ac",90:"Th", 91:"Pa", 92:"U",  93:"Np", 94:"Pu", 95:"Am", 96:"Cm",
    97:"Bk",98:"Cf", 99:"Es",100:"Fm",101:"Md",102:"No",103:"Lr",104:"Rf",
   105:"Db",106:"Sg",107:"Bh",108:"Hs",109:"Mt",110:"Ds",111:"Rg",112:"Cn",
   113:"Nh",114:"Fl",115:"Mc",116:"Lv",117:"Ts",118:"Og",
}

# Hill order: C first, H second, then alphabetical
_HILL_PRIORITY = {"C": 0, "H": 1}

def chemical_formula(mol: "Molecule") -> str:
    """Return Hill-order chemical formula, e.g. C60Ca."""
    from collections import Counter
    counts = Counter(a.element for a in mol.atoms)
    elems = sorted(counts.keys(),
                   key=lambda e: (_HILL_PRIORITY.get(e, 2), e))
    formula = "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in elems)
    return formula

def molecular_mass(mol: "Molecule") -> float:
    """Return the monoisotopic-approximate molecular mass in Da (g/mol)."""
    return sum(ATOMIC_MASSES.get(a.element, 0.0) for a in mol.atoms)

def calculate_rotational_constants(mol: "Molecule") -> Tuple[float, float, float]:
    """Return the rotational constants (A, B, C) in MHz for the molecule."""
    coords = []
    masses = []
    for atom in mol.atoms:
        coords.append((atom.x, atom.y, atom.z))
        masses.append(ATOMIC_MASSES[atom.element])

    coords = np.array(coords)
    masses = np.array(masses)

    coords_principal_axes = strudel.transform_to_principal_axes(coords, masses)
    moments_of_inertia, _ = strudel.diagonalize_I_tensor(coords_principal_axes, masses)
    rot_consts = strudel.moments_of_inertia_to_rotational_constants(moments_of_inertia)
    return rot_consts

# ─────────────────────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_xyz(text: str, name: str = "molecule") -> Molecule:
    """Standard XYZ: <N>\n<comment>\n<elem x y z> x N"""
    lines = [l.strip() for l in text.strip().splitlines()]
    n = int(lines[0])
    comment = lines[1] if len(lines) > 1 else ""
    
    # Try to extract charge from comment line (e.g. "charge=1" or "charge 1")
    import re
    charge = 0
    charge_match = re.search(r"charge\s*[:=]?\s*([+-]?\d+)", comment, re.IGNORECASE)
    if charge_match:
        try:
            charge = int(charge_match.group(1))
        except ValueError:
            pass

    atoms = []
    for line in lines[2:2 + n]:
        p = line.split()
        if len(p) >= 4:
            atoms.append(Atom(p[0], float(p[1]), float(p[2]), float(p[3])))
    
    if not atoms:
        raise ValueError("No valid atoms found in XYZ.")
        
    mol = Molecule(name=name, atoms=atoms, charge=charge)
    mol.name = chemical_formula(mol)
    return mol


def parse_pdb(text: str, name: str = "PDB Molecule") -> Molecule:
    """Basic PDB parser for ATOM/HETATM and CONECT records."""
    lines = text.splitlines()
    atoms = []
    bonds = []
    serial_to_idx = {}
    import re
    
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                # Fixed-width column parsing
                serial_str = line[6:11].strip()
                if not serial_str: continue
                serial = int(serial_str)
                
                atom_name = line[12:16].strip()
                x_str = line[30:38].strip()
                y_str = line[38:46].strip()
                z_str = line[46:54].strip()
                x, y, z = float(x_str), float(y_str), float(z_str)
                
                # Element symbol at 77-78, or infer from name
                elem = line[76:78].strip()
                if not elem:
                    elem = re.sub(r'[^a-zA-Z]', '', atom_name)
                    if len(elem) > 1 and elem[1].islower():
                        elem = elem[:2]
                    else:
                        elem = elem[:1]
                
                if len(elem) == 1: elem = elem.upper()
                else: elem = elem.capitalize()
                
                serial_to_idx[serial] = len(atoms)
                atoms.append(Atom(elem, x, y, z))
            except (ValueError, IndexError):
                continue
        
        elif line.startswith("CONECT"):
            try:
                # CONECT serial1 serial2 serial3...
                # Columns: 7-11, 12-16, 17-21, 22-26, 27-31
                parts = []
                for i in range(6, len(line), 5):
                    s = line[i:i+5].strip()
                    if s: parts.append(s)
                
                if not parts: continue
                src_serial = int(parts[0])
                if src_serial not in serial_to_idx: continue
                src_idx = serial_to_idx[src_serial]
                
                for target_str in parts[1:]:
                    target_serial = int(target_str)
                    if target_serial not in serial_to_idx: continue
                    target_idx = serial_to_idx[target_serial]
                    
                    if src_idx < target_idx:
                        bonds.append(Bond(src_idx, target_idx))
            except (ValueError, IndexError):
                continue

    if not atoms:
        raise ValueError("No valid ATOM or HETATM records found in PDB.")

    mol = Molecule(name=name, atoms=atoms, bonds=bonds)
    # Refine name from HEADER or TITLE
    for line in lines:
        if line.startswith("HEADER") and len(line) > 10:
            mol.name = line[10:50].strip() or name
            break
        elif line.startswith("TITLE ") and len(line) > 10:
            mol.name = line[10:70].strip() or name
            break
    
    return mol


def parse_gaussian(text: str) -> Molecule:
    """
    Gaussian .gjf / .com input file.
    Sections separated by blank lines:
      route  ->  title  ->  charge/mult + coords
    """
    lines = text.strip().splitlines()
    sections, cur = [], []
    for l in lines:
        if l.strip() == "":
            if cur:
                sections.append(cur)
                cur = []
        else:
            cur.append(l.strip())
    if cur:
        sections.append(cur)

    name = sections[1][0] if len(sections) > 1 else "molecule"
    mol = Molecule(name=name)

    # Extract header info from route section
    if sections:
        for line in sections[0]:
            low = line.strip().lower()
            if low.startswith("%nprocshared=") or low.startswith("%nproc="):
                try:
                    mol.g16_nproc = int(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
            elif low.startswith("%mem="):
                mol.g16_mem = line.split("=", 1)[1].strip()
            elif line.strip().startswith("#"):
                mol.g16_route = line.strip()

    if len(sections) >= 3:
        # First line of section 3 is "charge multiplicity"
        try:
            charge_mult = sections[2][0].split()
            mol.charge = int(charge_mult[0])
        except (IndexError, ValueError):
            pass
        for line in sections[2][1:]:
            p = line.split()
            if len(p) >= 4:
                try:
                    elem = p[0]
                    if elem.isdigit() or (elem.startswith('-') and elem[1:].isdigit()):
                        z = int(elem)
                        elem = Z_TO_SYM.get(z, f"X{z}")
                    mol.atoms.append(Atom(elem, float(p[1]), float(p[2]), float(p[3])))
                except ValueError:
                    pass
    
    if not mol.atoms:
        raise ValueError("No valid atoms found in Gaussian input.")

    # Override name with formula for consistency
    mol.name = chemical_formula(mol)
    return mol


def parse_mol(text: str, name: str = "Molecule") -> Molecule:
    """
    MDL Molfile V2000 / V3000 parser.
    Supports both V2000 and V3000 formats.
    """
    lines = text.splitlines()
    if len(lines) < 4:
        raise ValueError("Too few lines for a Molfile.")

    name = lines[0].strip() or name

    # Detect V3000: the "V3000" marker appears on the counts line (4th line),
    # but files may have blank lines. Check first 10 lines for the marker.
    is_v3000 = any("V3000" in line for line in lines[:10])

    if is_v3000:
        return _parse_mol_v3000(text, name)
    return _parse_mol_v2000(text, name)


def _parse_mol_v2000(text: str, name: str) -> Molecule:
    lines = text.splitlines()
    counts_line = ""
    counts_idx = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if s and "V2000" in s:
            counts_line = s
            counts_idx = i
            break
    if counts_idx < 0:
        raise ValueError("Could not find V2000 counts line in MOL file.")
    counts_parts = counts_line.split()
    if not counts_parts:
        raise ValueError("Could not parse counts line in MOL file.")
    n_atoms = int(counts_parts[0])
    n_bonds = int(counts_parts[1]) if len(counts_parts) > 1 else 0

    atom_lines = []
    bond_lines = []
    props_start = counts_idx + 1 + n_atoms + n_bonds

    for i in range(counts_idx + 1, counts_idx + 1 + n_atoms):
        if i < len(lines):
            atom_lines.append(lines[i])
    for i in range(counts_idx + 1 + n_atoms, counts_idx + 1 + n_atoms + n_bonds):
        if i < len(lines):
            bond_lines.append(lines[i])

    atoms = []
    for line in atom_lines:
        try:
            x = float(line[0:10].strip())
            y = float(line[10:20].strip())
            z = float(line[20:30].strip())
            elem = line[31:34].strip()
            if not elem:
                elem = line[30:33].strip()
            if not elem or not elem.isalpha():
                raise ValueError(f"Invalid element in atom line: {line}")
            elem = elem.capitalize()
            atoms.append(Atom(elem, x, y, z))
        except (ValueError, IndexError):
            continue

    if not atoms:
        raise ValueError("No valid atoms found in MOL file.")

    bonds = []
    seen_bonds = set()
    for line in bond_lines:
        try:
            parts = line.split()
            if len(parts) < 3:
                continue
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            order = int(parts[2])
            if order == 4:
                order = 5  # aromatic -> our resonance bond type
            if i < len(atoms) and j < len(atoms) and i != j:
                a, b = min(i, j), max(i, j)
                key = (a, b)
                if key not in seen_bonds:
                    seen_bonds.add(key)
                    bonds.append(Bond(a, b, order))
        except (ValueError, IndexError):
            continue

    # Parse properties block for charge
    import re
    mol_charge = 0
    for line in lines[props_start:]:
        s = line.strip()
        if s == "M  END":
            break
        if s.startswith("M  CHG"):
            parts = s.split()
            # M  CHG n a1 c1 a2 c2 ...
            try:
                n = int(parts[2])
                for k in range(n):
                    aidx = int(parts[3 + k * 2]) - 1
                    chg = int(parts[4 + k * 2])
                    mol_charge += chg
            except (ValueError, IndexError):
                pass

    mol = Molecule(name=name, atoms=atoms, bonds=bonds, charge=mol_charge)
    mol.name = chemical_formula(mol)
    return mol


def _parse_mol_v3000(text: str, name: str) -> Molecule:
    lines = text.splitlines()
    atoms, bonds = [], []
    seen_bonds = set()
    mol_charge = 0
    in_atom_block = False
    in_bond_block = False

    for line in lines:
        s = line.strip()
        if "BEGIN ATOM" in s:
            in_atom_block = True
            continue
        if "BEGIN BOND" in s:
            in_bond_block = True
            continue
        if "END ATOM" in s:
            in_atom_block = False
            continue
        if "END BOND" in s:
            in_bond_block = False
            continue

        if in_atom_block:
            try:
                parts = s.split()
                if len(parts) >= 7:
                    elem = parts[3]
                    x = float(parts[4])
                    y = float(parts[5])
                    z = float(parts[6])
                    atoms.append(Atom(elem, x, y, z))
            except (ValueError, IndexError):
                continue

        if in_bond_block:
            try:
                parts = s.split()
                if len(parts) >= 6:
                    order = int(parts[3])
                    i = int(parts[4]) - 1
                    j = int(parts[5]) - 1
                    if order == 4:
                        order = 5
                    if i < len(atoms) and j < len(atoms) and i != j:
                        a, b = min(i, j), max(i, j)
                        key = (a, b)
                        if key not in seen_bonds:
                            seen_bonds.add(key)
                            bonds.append(Bond(a, b, order))
            except (ValueError, IndexError):
                continue

        # Parse charge from V3000 properties
        if "M  CHG" in s or "M  V30" in s:
            import re
            chg_match = re.search(r"CHG\s+(\d+)\s+(-?\d+)", s)
            if chg_match:
                try:
                    aidx = int(chg_match.group(1)) - 1
                    chg = int(chg_match.group(2))
                    mol_charge += chg
                except (ValueError, IndexError):
                    pass

    if not atoms:
        raise ValueError("No valid atoms found in V3000 MOL file.")

    mol = Molecule(name=name, atoms=atoms, bonds=bonds, charge=mol_charge)
    mol.name = chemical_formula(mol)
    return mol


# ── cclib-backed Gaussian log parser ─────────────────────────────────────────

def _parse_gaussian_log_cclib(text: str) -> Molecule:
    """
    Parse a Gaussian .log / .out file using cclib.

    Falls through to the legacy parser if cclib is unavailable or fails.
    """
    if not _HAS_CCLIB:
        raise ImportError("cclib is not installed")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(text)
        tmp_path = f.name

    try:
        with _suppress_ob_stderr():
            data = cclib.io.ccread(tmp_path)
    finally:
        os.unlink(tmp_path)

    if data is None:
        raise ValueError("cclib could not parse this file")

    # ── Atoms ────────────────────────────────────────────────────────────
    atomcoords = np.array(data.atomcoords)
    atomnos = np.array(data.atomnos)
    atoms = []
    for i in range(len(atomnos)):
        z = int(atomnos[i])
        elem = Z_TO_SYM.get(z, f"X{z}")
        x, y, z_pos = atomcoords[-1][i]  # last geometry step
        atoms.append(Atom(elem, float(x), float(y), float(z_pos)))

    if not atoms:
        raise ValueError("cclib parsed no atoms from the file")

    # ── Charge & multiplicity ────────────────────────────────────────────
    charge = int(data.charge) if hasattr(data, "charge") else 0

    # ── Name from metadata or chemical formula ───────────────────────────
    name = "Gaussian output"
    if hasattr(data, "metadata") and data.metadata:
        comments = data.metadata.get("comments", None)
        if comments:
            name = comments.strip().splitlines()[0][:80]

    # ── Header info from raw text (cclib doesn't parse Link0 / route) ────
    nproc = None
    mem = None
    chk = None
    route_str = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("%nprocshared="):
            try:
                nproc = int(stripped.split("=", 1)[1])
            except ValueError:
                pass
        elif stripped.startswith("%nproc="):
            try:
                nproc = int(stripped.split("=", 1)[1])
            except ValueError:
                pass
        elif stripped.startswith("%mem="):
            mem = stripped.split("=", 1)[1]
        elif stripped.startswith("%chk="):
            chk = stripped.split("=", 1)[1]
        elif stripped.startswith("#"):
            route_parts = [stripped.lstrip("#").strip()]
            for j in range(i + 1, min(i + 30, len(lines))):
                s = lines[j].strip()
                if not s or s.startswith("-") or s.startswith("*"):
                    break
                if s.startswith("#"):
                    route_parts.append(s.lstrip("#").strip())
                elif s.startswith("%"):
                    break
                else:
                    if s and route_parts[-1] and (s[0].isupper() or s[0] in "(+-*"):
                        route_parts[-1] = route_parts[-1] + s
                    else:
                        route_parts[-1] = route_parts[-1] + " " + s
            route_str = "# " + " ".join(route_parts)
            break

    # ── SCF energy (eV → Hartree) ────────────────────────────────────────
    scf_energy = None
    if hasattr(data, "scfenergies") and data.scfenergies is not None and len(data.scfenergies) > 0:
        scf_energy = float(data.scfenergies[-1]) / 27.211386245988

    # ── Optimization energy (from metadata or SCF) ───────────────────────
    opt_energy = None
    # cclib doesn't separate opt energy; use last SCF for opt jobs
    if scf_energy is not None:
        opt_energy = scf_energy

    # ── Dipole moment (Debye) ────────────────────────────────────────────
    dipole_moment = None
    if hasattr(data, "moments") and data.moments is not None and len(data.moments) > 1:
        dip_vec = np.array(data.moments[1])
        if len(dip_vec) >= 3:
            dip_total = float(np.linalg.norm(dip_vec[:3]))
            dipole_moment = (float(dip_vec[0]), float(dip_vec[1]), float(dip_vec[2]), dip_total)

    # ── Rotational constants (MHz) ───────────────────────────────────────
    rot_constants = None
    if hasattr(data, "rotationalconstants") and data.rotationalconstants is not None:
        rc = np.array(data.rotationalconstants)
        if rc.ndim == 2 and len(rc) > 0:
            rc = rc[-1]  # last step
        if len(rc) >= 3:
            rot_constants = (float(rc[0]), float(rc[1]), float(rc[2]))

    # ── Vibrational modes ────────────────────────────────────────────────
    vibrational_modes = []
    if hasattr(data, "vibfreqs") and data.vibfreqs is not None:
        freqs = np.array(data.vibfreqs)
        ir_ints = np.array(data.vibirs) if hasattr(data, "vibirs") and data.vibirs is not None else np.zeros_like(freqs)
        disps = np.array(data.vibdisps) if hasattr(data, "vibdisps") and data.vibdisps is not None else None
        for i, freq in enumerate(freqs):
            disp = disps[i] if disps is not None and i < len(disps) else None
            vibrational_modes.append(VibrationalMode(
                index=i + 1,
                frequency=float(freq),
                intensity=float(ir_ints[i]) if i < len(ir_ints) else 0.0,
                displacements=disp,
            ))

    # ── Excited states (TD-DFT) ──────────────────────────────────────────
    excited_states = []
    if hasattr(data, "etenergies") and data.etenergies is not None:
        et_energies = np.array(data.etenergies)  # in cm^-1
        et_oscs = np.array(data.etoscs) if hasattr(data, "etoscs") and data.etoscs is not None else np.zeros_like(et_energies)
        et_syms = data.etsyms if hasattr(data, "etsyms") and data.etsyms is not None else ["" for _ in et_energies]
        for i, e_cm in enumerate(et_energies):
            e_ev = float(e_cm) * 1.2398419843320025e-4  # cm^-1 → eV
            wl_nm = 1e7 / float(e_cm) if float(e_cm) > 0 else 0.0  # cm^-1 → nm
            sym = et_syms[i] if i < len(et_syms) else ""
            excited_states.append(ExcitedState(
                index=i + 1,
                energy_ev=e_ev,
                wavelength_nm=wl_nm,
                oscillator_strength=float(et_oscs[i]) if i < len(et_oscs) else 0.0,
                symmetry=str(sym),
            ))

    # ── Thermochemistry ──────────────────────────────────────────────────
    zpve = float(data.zpve) if hasattr(data, "zpve") and data.zpve is not None else None
    enthalpy = float(data.enthalpy) if hasattr(data, "enthalpy") and data.enthalpy is not None else None
    entropy = float(data.entropy) if hasattr(data, "entropy") and data.entropy is not None else None
    freeenergy = float(data.freeenergy) if hasattr(data, "freeenergy") and data.freeenergy is not None else None
    temperature = float(data.temperature) if hasattr(data, "temperature") and data.temperature is not None else None

    # ── Point group (from metadata) ──────────────────────────────────────
    point_group = None
    if hasattr(data, "metadata") and data.metadata:
        sym = data.metadata.get("symmetry_detected", None)
        if sym:
            point_group = str(sym)

    # ── Build Molecule ───────────────────────────────────────────────────
    mol = Molecule(
        name=name,
        atoms=atoms,
        charge=charge,
        excited_states=excited_states,
        vibrational_modes=vibrational_modes,
    )
    mol.g16_nproc = nproc
    mol.g16_mem = mem
    mol.g16_dipole = dipole_moment
    mol.g16_rotconst = rot_constants
    mol.g16_scf_energy = scf_energy
    mol.g16_opt_energy = opt_energy
    mol.g16_point_group = point_group
    mol.g16_zpve = zpve
    mol.g16_enthalpy = enthalpy
    mol.g16_entropy = entropy
    mol.g16_freeenergy = freeenergy
    mol.g16_temperature = temperature
    if chk:
        mol.g16_chk = chk
    if route_str:
        mol.g16_route = route_str
    mol.name = chemical_formula(mol)
    return mol


def parse_gaussian_log(text: str) -> Molecule:
    """
    Gaussian .log / .out output file.

    Tries the cclib-backed parser first for robust parsing and extra data
    (thermochemistry, etc.).  Falls back to the legacy hand-written parser
    if cclib is unavailable or fails on this particular file.
    """
    if _HAS_CCLIB:
        try:
            return _parse_gaussian_log_cclib(text)
        except Exception:
            pass  # fall through to legacy
    return _parse_gaussian_log_legacy(text)


def _parse_gaussian_log_legacy(text: str) -> Molecule:
    """
    Gaussian .log / .out output file.

    Finds the LAST 'Standard orientation:' block (or 'Input orientation:'
    if standard is absent) — this is the final / optimised geometry.

    The coordinate table columns are:
      Center#  AtomicNum  AtomicType  X  Y  Z
    Atomic number is converted to element symbol via a built-in table.
    """
    lines = text.splitlines()

    # Extract header info (nproc, mem, chk, route)
    nproc = None
    mem = None
    chk = None
    route_str = None
    name = "Gaussian output"
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("%nprocshared="):
            try:
                nproc = int(stripped.split("=", 1)[1])
            except ValueError:
                pass
        elif stripped.startswith("%nproc="):
            try:
                nproc = int(stripped.split("=", 1)[1])
            except ValueError:
                pass
        elif stripped.startswith("%mem="):
            mem = stripped.split("=", 1)[1]
        elif stripped.startswith("%chk="):
            chk = stripped.split("=", 1)[1]
        elif stripped.startswith("#"):
            route_parts = [stripped.lstrip("#").strip()]
            for j in range(i + 1, min(i + 30, len(lines))):
                s = lines[j].strip()
                if not s or s.startswith("-") or s.startswith("*"):
                    break
                if s.startswith("#"):
                    route_parts.append(s.lstrip("#").strip())
                elif s.startswith("%"):
                    break
                else:
                    if s and route_parts[-1] and (s[0].isupper() or s[0] in "(+-*"):
                        route_parts[-1] = route_parts[-1] + s
                    else:
                        route_parts[-1] = route_parts[-1] + " " + s
            route_str = "# " + " ".join(route_parts)
            # Extract job title after route card
            for j in range(i + 1, min(i + 30, len(lines))):
                s = lines[j].strip()
                if s and not s.startswith("-") and not s.startswith("#") \
                        and not s.startswith("%") and not s.startswith("*") \
                        and len(s) > 2:
                    name = s
                    break
            break

    # Find all orientation block header positions
    MARKERS = ["Standard orientation:", "Input orientation:", "Z-Matrix orientation:"]
    block_starts = []
    for i, line in enumerate(lines):
        for marker in MARKERS:
            if marker in line:
                block_starts.append((i, marker))
                break

    if not block_starts:
        raise ValueError(
            "No orientation block found in this file.\n"
            "Make sure it is a Gaussian output (.log/.out) with geometry data."
        )

    # Prefer last Standard orientation; fall back to last of any kind
    std = [b for b in block_starts if "Standard" in b[1]]
    chosen_idx = (std or block_starts)[-1][0]

    # The table starts 5 lines after the marker:
    #  +0  "Standard orientation:"
    #  +1  "------..."
    #  +2  column header line 1
    #  +3  column header line 2
    #  +4  "------..."
    #  +5  first data row
    data_start = chosen_idx + 5

    atoms = []
    for line in lines[data_start:]:
        s = line.strip()
        if s.startswith("-"):
            break
        parts = s.split()
        if len(parts) >= 6:
            try:
                atomic_num = int(parts[1])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                elem = Z_TO_SYM.get(atomic_num, f"X{atomic_num}")
                atoms.append(Atom(elem, x, y, z))
            except ValueError:
                continue

    if not atoms:
        raise ValueError(
            "Could not parse atom coordinates from the orientation block.\n"
            "The file may be truncated or use an unexpected format."
        )

    # Extract charge from "Charge = X  Multiplicity = Y" line
    charge = 0
    for line in lines:
        if "Charge =" in line and "Multiplicity =" in line:
            try:
                charge = int(line.split("Charge =")[1].split()[0])
            except (IndexError, ValueError):
                pass
            break

    # Check if this is a SelectNormalModes output (non-standard freq format)
    import re
    has_select_modes = False
    for line in lines[:200]:
        if "SelectNormalModes" in line:
            has_select_modes = True
            break

    # Extract excited states (TDDFT)
    excited_states = []
    for line in lines:
        # Excited State   1:  2.002-?Sym    0.1463 eV 8473.13 nm  f=0.0000  <S**2>=0.752
        m = re.search(r"Excited State\s+(\d+):\s+(.+?)\s+([\d.]+) eV\s+([\d.]+) nm\s+f=([\d.]+)", line)
        if m:
            idx, sym, ev, nm, f = m.groups()
            excited_states.append(ExcitedState(int(idx), float(ev.strip()), float(nm.strip()), float(f.strip()), sym.strip()))

    # Extract vibrational modes
    vibrational_modes = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("Frequencies --"):
            # Handle cases with -- or ---
            after_marker = line.split("--", 1)[1].lstrip("-")
            freqs = [float(f) for f in after_marker.split()]
            n_freqs = len(freqs)
            
            # Find IR intensities if present
            intensities = [0.0] * n_freqs
            j = i + 1
            while j < i + 15 and j < len(lines):
                if "IR Inten    --" in lines[j]:
                    try:
                        after_marker_ir = lines[j].split("--", 1)[1].lstrip("-")
                        intensities = [float(x) for x in after_marker_ir.split()]
                    except ValueError:
                        pass
                    break
                j += 1

            # Find the start of the displacement table
            has_coord_atom = False
            while i < len(lines):
                if "  Atom  AN      X      Y      Z" in lines[i]:
                    has_coord_atom = False
                    break
                if " Coord Atom Element:" in lines[i]:
                    has_coord_atom = True
                    break
                i += 1
            i += 1 # skip header
            
            disps = [[] for _ in range(n_freqs)]
            if has_coord_atom:
                # Format: 3 lines per atom (X, then Y, then Z)
                for _ in range(len(atoms)):
                    # X line
                    parts_x = lines[i].split(); i += 1
                    # Y line
                    parts_y = lines[i].split(); i += 1
                    # Z line
                    parts_z = lines[i].split(); i += 1
                    
                    for f_idx in range(n_freqs):
                        dx = float(parts_x[3 + f_idx])
                        dy = float(parts_y[3 + f_idx])
                        dz = float(parts_z[3 + f_idx])
                        disps[f_idx].append([dx, dy, dz])
            else:
                # Standard format: 1 line per atom
                for _ in range(len(atoms)):
                    if i >= len(lines): break
                    parts = lines[i].split()
                    for f_idx in range(n_freqs):
                        dx = float(parts[2 + f_idx*3])
                        dy = float(parts[3 + f_idx*3])
                        dz = float(parts[4 + f_idx*3])
                        disps[f_idx].append([dx, dy, dz])
                    i += 1
            
            # Clear existing modes if we found a "fresh" set (index 1 to 3/5)
            # Gaussian usually restarts numbering for a new frequency block.
            if vibrational_modes and vibrational_modes[-1].index >= freqs[0]:
                vibrational_modes = []
            
            for f_idx in range(n_freqs):
                vibrational_modes.append(VibrationalMode(
                    index=len(vibrational_modes) + 1,
                    frequency=freqs[f_idx],
                    intensity=intensities[f_idx] if f_idx < len(intensities) else 0.0,
                    displacements=np.array(disps[f_idx])
                ))
            continue
        i += 1

    # ── SelectNormalModes fallback ──────────────────────────────────────
    if has_select_modes and not vibrational_modes:
        try:
            _parse_select_modes(text, atoms, vibrational_modes)
        except Exception:
            pass

    # ── Extract last-iteration results ──────────────────────────────────
    dipole_moment = None  # (X, Y, Z, Total) in Debye
    rot_constants = None  # (A, B, C) in MHz
    scf_energy = None     # in Hartree
    opt_energy = None     # from optimization ! line, in Hartree
    point_group = None
    for line in reversed(lines):
        stripped = line.strip()
        if dipole_moment is None and "Dipole moment (field-independent basis, Debye):" in line:
            continue
        if dipole_moment is None and "Dipole moment (Debye):" in line:
            continue
        if dipole_moment is None and "X=" in stripped and "Y=" in stripped and "Z=" in stripped and "Tot=" in stripped:
            try:
                parts = stripped.replace("=", " ").split()
                x = float(parts[parts.index("X") + 1])
                y = float(parts[parts.index("Y") + 1])
                z = float(parts[parts.index("Z") + 1])
                t = float(parts[parts.index("Tot") + 1])
                dipole_moment = (x, y, z, t)
            except (ValueError, IndexError):
                pass
        if rot_constants is None and "Rotational constants (MHz):" in stripped:
            try:
                vals = stripped.split(":")[1].strip().split()
                a, b, c = float(vals[0]), float(vals[1]), float(vals[2])
                rot_constants = (a, b, c)
            except (ValueError, IndexError):
                pass
        if scf_energy is None and stripped.startswith("SCF Done:"):
            try:
                e_part = stripped.split("=")[1].strip().split()[0]
                scf_energy = float(e_part)
            except (ValueError, IndexError):
                pass
        if opt_energy is None and stripped.startswith("!") and "E=" in stripped:
            try:
                after_E = stripped.split("E=")[1].strip().split()[0]
                opt_energy = float(after_E)
            except (ValueError, IndexError):
                pass
        if dipole_moment is not None and rot_constants is not None and scf_energy is not None and opt_energy is not None:
            break

    # Scan forward for point group (single occurrence, typically near start)
    for line in lines:
        if "Full point group" in line:
            try:
                pg = line.split("Full point group")[1].strip().split()[0]
                point_group = pg
            except (IndexError, ValueError):
                pass
            break

    mol = Molecule(name=name, atoms=atoms, charge=charge, 
                   excited_states=excited_states, 
                   vibrational_modes=vibrational_modes)
    mol.g16_nproc = nproc
    mol.g16_mem = mem
    mol.g16_dipole = dipole_moment
    mol.g16_rotconst = rot_constants
    mol.g16_scf_energy = scf_energy
    mol.g16_opt_energy = opt_energy
    mol.g16_point_group = point_group
    if chk:
        mol.g16_chk = chk
    if route_str:
        mol.g16_route = route_str
    mol.name = chemical_formula(mol)
    return mol


def _parse_select_modes(text: str, atoms: list, vibrational_modes: list):
    """
    Parse Gaussian log output produced with ``freq=(SelectNormalModes, SaveNormalModes)``.

    This keyword suppresses the usual ``Frequencies --`` lines and packs mode
    data into the route section.  Frequencies are recovered from the
    ``Vibrational temperatures`` (Kelvin) block.
    """
    import re
    import numpy as np

    n_atoms = len(atoms)
    n_vib = 3 * n_atoms - 6

    # ── 1.  Vibrational temperatures → frequencies (cm⁻¹) ──────────────
    K_TO_CM1 = 0.695028
    freqs = []
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if "Vibrational temperatures:" not in line:
            continue
        vals = []
        after = line.split(":", 1)[1].strip()
        if after:
            vals.extend(float(x) for x in after.split())
        for j in range(i + 1, min(i + 20, len(lines))):
            s = lines[j].strip()
            if not s:
                break
            for tok in s.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
            if len(vals) >= n_vib:
                break
        freqs = [v * K_TO_CM1 for v in vals[:n_vib]]
        break

    if not freqs:
        raise ValueError(
            "Could not parse frequencies from Vibrational temperatures."
        )

    # ── 2.  Route section: displacement vectors ────────────────────────
    disp_data = None
    start = text.find("NImag=0")
    if start >= 0:
        bs = text.find("\\\\", start)
        if bs >= 0:
            sep = text.find("\\\\", bs + 2)
            if sep >= 0:
                raw = text[bs + 2:sep]
                flat = re.sub(r"\s+", "", raw)
                parts = flat.split(",")
                vals = []
                for p in parts:
                    if p:
                        try:
                            vals.append(float(p))
                        except ValueError:
                            pass
                if len(vals) == 1485:
                    disp_data = np.array(vals).reshape(27, 55)[:, :54]

    # ── 3.  Build mode objects ─────────────────────────────────────────
    zero = np.zeros((n_atoms, 3))
    n_avail = len(disp_data) if disp_data is not None else 0

    for idx in range(len(freqs)):
        if idx < n_avail:
            displacements = disp_data[idx].reshape(n_atoms, 3)
        else:
            displacements = zero.copy()
        vibrational_modes.append(
            VibrationalMode(
                index=idx + 1,
                frequency=freqs[idx],
                intensity=0.0,
                displacements=displacements,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# BOND INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

COVALENT_RADII: Dict[str, float] = {
    "H":0.31,"D":0.31,"T":0.31,"C":0.76,"13C":0.76,"14C":0.76,"N":0.71,"15N":0.71,"O":0.66,"F":0.57,
    "S":1.05,"P":1.07,"Cl":1.02,"Br":1.20,"I":1.39,
    "B":0.82,"Si":1.11,
}

def infer_bonds(mol: Molecule, tol: float = 0.40) -> None:
    atoms = mol.atoms
    existing = {(b.i, b.j) for b in mol.bonds}

    new_bonds = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            ri = COVALENT_RADII.get(atoms[i].element, 0.77)
            rj = COVALENT_RADII.get(atoms[j].element, 0.77)
            dist = np.linalg.norm(atoms[i].pos - atoms[j].pos)
            if dist <= ri + rj + tol:
                r_single = ri + rj
                if dist <= r_single * 0.85:
                    order = 3
                elif dist <= r_single * 0.95:
                    order = 2
                else:
                    order = 1
                if (i, j) in existing:
                    for b in mol.bonds:
                        if b.i == i and b.j == j:
                            b.order = order
                            break
                else:
                    new_bonds.append(Bond(i, j, order))

    mol.bonds.extend(new_bonds)


# ─────────────────────────────────────────────────────────────────────────────
# CPK COLOURS
# ─────────────────────────────────────────────────────────────────────────────

CPK_BASE: Dict[str, str] = {
    "H":"#d4d4d4","D":"#fffec7","T":"#fffeab","C":"#444444","13C":"#2f494a","14C":"#2f494a","N":"#2050d0","15N":"#232f8b","O":"#cc1111",
    "F":"#66cc22","S":"#ddcc00","P":"#ff8800","Cl":"#11bb22",
    "Br":"#882200","I":"#660099","B":"#ffaa33","Si":"#8888aa",
    "He":"#e0fefe","Ne":"#c2e9ee","Ar":"#a4d4de","Kr":"#86bfce","Xe":"#68aabe","Rn":"#4a95ae","Og":"#2c809e",
    "Al":"#abad98","Ga":"#abad98","In":"#abad98","Sn":"#abad98","Tl":"#abad98","Pb":"#abad98","Bi":"#abad98","Po":"#abad98","Nh":"#abad98","Fl":"#abad98","Mc":"#abad98","Lv":"#abad98",
    "La":"#e91e63","Ce":"#e91e63","Pr":"#e91e63","Nd":"#e91e63","Pm":"#e91e63","Sm":"#e91e63","Eu":"#e91e63","Gd":"#e91e63","Tb":"#e91e63","Dy":"#e91e63","Ho":"#e91e63","Er":"#e91e63","Tm":"#e91e63","Yb":"#e91e63","Lu":"#e91e63",
    "Ac":"#d32f2f","Th":"#d32f2f","Pa":"#d32f2f","U":"#d32f2f","Np":"#d32f2f","Pu":"#d32f2f","Am":"#d32f2f","Cm":"#d32f2f","Bk":"#d32f2f","Cf":"#d32f2f","Es":"#d32f2f","Fm":"#d32f2f","Md":"#d32f2f","No":"#d32f2f","Lr":"#d32f2f",
}

CPK_DARK: Dict[str, str] = {
    "H":"#888888","D":"#595845","T":"#59583b","C":"#111111","13C":"#101919","14C":"#101919","N":"#0a1a60","15N":"#0c1030","O":"#550000",
    "F":"#224400","S":"#554400","P":"#441100","Cl":"#003300",
    "Br":"#330000","I":"#220033","B":"#664400","Si":"#333344",
    "He":"#4e5858","Ne":"#435153","Ar":"#394a4d","Kr":"#2e4248","Xe":"#243b42","Rn":"#19343c","Og":"#0f2c37",
    "Al":"#3b3c35","Ga":"#3b3c35","In":"#3b3c35","Sn":"#3b3c35","Tl":"#3b3c35","Pb":"#3b3c35","Bi":"#3b3c35","Po":"#3b3c35","Nh":"#3b3c35","Fl":"#3b3c35","Mc":"#3b3c35","Lv":"#3b3c35",
    "La":"#510a22","Ce":"#510a22","Pr":"#510a22","Nd":"#510a22","Pm":"#510a22","Sm":"#510a22","Eu":"#510a22","Gd":"#510a22","Tb":"#510a22","Dy":"#510a22","Ho":"#510a22","Er":"#510a22","Tm":"#510a22","Yb":"#510a22","Lu":"#510a22",
    "Ac":"#491010","Th":"#491010","Pa":"#491010","U":"#491010","Np":"#491010","Pu":"#491010","Am":"#491010","Cm":"#491010","Bk":"#491010","Cf":"#491010","Es":"#491010","Fm":"#491010","Md":"#491010","No":"#491010","Lr":"#491010",
}

DEFAULT_BASE = "#cc44aa"
DEFAULT_DARK = "#440022"

VDW_RADII: Dict[str, float] = {
    "H":0.53,"D":0.53,"T":0.53,"C":0.77,"13C":0.77,"14C":0.77,"N":0.75,"15N":0.75,"O":0.73,"F":0.71,
    "S":1.02,"P":1.06,"Cl":0.99,"Br":1.14,"I":1.33,
    "B":0.87,"Si":1.10,
}
DEFAULT_VDW = 0.80


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h = h.lstrip("#")
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

def rgb_to_hex(r, g, b) -> str:
    return "#{:02x}{:02x}{:02x}".format(max(0,min(255,int(r))), max(0,min(255,int(g))), max(0,min(255,int(b))))

def lighten(hex_color: str, factor: float) -> str:
    r,g,b = hex_to_rgb(hex_color)
    return rgb_to_hex(r+(255-r)*factor, g+(255-g)*factor, b+(255-b)*factor)

def darken(hex_color: str, factor: float) -> str:
    r,g,b = hex_to_rgb(hex_color)
    return rgb_to_hex(r*(1-factor), g*(1-factor), b*(1-factor))

def auto_dark(base: str) -> str:
    return darken(base, 0.65)

def interpolate_color(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return rgb_to_hex(r1 + (r2-r1)*t, g1 + (g2-g1)*t, b1 + (b2-b1)*t)

def compute_gradient_stops(
    base: str, dark: str, li: float,
    roughness: float = 1.0,
    edge_fn=None,
):
    if edge_fn is None:
        edge_fn = lambda b, d, l: darken(b, 0.65 * l)
    r = max(0.0, roughness)
    spread = 0.3 + r * 0.7
    hi = max(0.0, 1.0 - r * 0.4)
    ef = max(0.0, 1.0 - r * 0.35)
    p1 = min(0.18 * spread, 0.35)
    p2 = min(0.48 * spread, 0.65)
    p3 = min(0.78 * spread, 0.85)
    return [
        (0.00,  lighten(base, 0.70 * li * hi)),
        (p1,    lighten(base, 0.40 * li * hi)),
        (p2,    lighten(base, 0.15 * li * hi)),
        (p3,    base),
        (1.00,  edge_fn(base, dark, li * ef)),
    ]

# Light position presets: maps name -> (gradient_cx, gradient_cy, Lx, Ly, Lz)
LIGHT_POSITIONS = {
    "top-left":      ("0.33", "0.28",  -0.34, -0.44,  0.83),
    "top":           ("0.50", "0.10",   0.00, -0.50,  0.87),
    "top-right":     ("0.67", "0.28",   0.34, -0.44,  0.83),
    "left":          ("0.10", "0.50",  -0.50,  0.00,  0.87),
    "center":        ("0.50", "0.50",   0.00,  0.00,  1.00),
    "right":         ("0.90", "0.50",   0.50,  0.00,  0.87),
    "bottom-left":   ("0.33", "0.72",  -0.34,  0.44,  0.83),
    "bottom":        ("0.50", "0.90",   0.00,  0.50,  0.87),
    "bottom-right":  ("0.67", "0.72",   0.34,  0.44,  0.83),
}

# ─────────────────────────────────────────────────────────────────────────────
# WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def save_xyz(mol: "Molecule") -> str:
    """Produce standard XYZ string."""
    lines = [str(len(mol.atoms)), mol.name]
    for a in mol.atoms:
        lines.append(f"{a.element:3s} {a.x:12.6f} {a.y:12.6f} {a.z:12.6f}")
    return "\n".join(lines)

def save_gaussian_input(mol: "Molecule") -> str:
    """Produce a basic Gaussian input (.gjf) string."""
    lines = [
        "%nprocshared=4",
        "%mem=4GB",
        f"# p opt freq b3lyp/6-31g(d)",
        "",
        mol.name,
        "",
        f"{mol.charge} 1"
    ]
    for a in mol.atoms:
        lines.append(f"{a.element:3s} {a.x:12.6f} {a.y:12.6f} {a.z:12.6f}")
    lines.append("")  # Gaussian needs trailing blank line
    return "\n".join(lines)

def save_pdb(mol: "Molecule") -> str:
    """Produce a basic PDB string."""
    lines = [f"HEADER    {mol.name[:40]:<40}", f"TITLE     {mol.name}"]
    for i, a in enumerate(mol.atoms):
        serial = (i + 1) % 100000
        line = f"HETATM{serial:5d} {a.element:<3s}  UNL A   1    {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  1.00  0.00          {a.element:>2s}"
        lines.append(line)
    
    for b in mol.bonds:
        lines.append(f"CONECT{b.i+1:5d}{b.j+1:5d}")
        
    lines.append("END")
    return "\n".join(lines)

def save_mol(mol: "Molecule") -> str:
    """Produce an MDL Molfile V2000 string."""
    lines = [
        mol.name[:80],
        "  Molvector",
        "",
        f" {len(mol.atoms):3d}{len(mol.bonds):3d}  0  0  0  0  0  0  0  0999 V2000",
    ]
    for a in mol.atoms:
        lines.append(
            f"{a.x:10.4f}{a.y:10.4f}{a.z:10.4f} {a.element:<3s} 0  0  0  0  0  0  0  0  0  0  0  0"
        )
    for b in mol.bonds:
        bo = b.order if b.order in (1, 2, 3) else 1
        lines.append(f"{b.i+1:3d}{b.j+1:3d}{bo:3d}  0  0  0  0")
    if mol.charge != 0:
        # Find atoms with non-zero formal charge
        # Approximate by evenly distributing total charge
        chg_parts = []
        if len(mol.atoms) > 0:
            q = mol.charge // len(mol.atoms)
            r = mol.charge % len(mol.atoms)
            for i in range(len(mol.atoms)):
                c = q + (1 if i < r else 0)
                if c != 0:
                    chg_parts.append(f"{i+1:4d}{c:4d}")
            if chg_parts:
                # Must write in pairs; M  CHG n a1 c1 a2 c2 ...
                # Group in chunks of 8 pairs to fit line length
                n_per_line = 8
                for chunk_start in range(0, len(chg_parts), n_per_line):
                    chunk = chg_parts[chunk_start:chunk_start + n_per_line]
                    n = len(chunk)
                    lines.append("M  CHG" + f"{n:3d}" + "".join(chunk))
    lines.append("M  END")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx,sx = math.cos(rx),math.sin(rx)
    cy,sy = math.cos(ry),math.sin(ry)
    cz,sz = math.cos(rz),math.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx
def center_positions(atoms: List[Atom]) -> np.ndarray:
    if not atoms:
        return np.zeros((0, 3))
    pos = np.array([a.pos for a in atoms])
    return pos - pos.mean(axis=0)

def optimize_geometry(mol: Molecule, max_steps: int = 500, tol: float = 0.01,
                      fixed_indices: List[int] = None, k_bond: float = None, k_rep: float = None):
    """
    Geometry optimization using OpenBabel's MMFF94s or UFF force field.
    
    Converts the Molvector molecule to OpenBabel's internal representation,
    runs conjugate-gradient minimization, and writes the optimized coordinates
    back.  Automatically picks the best available force field:
      MMFF94s  →  best for organic elements (H,C,N,O,F,S,P,Cl,Br,I)
      UFF      →  fallback, covers the full periodic table
    
    Parameters
    ----------
    mol : Molecule
    max_steps : int
        Maximum conjugate-gradient iterations (default 500; ~0.3-0.7 ms total).
    tol : float
        Convergence threshold (kept for backward compatibility; OpenBabel uses
        its own internal convergence criteria).
    fixed_indices : list of int, optional
        Atom indices (0-based) that should remain frozen.
    k_bond, k_rep : ignored
        Legacy parameters accepted for backward-compatible call sites.
    
    Returns
    -------
    int
        Number of iterations taken (approximate).
    """
    if not mol.atoms or not HAS_OPENBABEL:
        return 0
    if len(mol.atoms) < 2:
        return 0

    import os
    from openbabel import openbabel as ob

    n = len(mol.atoms)

    # ── 1. Element symbol → atomic number ─────────────────────────────────
    _SYM_TO_Z = {
        "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9,
        "Ne":10, "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16,
        "Cl":17, "Ar":18, "K":19, "Ca":20, "Fe":26, "Ni":28, "Cu":29,
        "Zn":30, "Br":35, "I":53, "Au":79, "Hg":80,
    }

    # ── 2. Build OBMol ────────────────────────────────────────────────────
    obmol = ob.OBMol()
    obmol.SetDimension(3)

    for a in mol.atoms:
        oba = obmol.NewAtom()
        oba.SetAtomicNum(_SYM_TO_Z.get(a.element, 6))
        oba.SetVector(a.x, a.y, a.z)

    # Bond-order translation map
    # Molvector uses 5 for aromatic/resonance bonds.
    # For MMFF94s we convert to alternating 1/2 (Kekulé pattern) which OB
    # will auto-detect as aromatic during setup.
    _kekule_prepass(mol, obmol)

    # Translate remaining bonds (order 1/2/3 directly)
    for b in mol.bonds:
        if b.order in (1, 2, 3):
            obmol.AddBond(int(b.i) + 1, int(b.j) + 1, int(b.order))

    obmol.SetTotalCharge(mol.charge)

    # ── 3. Select and set up force field ──────────────────────────────────
    ff = None
    ff_name = "MMFF94s"
    with _suppress_ob_stderr():
        ff = ob.OBForceField.FindForceField(ff_name)
        if not ff.Setup(obmol):
            ff = ob.OBForceField.FindForceField("UFF")
            ff_name = "UFF"
            if not ff.Setup(obmol):
                raise RuntimeError(
                    "Force field setup failed.  UFF.prm was found at\n"
                    f"  {os.environ.get('BABEL_DATADIR', '?')}\n"
                    "but the force field could not initialise for this molecule.\n"
                    "Try:  pip install --force-reinstall openbabel-wheel"
                )

        if fixed_indices:
            for idx in fixed_indices:
                ff.SetFixAtom(int(idx) + 1)  # OB is 1-indexed

        # ── 4. Optimize ───────────────────────────────────────────────────
        ff.ConjugateGradients(int(max_steps))

    # ── 5. Read back coordinates ──────────────────────────────────────────
    ff.GetCoordinates(obmol)
    for i in range(n):
        oba = obmol.GetAtom(i + 1)
        mol.atoms[i].x = oba.GetX()
        mol.atoms[i].y = oba.GetY()
        mol.atoms[i].z = oba.GetZ()

    return int(max_steps)


def calculate_dipole_moment(mol: Molecule):
    """
    Estimate the dipole moment using EEM partial charges via OpenBabel.

    The Electronegativity Equalization Method (EEM) distributes partial
    charges so that each atom reaches a common electronegativity, then the
    dipole moment is computed relative to the centre of mass.

    Returns
    -------
    tuple (dipole_vector, dipole_magnitude, atom_charges)
        dipole_vector   : np.ndarray of shape (3,) — (μx, μy, μz) in Debye
        dipole_magnitude: float — |μ| in Debye
        atom_charges    : list of (element, charge) tuples

    Returns (None, None, None) when OpenBabel is unavailable or the
    molecule is empty.
    """
    if not mol.atoms or not HAS_OPENBABEL:
        return None, None, None
    if len(mol.atoms) < 2:
        return None, None, None

    import os
    from openbabel import openbabel as ob

    _SYM_TO_Z = {
        "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9,
        "Ne":10, "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16,
        "Cl":17, "Ar":18, "K":19, "Ca":20, "Fe":26, "Ni":28, "Cu":29,
        "Zn":30, "Br":35, "I":53, "Au":79, "Hg":80,
    }

    n = len(mol.atoms)

    # ── 1. Build OBMol ────────────────────────────────────────────────────
    obmol = ob.OBMol()
    obmol.SetDimension(3)

    for a in mol.atoms:
        oba = obmol.NewAtom()
        oba.SetAtomicNum(_SYM_TO_Z.get(a.element, 6))
        oba.SetVector(a.x, a.y, a.z)

    _kekule_prepass(mol, obmol)

    for b in mol.bonds:
        if b.order in (1, 2, 3):
            obmol.AddBond(int(b.i) + 1, int(b.j) + 1, int(b.order))

    if obmol.NumBonds() == 0:
        obmol.ConnectTheDots()

    obmol.SetTotalCharge(mol.charge)

    # ── 2. Compute EEM partial charges ────────────────────────────────────
    with _suppress_ob_stderr():
        charge_model = ob.OBChargeModel.FindType("eem")
        if charge_model is None:
            return None, None, None
        if not charge_model.ComputeCharges(obmol):
            return None, None, None

    # ── 3. Extract charges and centre of mass ────────────────────────────
    charges = []
    com = np.zeros(3)
    total_mass = 0.0
    for i in range(n):
        oba = obmol.GetAtom(i + 1)
        q = oba.GetPartialCharge()
        elem = mol.atoms[i].element
        charges.append((elem, q))
        mass = ATOMIC_MASSES.get(elem, 0.0)
        com += mass * np.array([mol.atoms[i].x, mol.atoms[i].y, mol.atoms[i].z])
        total_mass += mass
    if total_mass > 0:
        com /= total_mass

    # ── 4. Dipole moment vector (Debye) ──────────────────────────────────
    DEBYE_PER_EANGSTROM = 4.803204
    mu = np.zeros(3)
    for i in range(n):
        r = np.array([mol.atoms[i].x, mol.atoms[i].y, mol.atoms[i].z]) - com
        mu += charges[i][1] * r
    mu *= DEBYE_PER_EANGSTROM

    return mu, float(np.linalg.norm(mu)), charges


def generate_inchi(mol: Molecule) -> Optional[str]:
    """
    Generate an InChI identifier for *mol* using RDKit (preferred) or
    OpenBabel (fallback).

    Returns the InChI string on success, or ``None`` if conversion fails.
    """
    if not mol.atoms:
        return None

    try:
        return _generate_inchi_rdkit(mol)
    except Exception:
        pass

    if HAS_OPENBABEL:
        try:
            return _generate_inchi_openbabel(mol)
        except Exception:
            pass

    return None


_EXPECTED_VALENCE: Dict[str, int] = {
    "H": 1, "D": 1, "T": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4, "13C": 4, "14C": 4, "N": 4, "15N": 4, "O": 2,
    "F": 1, "Ne": 8, "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6,
    "Cl": 1, "Ar": 8, "K": 1, "Ca": 2, "Fe": 3, "Ni": 2, "Cu": 2,
    "Zn": 2, "Br": 1, "I": 1, "Au": 3, "Hg": 2,
}


def check_valence_issues(mol: Molecule) -> List[str]:
    """Return a list of valence warning strings for atoms in *mol*."""
    issues: List[str] = []

    charge_idx = -1
    if mol.charge != 0:
        charge_idx = next(
            (i for i, a in enumerate(mol.atoms) if a.element != "H"), 0
        )

    for idx, a in enumerate(mol.atoms):
        bond_sum = 0.0
        for b in mol.bonds:
            if b.i == idx or b.j == idx:
                bond_sum += b.order
        expected = _EXPECTED_VALENCE.get(a.element, 4)
        if idx == charge_idx:
            expected -= mol.charge
        if bond_sum > expected:
            issues.append(
                f"{a.element}#{idx + 1}: valence {int(bond_sum)} exceeds expected {expected}"
            )
        elif bond_sum < expected:
            issues.append(
                f"{a.element}#{idx + 1}: valence {int(bond_sum)} is below expected {expected}"
            )
    return issues


def _generate_inchi_rdkit(mol: Molecule) -> Optional[str]:
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    kek = _kekule_orders(mol)

    rw = Chem.RWMol()
    for i, a in enumerate(mol.atoms):
        rw.AddAtom(Chem.Atom(a.element))
        rw.GetAtomWithIdx(i).SetNumImplicitHs(0)
    for b in mol.bonds:
        if b.order == 5:
            k = (b.i, b.j) if b.i < b.j else (b.j, b.i)
            order = kek.get(k, 1)
        else:
            order = b.order
        btype = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                 3: Chem.BondType.TRIPLE}.get(order, Chem.BondType.SINGLE)
        rw.AddBond(b.i, b.j, btype)

    if mol.charge != 0:
        heavy = next((i for i, a in enumerate(mol.atoms) if a.element != "H"), 0)
        rw.GetAtomWithIdx(heavy).SetFormalCharge(mol.charge)

    conf = Chem.Conformer(len(mol.atoms))
    for i, a in enumerate(mol.atoms):
        conf.SetAtomPosition(i, Point3D(a.x, a.y, a.z))
    rw.AddConformer(conf, assignId=True)

    inchi = Chem.MolToInchi(rw)
    if inchi:
        return inchi
    return None


def _generate_inchi_openbabel(mol: Molecule) -> Optional[str]:
    import os

    from openbabel import openbabel as ob

    _SYM_TO_Z = {
        "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9,
        "Ne":10, "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16,
        "Cl":17, "Ar":18, "K":19, "Ca":20, "Fe":26, "Ni":28, "Cu":29,
        "Zn":30, "Br":35, "I":53, "Au":79, "Hg":80,
    }

    obmol = ob.OBMol()
    obmol.SetDimension(3)

    heavy_idx = -1
    for idx, a in enumerate(mol.atoms):
        oba = obmol.NewAtom()
        oba.SetAtomicNum(_SYM_TO_Z.get(a.element, 6))
        oba.SetVector(a.x, a.y, a.z)
        if heavy_idx < 0 and a.element != "H":
            heavy_idx = idx

    _kekule_prepass(mol, obmol)

    for b in mol.bonds:
        if b.order == 5:
            continue
        obmol.AddBond(b.i + 1, b.j + 1, b.order)

    if mol.charge != 0 and heavy_idx >= 0:
        obmol.GetAtom(heavy_idx + 1).SetFormalCharge(mol.charge)

    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()

    conv = ob.OBConversion()
    conv.SetOutFormat("inchi")
    with _suppress_ob_stderr():
        inchi = conv.WriteString(obmol).strip()
    return inchi if inchi else None


def _kekule_orders(mol: "Molecule") -> Dict[Tuple[int, int], int]:
    """
    Return a Kekulé bond-order map for resonance (order-5) bonds.

    For each connected component of resonance bonds:
      * **Simple cycle** (every node degree 2): walk around assigning
        alternating bond orders (2, 1, 2, 1, …).
      * **Otherwise** (fused rings, branched): fall back to all-single bonds.

    Returns a dict ``{(i, j): bond_order}`` where ``i < j``.
    """
    res_bonds = [(b.i, b.j) for b in mol.bonds if b.order == 5]
    if not res_bonds:
        return {}

    adj: Dict[int, set] = {}
    for i, j in res_bonds:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    def key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    orders: Dict[Tuple[int, int], int] = {}

    visited: set = set()
    for start in adj:
        if start in visited:
            continue

        comp_nodes: List[int] = []
        q = [start]
        visited.add(start)
        while q:
            node = q.pop(0)
            comp_nodes.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        is_simple_cycle = all(len(adj[n]) == 2 for n in comp_nodes) and len(comp_nodes) >= 3

        if is_simple_cycle:
            cycle: List[int] = []
            prev = -1
            cur = comp_nodes[0]
            while True:
                cycle.append(cur)
                nbs = [nb for nb in adj[cur] if nb != prev]
                if not nbs:
                    break
                nxt = nbs[0]
                if len(cycle) > 1 and nxt == cycle[0]:
                    break
                prev, cur = cur, nxt

            if len(cycle) == len(comp_nodes):
                for idx in range(len(cycle)):
                    a, b = cycle[idx], cycle[(idx + 1) % len(cycle)]
                    k = key(a, b)
                    if k not in orders:
                        if idx < len(cycle) - 1:
                            orders[k] = 2 if idx % 2 == 0 else 1
                        else:
                            orders[k] = 1
        else:
            for a, b in res_bonds:
                if a in comp_nodes or b in comp_nodes:
                    k = key(a, b)
                    if k not in orders:
                        orders[k] = 1

    return orders


def _kekule_prepass(mol: "Molecule", obmol) -> None:
    """
    Convert Molvector order-5 (resonance) bonds into alternating
    single/double (Kekulé pattern) for OpenBabel.

    For each connected component:
      * **Simple cycle** (every node degree 2): walk around assigning
        alternating bond orders (2, 1, 2, 1, …).
      * **Otherwise** (fused rings, branched): fall back to all-single bonds
        (still planar, still works with MMFF94s; bonds ≈1.45 Å instead of
         the delocalised 1.395 Å — acceptable for a builder tool).

    OpenBabel's MMFF94s auto-detects aromaticity from the alternating
    pattern and uses proper delocalised parameters.
    """
    orders = _kekule_orders(mol)
    if not orders:
        return

    for (i, j), bo in orders.items():
        obmol.AddBond(i + 1, j + 1, bo)

def bond_half_line(
    ax:float, ay:float, bx:float, by:float,
    atom_r_px:float, offset:float = 0.0,
) -> Tuple[Optional[Tuple[Tuple[float,float], Tuple[float,float]]], Tuple[float,float]]:
    dx, dy = bx-ax, by-ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None, (0.0, 0.0)
    ux,uy = dx/length, dy/length
    px,py = -uy, ux
    
    cx_a = ax + px*offset; cy_a = ay + py*offset
    cx_b = bx + px*offset; cy_b = by + py*offset
    
    x0 = cx_a + ux*atom_r_px;  y0 = cy_a + uy*atom_r_px
    x1 = (cx_a+cx_b)/2;        y1 = (cy_a+cy_b)/2
    return ((x0, y0), (x1, y1)), (px, py)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDERER
# ─────────────────────────────────────────────────────────────────────────────


def project_molecule(
    mol: Molecule,
    rot: np.ndarray,
    pan_x: float,
    pan_y: float,
    canvas_w: int,
    canvas_h: int,
    scale: float,
    atom_scale: float
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float, float, int]]]:
    """
    Project 3D molecule to 2D canvas coordinates.
    Returns:
      atom_projs: list of (px, py, pz, r_px)
      bond_projs: list of (ax, ay, bx, by, z_avg, bond_idx)
    """
    centered = center_positions(mol.atoms)
    cx = canvas_w/2 + pan_x
    cy = canvas_h/2 + pan_y
    CAMERA_Z = 60.0
    
    atom_projs = []
    for i, atom in enumerate(mol.atoms):
        rp = rot @ centered[i]
        elem = atom.element
        vdw  = VDW_RADII.get(elem, DEFAULT_VDW)
        z_factor = CAMERA_Z / (CAMERA_Z - rp[2]) if (CAMERA_Z - rp[2]) != 0 else 1.0
        r_px = vdw * scale * atom_scale * z_factor
        px = cx + rp[0] * scale * z_factor
        py = cy - rp[1] * scale * z_factor
        atom_projs.append((px, py, rp[2], r_px))
        
    bond_projs = []
    for i, bond in enumerate(mol.bonds):
        pA, pB = atom_projs[bond.i], atom_projs[bond.j]
        bond_projs.append((pA[0], pA[1], pB[0], pB[1], (pA[2] + pB[2])/2, i))
        
    return atom_projs, bond_projs

def render_molecule(
    mol: Molecule,
    rot_x: float = 0.0,
    rot_y: float = 0.0,
    rot_z: float = 0.0,
    rot_matrix_override: Optional[np.ndarray] = None,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
    canvas_w: int = 700,
    canvas_h: int = 600,
    scale: float = 110.0,
    atom_scale: float = 0.8,
    bond_width_px: float = 10.0,
    background: str = "#0a0a12",
    color_overrides: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = "molecule.svg",
    export_mode: bool = False,
    vectors: Optional[List[Tuple[int, float, float, float, str]]] = None,
    active_vectors: Optional[np.ndarray] = None,
    selected_indices: Optional[set] = None,
    animation_phase: float = 0.0,
    animation_amplitude: float = 0.0,
    bond_style: str = "gradient",
    bond_color: str = "#444444",
    atom_border_mode: str = "scaled",
    atom_border_scale: float = 1.04,
    atom_border_width: float = 2.0,
    lighting_intensity: float = 1.0,
    light_position: str = "top-left",
    roughness: float = 1.0,
    show_axes: bool = False,
    show_principal_axes: bool = False,
    axes_position: str = "bottom-left",
    principal_axes_position: str = "bottom-left",
    axes_rot_override: Optional[np.ndarray] = None,
    m_normal_override: Optional[np.ndarray] = None,
    fast_render: bool = False,
    svg_prefix: Optional[str] = None,
    cached_grad_stops: Optional[Dict[str, list]] = None,
) -> str:

    rot = rot_matrix_override if rot_matrix_override is not None \
          else rotation_matrix(rot_x, rot_y, rot_z)

    if light_position not in LIGHT_POSITIONS:
        light_position = "top-left"
    grad_cx, grad_cy, Lx, Ly, Lz = LIGHT_POSITIONS[light_position]

    base_colors = dict(CPK_BASE)
    dark_colors  = dict(CPK_DARK)
    if color_overrides:
        for elem, col in color_overrides.items():
            base_colors[elem] = col
            dark_colors[elem]  = auto_dark(col)

    # Per-atom custom colors (overrides element & color_overrides)
    _atom_custom_keys: Dict[int, str] = {}
    for idx, atom in enumerate(mol.atoms):
        if atom.color:
            key = f"_cust{idx}"
            _atom_custom_keys[idx] = key
            base_colors[key] = atom.color
            dark_colors[key] = auto_dark(atom.color)

    centered = center_positions(mol.atoms)
    cx = canvas_w/2 + pan_x
    cy = canvas_h/2 + pan_y

    CAMERA_Z = 60.0
    
    proj = []
    for i, atom in enumerate(mol.atoms):
        pos = centered[i]
        if animation_amplitude > 0 and active_vectors is not None:
            # Oscillation: pos + vector * amp * sin(phase)
            pos = pos + active_vectors[i] * animation_amplitude * math.sin(animation_phase)
            
        rp   = rot @ pos
        elem = atom.element
        vdw  = VDW_RADII.get(elem, DEFAULT_VDW)
        
        z_factor = CAMERA_Z / (CAMERA_Z - rp[2]) if (CAMERA_Z - rp[2]) != 0 else 1.0
        
        r_px = vdw * scale * atom_scale * z_factor
        # Safety clipping
        r_px = max(0.1, min(1000.0, r_px))
        if not math.isfinite(r_px): r_px = 10.0
        
        px_coord = cx + rp[0] * scale * z_factor
        py_coord = cy - rp[1] * scale * z_factor
        
        proj.append((px_coord, py_coord, rp[2], r_px))

    dwg  = svgwrite.Drawing(output_path or "__mem__", size=(canvas_w, canvas_h))
    defs = dwg.defs
    
    # Use a random prefix for all IDs to prevent collisions in Inkscape
    prefix = svg_prefix or "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    if not export_mode:
        dwg.add(dwg.rect(insert=(0,0), size=(canvas_w,canvas_h), fill=background))

    _grad_added: set = set()

    def ensure_grad(elem: str) -> str:
        gid = f"sph_{elem}_{prefix}"
        if gid in _grad_added:
            return gid
        _grad_added.add(gid)
        cached = (cached_grad_stops or {}).get(elem)
        if cached is not None:
            g = dwg.radialGradient(id=gid, center=(grad_cx, grad_cy), r="0.68")
            g["fx"]=grad_cx; g["fy"]=grad_cy
            g["gradientUnits"] = "objectBoundingBox"
            for pct, color in cached:
                g.add_stop_color(f"{pct*100:.1f}%", color, 1.0)
            defs.add(g)
            return gid
        base = base_colors.get(elem, DEFAULT_BASE)
        dark = dark_colors.get(elem, DEFAULT_DARK)
        li = lighting_intensity
        g = dwg.radialGradient(id=gid, center=(grad_cx, grad_cy), r="0.68")
        g["fx"]=grad_cx; g["fy"]=grad_cy
        g["gradientUnits"] = "objectBoundingBox"
        stops = compute_gradient_stops(base, dark, li, roughness)
        for pct, color in stops:
            g.add_stop_color(f"{pct*100:.1f}%", color, 1.0)
        defs.add(g)
        if cached_grad_stops is not None:
            cached_grad_stops[elem] = stops
        return gid

    if not fast_render:
        for idx, atom in enumerate(mol.atoms):
            key = _atom_custom_keys.get(idx, atom.element)
            ensure_grad(key)

    # SVD to find best-fit molecule plane normal
    if m_normal_override is not None:
        M_normal = m_normal_override
    elif len(centered) >= 3:
        U, S, Vt = np.linalg.svd(centered)
        M_normal = Vt[-1]
    else:
        M_normal = np.array([0.0, 0.0, 1.0])

    draw_list = []
    for bi, bond in enumerate(mol.bonds):
        ai,aj = bond.i,bond.j
        A_orig = centered[ai]
        B_orig = centered[aj]
        
        if animation_amplitude > 0 and active_vectors is not None:
            A_orig = A_orig + active_vectors[ai] * animation_amplitude * math.sin(animation_phase)
            B_orig = B_orig + active_vectors[aj] * animation_amplitude * math.sin(animation_phase)

        u_3D = B_orig - A_orig
        u_len = np.linalg.norm(u_3D)
        
        rp_A_orig = rot @ A_orig
        rp_B_orig = rot @ B_orig
        orig_az = rp_A_orig[2]
        orig_bz = rp_B_orig[2]
        
        dir_3D = np.cross(M_normal, u_3D)
        if np.linalg.norm(dir_3D) < 1e-6:
            fallback = np.array([1.0, 0.0, 0.0])
            if np.linalg.norm(np.cross(u_3D, fallback)) < 1e-6:
                fallback = np.array([0.0, 1.0, 0.0])
            dir_3D = np.cross(fallback, u_3D)
        dir_3D = dir_3D / np.linalg.norm(dir_3D)
        
        # Bond surface offset in Angstroms (push to sphere surface in 3D)
        if u_len > 1e-6:
            u_hat = u_3D / u_len
            vdw_A = VDW_RADII.get(mol.atoms[ai].element, DEFAULT_VDW)
            vdw_B = VDW_RADII.get(mol.atoms[aj].element, DEFAULT_VDW)
            surf_off_A = vdw_A * atom_scale
            surf_off_B = vdw_B * atom_scale
        else:
            u_hat = np.array([0.0, 0.0, 0.0])
            surf_off_A = 0.0
            surf_off_B = 0.0
        
        # Bond width offset in Angstroms
        hw_angstrom = bond_width_px / 110.0
        
        if bond.order == 1:
            offsets = [0.0]
            indiv_hw_angstrom = hw_angstrom
        elif bond.order == 2:
            offsets = [-hw_angstrom * 1.1, hw_angstrom * 1.1]
            indiv_hw_angstrom = hw_angstrom * 0.6
        elif bond.order == 5:
            offsets = [0.0]
            indiv_hw_angstrom = hw_angstrom
        else:
            offsets = [-hw_angstrom * 2.0, 0.0, hw_angstrom * 2.0]
            indiv_hw_angstrom = hw_angstrom * 0.5
            
        if u_len > 1e-6 and u_len <= surf_off_A + surf_off_B:
            continue
        
        for o_idx, offset_A in enumerate(offsets):
            A_offset = A_orig + offset_A * dir_3D + u_hat * surf_off_A
            B_offset = B_orig + offset_A * dir_3D - u_hat * surf_off_B
            
            rpA = rot @ A_offset
            rpB = rot @ B_offset
            
            zA_factor = CAMERA_Z / (CAMERA_Z - rpA[2]) if (CAMERA_Z - rpA[2]) != 0 else 1.0
            zB_factor = CAMERA_Z / (CAMERA_Z - rpB[2]) if (CAMERA_Z - rpB[2]) != 0 else 1.0
            
            ax, ay, az = cx + rpA[0]*scale*zA_factor, cy - rpA[1]*scale*zA_factor, rpA[2]
            bx, by, bz = cx + rpB[0]*scale*zB_factor, cy - rpB[1]*scale*zB_factor, rpB[2]
            
            avg_z_factor = (zA_factor + zB_factor) / 2.0
            # Safety: Ensure width is positive and finite
            indiv_hw_px = max(0.1, min(100.0, indiv_hw_angstrom * scale * avg_z_factor))
            if not math.isfinite(indiv_hw_px): indiv_hw_px = 1.0
            
            bdx = bx - ax
            bdy = by - ay
            bond_len_2d = math.hypot(bdx, bdy)
            if bond_len_2d < 0.01:
                continue
            ux_bond = bdx / bond_len_2d
            uy_bond = bdy / bond_len_2d
            px = -uy_bond
            py = ux_bond

            if fast_render:
                z_sort = (orig_az + orig_bz) / 2.0
                if bond_style in ("grey", "unicolor"):
                    flat_col = bond_color
                else:
                    flat_col = base_colors.get(_atom_custom_keys.get(ai, mol.atoms[ai].element), DEFAULT_BASE)
                draw_list.append((z_sort, 0, ("bond_flat", (ax, ay, bx, by), px, py, indiv_hw_px, flat_col)))
                continue

            if bond_style in ("grey", "unicolor"):
                base = bond_color
                dark = auto_dark(bond_color)
                z_sort = (orig_az + orig_bz) / 2.0
                pts = ((ax, ay), (bx, by))
                b_id = f"b_{bi}_{o_idx}_{prefix}"
                draw_list.append((z_sort, 0, ("bond_half", pts, px, py, indiv_hw_px, b_id, base, dark)))
            elif bond_style in ("match", "splitted"):
                keyA = _atom_custom_keys.get(ai, mol.atoms[ai].element)
                keyB = _atom_custom_keys.get(aj, mol.atoms[aj].element)
                base_A = base_colors.get(keyA, DEFAULT_BASE)
                base_B = base_colors.get(keyB, DEFAULT_BASE)
                dark_A = dark_colors.get(keyA, DEFAULT_DARK)
                dark_B = dark_colors.get(keyB, DEFAULT_DARK)
                z_sort = (orig_az + orig_bz) / 2.0
                overlap = 0.04
                # A-side half (extends slightly past midpoint)
                tA0, tA1 = 0.0, 0.5 + overlap * 0.5
                rpA_A = rpA * (1.0 - tA0) + rpB * tA0
                rpB_A = rpA * (1.0 - tA1) + rpB * tA1
                zA_A = CAMERA_Z / max(1e-6, CAMERA_Z - rpA_A[2]) if rpA_A[2] < CAMERA_Z else CAMERA_Z / 1e-6
                zB_A = CAMERA_Z / max(1e-6, CAMERA_Z - rpB_A[2]) if rpB_A[2] < CAMERA_Z else CAMERA_Z / 1e-6
                sAx = cx + rpA_A[0] * scale * zA_A; sAy = cy - rpA_A[1] * scale * zA_A
                eAx = cx + rpB_A[0] * scale * zB_A; eAy = cy - rpB_A[1] * scale * zB_A
                seg_bdx = eAx - sAx; seg_bdy = eAy - sAy
                seg_len = math.hypot(seg_bdx, seg_bdy)
                if seg_len >= 0.01:
                    ux = seg_bdx / seg_len; uy = seg_bdy / seg_len
                    ppx, ppy = -uy, ux
                    avg_z = (zA_A + zB_A) * 0.5
                    hw = max(0.1, min(100.0, indiv_hw_angstrom * scale * avg_z))
                    if not math.isfinite(hw): hw = 1.0
                    draw_list.append((z_sort, 0, ("bond_half", ((sAx,sAy),(eAx,eAy)), ppx, ppy, hw, f"b_{bi}_{o_idx}_A_{prefix}", base_A, dark_A)))
                # B-side half (starts slightly before midpoint)
                tB0, tB1 = 0.5 - overlap * 0.5, 1.0
                rpA_B = rpA * (1.0 - tB0) + rpB * tB0
                rpB_B = rpA * (1.0 - tB1) + rpB * tB1
                zA_B = CAMERA_Z / max(1e-6, CAMERA_Z - rpA_B[2]) if rpA_B[2] < CAMERA_Z else CAMERA_Z / 1e-6
                zB_B = CAMERA_Z / max(1e-6, CAMERA_Z - rpB_B[2]) if rpB_B[2] < CAMERA_Z else CAMERA_Z / 1e-6
                sBx = cx + rpA_B[0] * scale * zA_B; sBy = cy - rpA_B[1] * scale * zA_B
                eBx = cx + rpB_B[0] * scale * zB_B; eBy = cy - rpB_B[1] * scale * zB_B
                seg_bdx = eBx - sBx; seg_bdy = eBy - sBy
                seg_len = math.hypot(seg_bdx, seg_bdy)
                if seg_len >= 0.01:
                    ux = seg_bdx / seg_len; uy = seg_bdy / seg_len
                    ppx, ppy = -uy, ux
                    avg_z = (zA_B + zB_B) * 0.5
                    hw = max(0.1, min(100.0, indiv_hw_angstrom * scale * avg_z))
                    if not math.isfinite(hw): hw = 1.0
                    draw_list.append((z_sort, 0, ("bond_half", ((sBx,sBy),(eBx,eBy)), ppx, ppy, hw, f"b_{bi}_{o_idx}_B_{prefix}", base_B, dark_B)))
            else:
                keyA = _atom_custom_keys.get(ai, mol.atoms[ai].element)
                keyB = _atom_custom_keys.get(aj, mol.atoms[aj].element)
                base_A = base_colors.get(keyA, DEFAULT_BASE)
                base_B = base_colors.get(keyB, DEFAULT_BASE)
                dark_A = dark_colors.get(mol.atoms[ai].element, DEFAULT_DARK)
                dark_B = dark_colors.get(mol.atoms[aj].element, DEFAULT_DARK)
                z_sort = (orig_az + orig_bz) / 2.0
                pts = ((ax, ay), (bx, by))
                b_id = f"b_{bi}_{o_idx}_{prefix}"
                draw_list.append((z_sort, 0, ("bond_gradient", pts, px, py, indiv_hw_px, b_id, base_A, dark_A, base_B, dark_B)))

    for idx,atom in enumerate(mol.atoms):
        ax,ay,az,ar = proj[idx]
        key = _atom_custom_keys.get(idx, atom.element)
        base = base_colors.get(key, DEFAULT_BASE)
        dark = dark_colors.get(key, DEFAULT_DARK)
        is_sel = selected_indices and idx in selected_indices
        if fast_render:
            draw_list.append((az, 1, ("atom_flat",ax,ay,ar,base,dark,atom.element,is_sel)))
        else:
            gid  = ensure_grad(key)
            draw_list.append((az, 1, ("atom",ax,ay,ar,gid,base,dark,atom.element,is_sel)))

    # Add vectors (e.g. vibrational displacements)
    if vectors:
        # Vector format: (atom_idx, dx, dy, dz, color)
        for ai, vx, vy, vz, vcol in vectors:
            A_orig = centered[ai]
            if animation_amplitude > 0 and active_vectors is not None:
                A_orig = A_orig + active_vectors[ai] * animation_amplitude * math.sin(animation_phase)
            B_orig = A_orig + np.array([vx, vy, vz])
            
            rpA = rot @ A_orig
            rpB = rot @ B_orig
            
            zA_factor = CAMERA_Z / (CAMERA_Z - rpA[2]) if (CAMERA_Z - rpA[2]) != 0 else 1.0
            zB_factor = CAMERA_Z / (CAMERA_Z - rpB[2]) if (CAMERA_Z - rpB[2]) != 0 else 1.0
            
            ax, ay, az = cx + rpA[0]*scale*zA_factor, cy - rpA[1]*scale*zA_factor, rpA[2]
            bx, by, bz = cx + rpB[0]*scale*zB_factor, cy - rpB[1]*scale*zB_factor, rpB[2]
            
            # Simple arrow: line + triangle at the tip
            draw_list.append((max(az, bz), 2, ("vector", ax, ay, bx, by, vcol)))

    draw_list.sort(key=lambda x:(x[0],x[1]))

    molecule_group = dwg.g(id="molecule")
    for _,__,item in draw_list:
        kind = item[0]
        if kind == "bond_flat":
            _, (x0, y0, x1, y1), px, py, hw, flat_col = item
            dx = x1 - x0
            dy = y1 - y0
            length = math.hypot(dx, dy)
            if length > 0:
                ux = dx / length
                uy = dy / length
                x0_r = x0 - ux * hw
                y0_r = y0 - uy * hw
                x1_r = x1 + ux * hw
                y1_r = y1 + uy * hw
            else:
                x0_r, y0_r = x0, y0
                x1_r, y1_r = x1, y1
            rw = math.hypot(x1_r - x0_r, y1_r - y0_r)
            if rw < 0.01:
                continue
            rx = (x0_r + x1_r) / 2
            ry = (y0_r + y1_r) / 2
            angle = math.degrees(math.atan2(-px, py))
            molecule_group.add(dwg.rect(
                insert=(-rw / 2, -hw),
                size=(rw, hw * 2),
                rx=hw, ry=hw,
                fill=flat_col, stroke="none",
                transform=f"translate({rx:.1f},{ry:.1f}) rotate({angle:.1f})"
            ))
        elif kind == "bond_half":
            _, pts, px, py, indiv_hw, b_id, base, dark = item
            
            A = px * Lx + py * Ly
            I_max = math.hypot(A, Lz)
            if I_max == 0: I_max = 1e-6
            
            def get_color_from_ratio(r: float) -> str:
                d = 1.0 - r
                li = lighting_intensity
                stops = compute_gradient_stops(
                    base, dark, li, roughness,
                    edge_fn=lambda b, d, l: interpolate_color(b, d, l))
                if d <= 0: return stops[0][1]
                if d >= 1: return stops[-1][1]
                for i in range(len(stops)-1):
                    if stops[i][0] <= d <= stops[i+1][0]:
                        t_ratio = (d - stops[i][0]) / (stops[i+1][0] - stops[i][0])
                        return interpolate_color(stops[i][1], stops[i+1][1], t_ratio)
                return dark

            x0, y0 = pts[0]
            x1, y1 = pts[1]
            hw = indiv_hw

            dx = x1 - x0
            dy = y1 - y0
            length = math.hypot(dx, dy)
            if length > 0:
                ux = dx / length
                uy = dy / length
                x0_r = x0 - ux * hw
                y0_r = y0 - uy * hw
                x1_r = x1 + ux * hw
                y1_r = y1 + uy * hw
            else:
                x0_r, y0_r = x0, y0
                x1_r, y1_r = x1, y1

            rw = math.hypot(x1_r - x0_r, y1_r - y0_r)
            if rw < 0.01:
                continue
            rx = (x0_r + x1_r) / 2
            ry = (y0_r + y1_r) / 2
            angle = math.degrees(math.atan2(-px, py))

            g = dwg.linearGradient(
                id=b_id,
                start=(0.5, 1), end=(0.5, 0),
                gradientUnits="objectBoundingBox"
            )

            for i in range(11):
                v = i / 10.0
                s = 1.0 - 2.0 * v
                intensity = max(0.0, s * A + math.sqrt(max(0.0, 1.0 - s**2)) * Lz)
                ratio = intensity / I_max
                g.add_stop_color(f"{v*100:.1f}%", get_color_from_ratio(ratio), 1.0)
            defs.add(g)

            molecule_group.add(dwg.rect(
                insert=(-rw / 2, -hw),
                size=(rw, hw * 2),
                rx=hw, ry=hw,
                fill=f"url(#{b_id})",
                stroke="none",
                transform=f"translate({rx:.1f},{ry:.1f}) rotate({angle:.1f})"
            ))
        elif kind == "bond_gradient":
            _, pts, px, py, indiv_hw, b_id, base_A, dark_A, base_B, dark_B = item

            def mat_color(base, dark, ratio):
                d = 1.0 - ratio
                li = lighting_intensity
                stops = compute_gradient_stops(
                    base, dark, li, roughness,
                    edge_fn=lambda b, d, l: interpolate_color(b, d, l))
                if d <= 0: return stops[0][1]
                if d >= 1: return stops[-1][1]
                for i in range(len(stops)-1):
                    if stops[i][0] <= d <= stops[i+1][0]:
                        return interpolate_color(stops[i][1], stops[i+1][1], (d - stops[i][0]) / (stops[i+1][0] - stops[i][0]))
                return dark

            x0, y0 = pts[0]; x1, y1 = pts[1]
            hw = indiv_hw
            dx = x1 - x0; dy = y1 - y0
            length = math.hypot(dx, dy)
            if length > 0:
                ux = dx / length; uy = dy / length
                x0_r = x0 - ux * hw; y0_r = y0 - uy * hw
                x1_r = x1 + ux * hw; y1_r = y1 + uy * hw
            else:
                x0_r, y0_r = x0, y0; x1_r, y1_r = x1, y1
            rw = math.hypot(x1_r - x0_r, y1_r - y0_r)
            if rw < 0.01: continue
            rx = (x0_r + x1_r) / 2; ry = (y0_r + y1_r) / 2
            angle = math.degrees(math.atan2(-px, py))

            A_sh = px * Lx + py * Ly
            I_max = math.hypot(A_sh, Lz)
            if I_max == 0: I_max = 1e-6

            # Base layer: gradient along bond axis (A→B at peak brightness)
            g = dwg.linearGradient(
                id=b_id, start=(0, 0.5), end=(1, 0.5),
                gradientUnits="objectBoundingBox"
            )
            num_stops = 21
            for i in range(num_stops):
                t = i / (num_stops - 1)
                ca = mat_color(base_A, dark_A, 1.0)
                cb = mat_color(base_B, dark_B, 1.0)
                g.add_stop_color(f"{t*100:.1f}%", interpolate_color(ca, cb, t), 1.0)
            defs.add(g)

            molecule_group.add(dwg.rect(
                insert=(-rw / 2, -hw), size=(rw, hw * 2),
                rx=hw, ry=hw, fill=f"url(#{b_id})", stroke="none",
                transform=f"translate({rx:.1f},{ry:.1f}) rotate({angle:.1f})"
            ))

            # Overlay: perpendicular 3D shading (darken edges)
            sh_id = f"{b_id}_sh"
            sh_g = dwg.linearGradient(
                id=sh_id, start=(0.5, 1), end=(0.5, 0),
                gradientUnits="objectBoundingBox"
            )
            for i in range(11):
                v = i / 10.0
                s = 1.0 - 2.0 * v
                intensity = max(0.0, s * A_sh + math.sqrt(max(0.0, 1.0 - s**2)) * Lz)
                ratio = intensity / I_max
                sh_g.add_stop_color(f"{v*100:.1f}%", "#000000", (1.0 - ratio) * 0.75 * lighting_intensity)
            defs.add(sh_g)

            molecule_group.add(dwg.rect(
                insert=(-rw / 2, -hw), size=(rw, hw * 2),
                rx=hw, ry=hw, fill=f"url(#{sh_id})", stroke="none",
                transform=f"translate({rx:.1f},{ry:.1f}) rotate({angle:.1f})"
            ))
        elif kind == "atom_flat":
            _, ax, ay, ar, base, dark, elem, is_sel = item
            if is_sel:
                molecule_group.add(dwg.circle(center=(ax,ay), r=ar*1.35, fill="none", stroke="#00aaff", stroke_width=2.0, opacity="0.45"))
                molecule_group.add(dwg.circle(center=(ax,ay), r=ar*1.15, fill="none", stroke="#44ddff", stroke_width=1.2, opacity="0.75"))
            if atom_border_mode == "none":
                pass
            elif atom_border_mode == "constant":
                molecule_group.add(dwg.circle(center=(ax,ay),r=ar+atom_border_width,fill=dark,stroke="none"))
            else:
                molecule_group.add(dwg.circle(center=(ax,ay),r=ar*atom_border_scale,fill=dark,stroke="none"))
            molecule_group.add(dwg.circle(center=(ax,ay),r=ar,fill=base,stroke="none"))
        elif kind == "atom":
            _, ax, ay, ar, gid, base, dark, elem, is_sel = item
            if is_sel:
                molecule_group.add(dwg.circle(center=(ax,ay), r=ar*1.35, fill="none", stroke="#00aaff", stroke_width=2.0, opacity="0.45"))
                molecule_group.add(dwg.circle(center=(ax,ay), r=ar*1.15, fill="none", stroke="#44ddff", stroke_width=1.2, opacity="0.75"))
            if atom_border_mode == "none":
                pass
            elif atom_border_mode == "constant":
                molecule_group.add(dwg.circle(center=(ax,ay),r=ar+atom_border_width,fill=dark,stroke="none"))
            else:  # "scaled"
                molecule_group.add(dwg.circle(center=(ax,ay),r=ar*atom_border_scale,fill=dark,stroke="none"))
            molecule_group.add(dwg.circle(center=(ax,ay),r=ar,fill=f"url(#{gid})",stroke="none"))
        elif kind == "vector":
            _, x0, y0, x1, y1, col = item
            # Main line
            molecule_group.add(dwg.line(start=(x0, y0), end=(x1, y1), stroke=col, stroke_width=2.5, stroke_linecap="round"))
            # Arrowhead
            dx, dy = x1-x0, y1-y0
            L = math.hypot(dx, dy)
            if L > 1e-3:
                ux, uy = dx/L, dy/L
                px, py = -uy, ux
                # Points for triangle
                p1 = (x1, y1)
                p2 = (x1 - ux*6 + px*3.5, y1 - uy*6 + py*3.5)
                p3 = (x1 - ux*6 - px*3.5, y1 - uy*6 - py*3.5)
                molecule_group.add(dwg.polygon(points=[p1, p2, p3], fill=col))
    
    molecule_group['id'] = f"mol_{prefix}"
    dwg.add(molecule_group)

    if not export_mode:
        dwg.add(dwg.text(
            mol.name, insert=(16,26),
            font_size=15,font_family="monospace",
            fill="#88ccff",opacity=0.75,
        ))

    if show_axes and not export_mode:
        _ax_rot = axes_rot_override if axes_rot_override is not None else rot
        axis_len = 17.0
        margin = 30
        _ax_pos = axes_position.lower().replace(" ", "-")
        if _ax_pos == "bottom-right":
            origin_x, origin_y = canvas_w - margin, canvas_h - margin
        elif _ax_pos == "top-left":
            origin_x, origin_y = margin, margin
        elif _ax_pos == "top-right":
            origin_x, origin_y = canvas_w - margin, margin
        else:
            origin_x, origin_y = margin, canvas_h - margin
        for vec, col, lbl in [
            (np.array([1,0,0]), "#ff4444", "X"),
            (np.array([0,1,0]), "#44cc44", "Y"),
            (np.array([0,0,1]), "#4488ff", "Z"),
        ]:
            rv = _ax_rot @ vec
            ex = origin_x + rv[0] * axis_len
            ey = origin_y - rv[1] * axis_len
            dwg.add(dwg.line(start=(origin_x, origin_y), end=(ex, ey),
                             stroke="#222222", stroke_width=1.0, stroke_linecap="round"))
            tx = origin_x + rv[0] * (axis_len + 6) - 3
            ty = origin_y - rv[1] * (axis_len + 6) + 3
            dwg.add(dwg.text(lbl, insert=(tx, ty),
                             font_size=8, font_family="monospace",
                             fill=col, font_weight="bold"))

    if show_principal_axes and not export_mode and len(mol.atoms) >= 2:
        coords = np.array([[a.x, a.y, a.z] for a in mol.atoms])
        masses = np.array([ATOMIC_MASSES.get(a.element, 0.0) for a in mol.atoms])
        _, eigvecs = strudel.diagonalize_I_tensor(coords, masses)
        axis_len = 17.0
        margin = 30
        _pa_pos = principal_axes_position.lower().replace(" ", "-")
        if _pa_pos == "bottom-right":
            origin_x, origin_y = canvas_w - margin, canvas_h - margin
        elif _pa_pos == "top-left":
            origin_x, origin_y = margin, margin
        elif _pa_pos == "top-right":
            origin_x, origin_y = canvas_w - margin, margin
        else:
            origin_x, origin_y = margin, canvas_h - margin
        for i, col, lbl in [
            (0, "#ff8800", "a"),
            (1, "#cc44cc", "b"),
            (2, "#00bbbb", "c"),
        ]:
            vec = eigvecs[:, i]
            rv = rot @ vec
            ex = origin_x + rv[0] * axis_len
            ey = origin_y - rv[1] * axis_len
            dwg.add(dwg.line(start=(origin_x, origin_y), end=(ex, ey),
                             stroke="#222222", stroke_width=1.0, stroke_linecap="round",
                             stroke_dasharray="3,2"))
            tx = origin_x + rv[0] * (axis_len + 6) - 3
            ty = origin_y - rv[1] * (axis_len + 6) + 3
            dwg.add(dwg.text(lbl, insert=(tx, ty),
                             font_size=8, font_family="monospace",
                             fill=col, font_weight="bold"))

    if output_path:
        dwg.save()
        return output_path
    return dwg.tostring()