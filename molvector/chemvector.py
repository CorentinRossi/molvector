"""
chemvector.py — 2D organic-chemistry "whiteboard" depiction for Molvector
===========================================================================
Chemvector renders a `Molecule` (the same data model used by the 3D viewer
in render.py/gui.py) as a flat, hand-drawn-style 2D structural formula:

  * Bonds are plain SVG lines (double/triple bonds as parallel lines).
  * Atoms are colored element letters (CPK colors, shared with the 3D view).
  * Stereocenters perceived from the molecule's 3D coordinates are drawn
    with solid wedge (toward viewer) / hashed wedge (away from viewer)
    triangles, per standard organic chemistry convention.

Two 2D-layout backends are provided:
  * RDKit (preferred) — proper graph-based coordinate generation
    (rdCoordGen / Compute2DCoords) plus real stereo-wedge perception via
    Chem.AssignStereochemistryFrom3D + Chem.WedgeMolBonds.
  * A dependency-free fallback that projects the existing 3D coordinates
    onto their best-fit plane (SVD) and derives wedge/hash bonds from the
    out-of-plane component. Lower quality for genuinely 3D/crowded
    molecules, but requires nothing beyond numpy.

This module intentionally mirrors conventions already used in render.py
(svgwrite output, CPK_BASE colors, bond.order == 5 for aromatic/resonance
bonds, the `_kekule_orders` helper) so that the 3D and 2D views stay
visually and chemically consistent.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import svgwrite

from PyQt6.QtCore import QByteArray, QEvent, QPoint, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (QColorDialog, QFrame, QHBoxLayout, QLineEdit, QSizePolicy,
                             QToolButton, QVBoxLayout)

from molvector.render import (
    Atom,
    Bond,
    CPK_BASE,
    DEFAULT_BASE,
    ATOMIC_MASSES,
    Molecule,
    _kekule_orders,
)

# RDKit is needed for 2D depictions as soon as the window opens, so it is
# loaded eagerly — but only through native_guard, which first checks in a
# throw-away child that loading it next to everything already in this process
# (PyQt6, numpy, ...) does not SEGFAULT.  See molvector/native_guard.py.
from molvector import native_guard

HAS_RDKIT = native_guard.load("rdkit") is not None
if HAS_RDKIT:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")  # RDKit logs sanitize failures straight to stderr

if HAS_RDKIT:
    try:
        from rdkit.Chem import rdCoordGen
        HAS_COORDGEN = True
    except ImportError:
        HAS_COORDGEN = False
else:
    HAS_COORDGEN = False


# Effective drawing info for one bond, keyed by index into mol.bonds:
#   kind:  "none" | "wedge" | "hash"
#   apex:  atom index the wedge/hash narrows to (the in-plane atom)
#   order: effective bond order to draw (1/2/3 — aromatic already kekulized)
BondInfo = Dict[int, dict]
Positions2D = Dict[int, Tuple[float, float]]


# ─────────────────────────────────────────────────────────────────────────────
# 2D LAYOUT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

# How far (Angstrom) two bonded atoms may sit off the molecule's best-fit
# plane, relative to each other, and still be drawn as a plain in-plane bond
# rather than a wedge/hash — so a slightly puckered or noisy "flat" geometry
# doesn't sprout stereo bonds.
PLANARITY_TOLERANCE = 0.65

# The 3D view's camera distance (see render.project_molecule): atoms nearer the
# viewer are drawn slightly larger and spread from the center. The "match 3D"
# layout applies the same factor so it lines up with the 3D view atom for atom.
PERSPECTIVE_CAMERA_Z = 60.0

# The 2D view is drawn at the 3D view's own scale — pixels per Angstrom and
# zoom are shared — so a bond is the same length in both. Line thickness,
# letters and spacing are defined for a bond of this many pixels and scale
# in proportion to the actual drawn bond length, so their ratio to it is the
# same for every molecule.
CHEM_SIZE_REF_BOND_PX = 50.0


def _plane_heights(mol: Molecule) -> Optional[np.ndarray]:
    """Each atom's signed height above the molecule's best-fit plane (None if undefined)."""
    if len(mol.atoms) < 3:
        return None
    coords = np.array([[a.x, a.y, a.z] for a in mol.atoms], dtype=float)
    centered = coords - coords.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None
    return centered @ vt[2]


def _flatten_near_planar_bonds(mol: Molecule, bond_info: BondInfo, tolerance: float) -> None:
    """Turn wedge/hash bonds whose end atoms differ in height by no more than *tolerance* back into plain bonds."""
    z = _plane_heights(mol)
    if z is None:
        return
    for bi, info in bond_info.items():
        if info.get("kind", "none") == "none":
            continue
        b = mol.bonds[bi]
        if abs(z[b.i] - z[b.j]) <= tolerance:
            info["kind"] = "none"


def _plane_frame(centered: np.ndarray, rot: np.ndarray):
    """
    Right-handed (u, v, normal) basis of the best-fit plane of *centered* 3D
    coordinates, with the normal pointing toward the viewer of camera
    rotation *rot* — so "z > 0" means toward the viewer and a plane
    projection along u, v is the front view.
    """
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        u, n = vt[0], vt[2] if len(vt) > 2 else np.array([0.0, 0.0, 1.0])
    except Exception:
        u, n = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    if (rot @ n)[2] < 0:
        n = -n
    v = np.cross(n, u)
    return u, v, n


def _procrustes_view(Q: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Orthogonal 2x2 *M* (row-vector convention: Q_centered @ M ~ P_centered)
    and whether it is a reflection. A reflection is only chosen when it fits
    clearly better than a plain rotation — it means the layout is drawn as
    seen from behind relative to the 3D view (wedge/hash must then swap).
    """
    Qc, Pc = Q - Q.mean(axis=0), P - P.mean(axis=0)
    if float((Pc ** 2).sum()) < 1e-8 or float((Qc ** 2).sum()) < 1e-8:
        return np.eye(2), False
    U, _, Vt = np.linalg.svd(Qc.T @ Pc)
    m1 = U @ Vt
    m2 = U @ np.diag([1.0, -1.0]) @ Vt
    rot_m, refl_m = (m1, m2) if np.linalg.det(m1) > 0 else (m2, m1)
    res = lambda m: float(((Qc @ m - Pc) ** 2).sum())
    if res(refl_m) < 0.6 * res(rot_m):
        return refl_m, True
    return rot_m, False


def _swap_wedge_hash(bond_info: BondInfo) -> None:
    for info in bond_info.values():
        if info.get("kind") == "wedge":
            info["kind"] = "hash"
        elif info.get("kind") == "hash":
            info["kind"] = "wedge"


def _align_layout_to_view(mol: Molecule, positions: Positions2D, rot: np.ndarray,
                          rescale: bool) -> Tuple[Positions2D, bool]:
    """
    Rotate (and, if it fits clearly better, mirror) the 2D *positions* so the
    depiction matches the orientation of the 3D view with camera rotation
    *rot* — the layout is centered on the projection of the molecule's 3D
    centroid. With *rescale*, the layout units are also converted to
    Angstrom (so they line up with the 3D coordinates, independent of tilt).
    Returns (positions, reflected).
    """
    idx = [i for i in range(len(mol.atoms)) if i in positions]
    if len(idx) < 2:
        return positions, False
    coords = np.array([[mol.atoms[i].x, mol.atoms[i].y, mol.atoms[i].z] for i in idx], dtype=float)
    origin = np.array([[a.x, a.y, a.z] for a in mol.atoms], dtype=float).mean(axis=0)
    P = ((rot @ (coords - origin).T).T)[:, :2]
    Q = np.array([positions[i] for i in idx], dtype=float)
    M, reflected = _procrustes_view(Q, P)
    Qc = Q - Q.mean(axis=0)
    scale = 1.0
    if rescale:
        lens3d, lens2d = [], []
        for heavy_only in (True, False):   # match the 3D heavy-atom bonds; hydrogens only if there are none
            for b in mol.bonds:
                if b.i in positions and b.j in positions:
                    if heavy_only and "H" in (mol.atoms[b.i].element, mol.atoms[b.j].element):
                        continue
                    d3 = math.dist((mol.atoms[b.i].x, mol.atoms[b.i].y, mol.atoms[b.i].z),
                                   (mol.atoms[b.j].x, mol.atoms[b.j].y, mol.atoms[b.j].z))
                    d2 = math.dist(positions[b.i], positions[b.j])
                    if d3 > 1e-6 and d2 > 1e-6:
                        lens3d.append(d3)
                        lens2d.append(d2)
            if lens3d:
                break
        if lens3d:
            scale = float(np.median(lens3d) / np.median(lens2d))
    new = scale * (Qc @ M) + P.mean(axis=0)
    out = dict(positions)
    for k, i in enumerate(idx):
        out[i] = (float(new[k, 0]), float(new[k, 1]))
    return out, reflected


def best_fit_plane_normal(mol: Molecule) -> Optional[np.ndarray]:
    """Unit normal of *mol*'s least-squares plane (heavy atoms when there are
    at least 3), or None for fewer than 3 atoms or a (near-)linear molecule."""
    pts = [[a.x, a.y, a.z] for a in mol.atoms if a.element not in ("H", "D", "T")]
    if len(pts) < 3:
        pts = [[a.x, a.y, a.z] for a in mol.atoms]
    if len(pts) < 3:
        return None
    centered = np.array(pts, dtype=float)
    centered -= centered.mean(axis=0)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    if sv[1] < 1e-3 * max(sv[0], 1e-12):
        return None
    return vt[2]


def face_plane_rotation(normal: np.ndarray, view_rot: np.ndarray) -> np.ndarray:
    """
    The camera rotation that turns the plane with unit *normal* (molecule
    coordinates) to face the screen, by the smallest turn from *view_rot*:
    the side already facing the viewer stays in front and the in-plane
    angle is kept as far as possible.
    """
    n_view = view_rot @ normal
    target = np.array([0.0, 0.0, 1.0])
    if n_view[2] < 0:
        target = -target
    axis = np.cross(n_view, target)
    s_ = float(np.linalg.norm(axis))
    if s_ < 1e-10:
        return np.array(view_rot, dtype=float)
    axis /= s_
    c_ = float(np.dot(n_view, target))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return (np.eye(3) + s_ * K + (1.0 - c_) * (K @ K)) @ view_rot


def generate_2d_layout(mol: Molecule, planarity_tolerance: float = PLANARITY_TOLERANCE,
                       view_rot: Optional[np.ndarray] = None, projection: bool = False,
                       plane: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[Positions2D, BondInfo]:
    """
    Compute a 2D depiction layout for *mol*, oriented to match a 3D view.

    With *projection*, no idealized depiction is made at all: every atom sits
    exactly where the 3D view (camera rotation *view_rot*, same perspective)
    draws it. Bonds are plain unless an end is off the molecule's best-fit
    plane by more than the planarity tolerance — or off *plane*, a
    (unit normal, point) pair in 3D coordinates, when given ("the molecule
    drawn in that plane") — in which case wedge/hash shows toward/away.

    *view_rot* is the 3D camera rotation (world -> view); the depiction is
    turned (and mirrored, with wedge/hash swapped, if that fits better) to
    be as close as possible to the 3D projection seen through it, and its
    units are Angstrom. Left None, the identity view is assumed.

    Bonds between atoms within *planarity_tolerance* Angstrom of the
    molecule's best-fit plane (relative to each other) are always drawn as
    plain bonds, never wedge/hash.

    Returns (positions, bond_info):
      positions: {atom_idx: (x, y)} in Angstrom
      bond_info: {bond_idx: {"kind", "apex", "order", "inside"?}} — see
        BondInfo above. Every double-order bond gets an "inside" unit vector
        (in layout space): for a ring bond it points toward its smallest
        containing ring's centroid; for a chain bond it points away from the
        bond's other substituents. It is used to draw the second line of a
        double bond as a short inner line on that side, while the bond's own
        endpoint-to-endpoint line (the "outer" line) is always drawn exactly
        as a single bond would be.
    """
    if not mol.atoms:
        return {}, {}
    rot = np.eye(3) if view_rot is None else np.asarray(view_rot, dtype=float)

    if projection:
        positions, bond_info = _fallback_layout(mol, planarity_tolerance, rot, True, plane)
        _annotate_double_bond_insets(mol, positions, bond_info)
        return positions, bond_info

    layout = None
    if HAS_RDKIT:
        try:
            layout = _rdkit_layout(mol)
        except Exception:
            layout = None
    if layout is None:
        layout = _fallback_layout(mol, planarity_tolerance, rot)
    positions, bond_info = layout

    positions, reflected = _align_layout_to_view(mol, positions, rot, rescale=True)
    if reflected:
        _swap_wedge_hash(bond_info)
    _flatten_near_planar_bonds(mol, bond_info, planarity_tolerance)
    _annotate_double_bond_insets(mol, positions, bond_info)
    _add_cosmetic_tetrahedral_wedges(mol, bond_info, planarity_tolerance, rot)
    return positions, bond_info


def _add_cosmetic_tetrahedral_wedges(mol: Molecule, bond_info: BondInfo,
                                     planarity_tolerance: float = PLANARITY_TOLERANCE,
                                     view_rot: Optional[np.ndarray] = None) -> None:
    """
    RDKit only wedges a real CIP stereocenter — e.g. methane's carbon, with
    four identical H substituents, correctly gets no wedge at all, even
    though it's genuinely non-planar. For a "star" atom whose substituents
    are *all* terminal (nothing to draw beyond them, like methane or
    CH2Cl2), that leaves it with no way to show its real 3D shape once
    implicit hydrogens hide its plain bonds. Pick its most out-of-plane
    substituent as a cosmetic wedge and its most out-of-plane on the other
    side as a cosmetic hash, from the real 3D coordinates — purely to
    convey shape, not stereochemistry. Atoms that already have a wedge/hash
    (real stereocenters) are left untouched.
    """
    degree: Dict[int, int] = {}
    adjacency: Dict[int, List[int]] = {}
    for bi, b in enumerate(mol.bonds):
        degree[b.i] = degree.get(b.i, 0) + 1
        degree[b.j] = degree.get(b.j, 0) + 1
        adjacency.setdefault(b.i, []).append(bi)
        adjacency.setdefault(b.j, []).append(bi)

    coords = np.array([[a.x, a.y, a.z] for a in mol.atoms], dtype=float)
    if len(coords) < 4:
        return
    centered = coords - coords.mean(axis=0)
    _, _, normal = _plane_frame(centered, np.eye(3) if view_rot is None else view_rot)
    z = centered @ normal

    OOP_THRESHOLD = planarity_tolerance
    for atom_idx, bond_indices in adjacency.items():
        if len(bond_indices) < 3:
            continue
        if any(mol.bonds[bi].order != 1 for bi in bond_indices):
            continue  # only decorate genuinely sp3 (all-single-bond) centers
        if any(bond_info.get(bi, {}).get("kind", "none") != "none" for bi in bond_indices):
            continue  # already has a real stereo wedge — don't override it
        others = [mol.bonds[bi].j if mol.bonds[bi].i == atom_idx else mol.bonds[bi].i
                  for bi in bond_indices]
        if any(degree.get(o, 0) != 1 for o in others):
            continue  # only a fully "isolated" center (all leaf substituents)

        rels = [(z[o] - z[atom_idx], bi) for bi, o in zip(bond_indices, others)]
        max_dz, max_bi = max(rels, key=lambda r: r[0])
        min_dz, min_bi = min(rels, key=lambda r: r[0])
        if max_bi == min_bi or (max_dz - min_dz) < OOP_THRESHOLD:
            continue  # essentially planar — no 3D shape worth conveying

        b_max = mol.bonds[max_bi]
        bond_info.setdefault(max_bi, {"kind": "none", "apex": b_max.i, "order": b_max.order})
        bond_info[max_bi]["kind"] = "wedge"
        bond_info[max_bi]["apex"] = atom_idx

        b_min = mol.bonds[min_bi]
        bond_info.setdefault(min_bi, {"kind": "none", "apex": b_min.i, "order": b_min.order})
        bond_info[min_bi]["kind"] = "hash"
        bond_info[min_bi]["apex"] = atom_idx


def _smallest_ring_via_edge(mol: Molecule, a: int, b: int, exclude_bond_idx: int) -> Optional[List[int]]:
    """
    Shortest alternate path between atoms *a* and *b*, excluding the direct
    edge at *exclude_bond_idx* — if one exists, it plus that edge is *a* and
    *b*'s smallest common ring. Returns the ordered ring atom indices, or
    None if they aren't on a common cycle.
    """
    adjacency: Dict[int, List[Tuple[int, int]]] = {}
    for bi, bond in enumerate(mol.bonds):
        adjacency.setdefault(bond.i, []).append((bond.j, bi))
        adjacency.setdefault(bond.j, []).append((bond.i, bi))

    prev: Dict[int, Optional[int]] = {a: None}
    queue = [a]
    while queue:
        node = queue.pop(0)
        if node == b:
            break
        for nb, obi in adjacency.get(node, []):
            if obi == exclude_bond_idx:
                continue
            if nb not in prev:
                prev[nb] = node
                queue.append(nb)

    if b not in prev:
        return None
    path = [b]
    cur = b
    while prev[cur] is not None:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path if len(path) >= 3 else None


def _find_ring_for_atom(mol: Molecule, atom_idx: int) -> Optional[List[int]]:
    """The smallest ring containing atom_idx (via any of its bonds), or None."""
    best: Optional[List[int]] = None
    for bi, b in enumerate(mol.bonds):
        if b.i != atom_idx and b.j != atom_idx:
            continue
        other = b.j if b.i == atom_idx else b.i
        ring = _smallest_ring_via_edge(mol, atom_idx, other, exclude_bond_idx=bi)
        if ring and (best is None or len(ring) < len(best)):
            best = ring
    return best


def _find_ring_for_bond(mol: Molecule, bond_idx: int) -> Optional[List[int]]:
    """The smallest ring containing bond_idx, or None if it's a bridge/chain bond."""
    b = mol.bonds[bond_idx]
    return _smallest_ring_via_edge(mol, b.i, b.j, exclude_bond_idx=bond_idx)


def _find_aromatic_rings(mol: Molecule, extra_aromatic_bond_idxs: Optional[set] = None) -> List[List[int]]:
    """
    Rings whose every bond counts as aromatic — either genuinely marked
    order 5 on *mol* itself (e.g. an aromatic input file), or listed in
    *extra_aromatic_bond_idxs* (ChemCanvas's own 2D-only overlay from
    Check Aromaticity, which never touches mol.bonds — see check_aromaticity).
    Used purely for depiction: render_chemvector draws a circle in each of
    these instead of Kekulized double-bond lines.
    """
    extra = extra_aromatic_bond_idxs or set()

    def is_aromatic(bi: int) -> bool:
        return mol.bonds[bi].order == 5 or bi in extra

    pair_to_bi: Dict[Tuple[int, int], int] = {}
    for bi, b in enumerate(mol.bonds):
        pair_to_bi[(b.i, b.j)] = bi
        pair_to_bi[(b.j, b.i)] = bi

    rings: Dict[frozenset, List[int]] = {}
    for bi in range(len(mol.bonds)):
        if not is_aromatic(bi):
            continue
        ring = _find_ring_for_bond(mol, bi)
        if not ring:
            continue
        n = len(ring)
        if all(is_aromatic(pair_to_bi[(ring[k], ring[(k + 1) % n])]) for k in range(n)):
            rings.setdefault(frozenset(ring), ring)
    return list(rings.values())


def check_aromaticity(mol: Molecule) -> set:
    """
    "Check Aromaticity" (2D Viewer only) — perceive aromaticity from *mol*'s
    current, explicit bond orders (a Kekulé structure, e.g. one drawn by
    hand in 2D build mode with alternating single/double bonds around a
    ring). Purely a detector: it never mutates *mol* — bond.order (and so
    the 3D view, which reads it directly) is completely untouched. The
    caller (ChemCanvas) keeps the result in its own 2D-only overlay, which
    render_chemvector uses to draw a circle in place of Kekulized lines for
    those rings — see render_chemvector's aromatic_bond_idxs.

    Uses RDKit's own aromaticity model when available — the reliable,
    general case, correctly handling fused/bridged systems and
    heteroaromatics (furan, pyrrole, thiophene, ...). Without RDKit, falls
    back to a simple heuristic that only recognizes the common case: an
    all-C/N/O/S fused ring system whose bonds admit a valid alternating
    Kekulé structure (each atom has exactly one double bond in the ring).

    Returns the set of bond indices identified as aromatic.
    """
    if HAS_RDKIT:
        try:
            return _check_aromaticity_rdkit(mol)
        except Exception:
            pass
    return _check_aromaticity_fallback(mol)


def _check_aromaticity_rdkit(mol: Molecule) -> set:
    rw = Chem.RWMol()
    for a in mol.atoms:
        atom = Chem.Atom(a.element)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)

    type_map = {
        1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE, 5: Chem.BondType.AROMATIC,
    }
    aromatic_atoms: set = set()
    for b in mol.bonds:
        bt = type_map.get(b.order, Chem.BondType.SINGLE)
        idx = rw.AddBond(b.i, b.j, bt) - 1
        if bt == Chem.BondType.AROMATIC:
            rw.GetBondWithIdx(idx).SetIsAromatic(True)
            aromatic_atoms.add(b.i)
            aromatic_atoms.add(b.j)
    for i in aromatic_atoms:
        rw.GetAtomWithIdx(i).SetIsAromatic(True)

    if mol.charge != 0:
        heavy = next((i for i, a in enumerate(mol.atoms) if a.element != "H"), None)
        if heavy is not None:
            rw.GetAtomWithIdx(heavy).SetFormalCharge(mol.charge)

    m = rw.GetMol()
    Chem.SanitizeMol(m)  # runs RDKit's aromaticity model over the explicit bond pattern

    aromatic_bond_idxs: set = set()
    for bi, b in enumerate(mol.bonds):
        rb = m.GetBondBetweenAtoms(b.i, b.j)
        if rb is not None and rb.GetIsAromatic():
            aromatic_bond_idxs.add(bi)
    return aromatic_bond_idxs


def _check_aromaticity_fallback(mol: Molecule) -> set:
    """
    Dependency-free aromaticity heuristic. Rather than checking each small
    ring in isolation — which breaks on fused systems, since the shared
    bond between two rings doesn't have to alternate with *both* of its
    rings at once (naphthalene's central bond is single in both) — this
    checks the whole fused ring network (all "subsequent" rings sharing a
    bond with one another) as one unit: it qualifies as aromatic iff every
    atom in it has *exactly one* double bond among its own ring bonds — the
    real, general condition for a valid whole-system Kekulé structure.
    """
    aromatic_elements = {"C", "N", "O", "S"}
    n_bonds = len(mol.bonds)

    is_ring_bond = [
        _smallest_ring_via_edge(mol, b.i, b.j, exclude_bond_idx=bi) is not None
        for bi, b in enumerate(mol.bonds)
    ]
    ring_adjacency: Dict[int, List[int]] = {}
    for bi, b in enumerate(mol.bonds):
        if not is_ring_bond[bi]:
            continue
        ring_adjacency.setdefault(b.i, []).append(bi)
        ring_adjacency.setdefault(b.j, []).append(bi)

    aromatic_bond_idxs: set = set()
    visited: set = set()
    for start_bi in range(n_bonds):
        if not is_ring_bond[start_bi] or start_bi in visited:
            continue

        # Flood-fill the whole fused ring system connected to this bond,
        # through the ring-bond-only subgraph.
        component_bonds = {start_bi}
        atoms = [mol.bonds[start_bi].i, mol.bonds[start_bi].j]
        seen_atoms = set(atoms)
        while atoms:
            atom = atoms.pop()
            for bi in ring_adjacency.get(atom, []):
                if bi in component_bonds:
                    continue
                component_bonds.add(bi)
                b = mol.bonds[bi]
                other = b.j if b.i == atom else b.i
                if other not in seen_atoms:
                    seen_atoms.add(other)
                    atoms.append(other)
        visited.update(component_bonds)

        if not all(mol.atoms[a].element in aromatic_elements for a in seen_atoms):
            continue

        valid = True
        for a in seen_atoms:
            bond_idxs = [bi for bi in ring_adjacency.get(a, []) if bi in component_bonds]
            orders = [mol.bonds[bi].order for bi in bond_idxs]
            if orders.count(2) != 1 or any(o not in (1, 2) for o in orders):
                valid = False
                break
        if not valid:
            continue

        aromatic_bond_idxs.update(component_bonds)
    return aromatic_bond_idxs


def _extract_full_rings(mol: Molecule, atom_set: set) -> Tuple[List[List[int]], set]:
    """
    Split *atom_set* into whole rings it fully contains, plus a leftover set.

    Returns (rings, leftover): each entry in *rings* is an ordered ring-atom
    list (suitable for drawing as a polygon) whose every atom is in
    *atom_set*; *leftover* is whatever's left over (atoms not covered by any
    of those rings) — used to fall back to per-atom highlighting for a
    selection/paint that isn't a clean whole ring. Rings may share atoms or
    bonds (fused ring systems): every smallest ring around a bond of
    *atom_set* that lies entirely inside it is returned, so several
    neighbouring rings selected together each stay a ring instead of one
    swallowing its neighbours' shared atoms.
    """
    covered: set = set()
    rings: List[List[int]] = []
    seen: set = set()
    for bi, b in enumerate(mol.bonds):
        if b.i not in atom_set or b.j not in atom_set:
            continue
        ring = _find_ring_for_bond(mol, bi)
        if not ring or not all(r in atom_set for r in ring):
            continue
        key = frozenset(ring)
        if key in seen:
            continue
        seen.add(key)
        rings.append(ring)
        covered.update(ring)
    return rings, (atom_set - covered)


def _point_in_polygon(px: float, py: float, pts: List[Tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test."""
    inside = False
    n = len(pts)
    x1, y1 = pts[0]
    for k in range(1, n + 1):
        x2, y2 = pts[k % n]
        if (y1 > py) != (y2 > py):
            x_at_py = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < x_at_py:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def _annotate_double_bond_insets(mol: Molecule, positions: Positions2D, bond_info: BondInfo) -> None:
    """
    For every C=C double bond, store bond_info[bi]["inside"]: a unit vector,
    in layout space, saying which side the bond's short inner line should
    sit on. Other double bonds (C=O, C=N, ...) keep the plain symmetric
    double-line style and are left unannotated — see _draw_bond.

      * Ring bond — "inside" points from the bond toward its smallest
        containing ring's centroid (see _smallest_ring_via_edge).
      * Chain (non-ring) bond — "inside" points away from the average
        direction of the bond's other substituents, so the inner line
        doesn't cross into a neighboring group. Falls back to an arbitrary
        perpendicular when the bond has no other neighbors at all.
    """
    adjacency: Dict[int, List[Tuple[int, int]]] = {}
    for bi, b in enumerate(mol.bonds):
        adjacency.setdefault(b.i, []).append((b.j, bi))
        adjacency.setdefault(b.j, []).append((b.i, bi))

    for bi, b in enumerate(mol.bonds):
        if bond_info.get(bi, {}).get("order", b.order) != 2:
            continue
        if mol.atoms[b.i].element != "C" or mol.atoms[b.j].element != "C":
            continue
        if b.i not in positions or b.j not in positions:
            continue

        ax, ay = positions[b.i]
        bx, by = positions[b.j]
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0

        idx = idy = 0.0
        ring = _smallest_ring_via_edge(mol, b.i, b.j, exclude_bond_idx=bi)
        if ring is not None:
            ring_atoms = [i for i in ring if i in positions]
            if len(ring_atoms) >= 3:
                cxr = sum(positions[i][0] for i in ring_atoms) / len(ring_atoms)
                cyr = sum(positions[i][1] for i in ring_atoms) / len(ring_atoms)
                idx, idy = cxr - mx, cyr - my

        if math.hypot(idx, idy) < 1e-9:
            # Not a ring bond (or a degenerate one) — steer away from the
            # bond's other substituents instead.
            other_dirs = []
            for nb, obi in adjacency.get(b.i, []):
                if obi != bi and nb in positions:
                    nx, ny = positions[nb]
                    other_dirs.append((nx - ax, ny - ay))
            for nb, obi in adjacency.get(b.j, []):
                if obi != bi and nb in positions:
                    nx, ny = positions[nb]
                    other_dirs.append((nx - bx, ny - by))
            if other_dirs:
                sx = sum(d[0] for d in other_dirs) / len(other_dirs)
                sy = sum(d[1] for d in other_dirs) / len(other_dirs)
                idx, idy = -sx, -sy
            if math.hypot(idx, idy) < 1e-9:
                idx, idy = -(by - ay), (bx - ax)  # arbitrary perpendicular fallback

        ilen = math.hypot(idx, idy)
        if ilen < 1e-9:
            continue
        bond_info.setdefault(bi, {"kind": "none", "apex": b.i, "order": b.order})
        bond_info[bi]["inside"] = (idx / ilen, idy / ilen)


def _build_rdkit_mol(mol: Molecule):
    """Build a sanitized, kekulized RDKit Mol (with a 3D conformer) from *mol*."""
    from rdkit.Geometry import Point3D

    rw = Chem.RWMol()
    for a in mol.atoms:
        atom = Chem.Atom(a.element)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)

    aromatic_atoms: set = set()
    for b in mol.bonds:
        if b.order == 5:
            btype = Chem.BondType.AROMATIC
            aromatic_atoms.add(b.i)
            aromatic_atoms.add(b.j)
        else:
            btype = {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
            }.get(b.order, Chem.BondType.SINGLE)
        bidx = rw.AddBond(b.i, b.j, btype) - 1
        rw.GetBondWithIdx(bidx).SetIsAromatic(b.order == 5)

    for i in aromatic_atoms:
        rw.GetAtomWithIdx(i).SetIsAromatic(True)

    if mol.charge != 0:
        heavy = next((i for i, a in enumerate(mol.atoms) if a.element != "H"), None)
        if heavy is not None:
            rw.GetAtomWithIdx(heavy).SetFormalCharge(mol.charge)

    conf = Chem.Conformer(len(mol.atoms))
    for i, a in enumerate(mol.atoms):
        conf.SetAtomPosition(i, Point3D(float(a.x), float(a.y), float(a.z)))
    rw.AddConformer(conf, assignId=True)

    m = rw.GetMol()
    Chem.SanitizeMol(m)
    if aromatic_atoms:
        Chem.Kekulize(m, clearAromaticFlags=False)
    return m


def _rdkit_layout(mol: Molecule) -> Tuple[Positions2D, BondInfo]:
    m3d = _build_rdkit_mol(mol)
    try:
        Chem.AssignStereochemistryFrom3D(m3d)
    except Exception:
        pass

    m2d = Chem.Mol(m3d)
    if HAS_COORDGEN:
        rdCoordGen.AddCoords(m2d)
    else:
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(m2d)

    conf = m2d.GetConformer()
    try:
        Chem.WedgeMolBonds(m2d, conf)
    except Exception:
        pass

    positions: Positions2D = {}
    for i in range(m2d.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        positions[i] = (float(p.x), float(p.y))

    order_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
    }

    bond_info: BondInfo = {}
    for bi, b in enumerate(mol.bonds):
        rb = m2d.GetBondBetweenAtoms(b.i, b.j)
        kind, apex = "none", b.i
        order = 1 if b.order == 5 else b.order
        if rb is not None:
            order = order_map.get(rb.GetBondType(), order)
            d = rb.GetBondDir()
            if d == Chem.BondDir.BEGINWEDGE:
                kind, apex = "wedge", rb.GetBeginAtomIdx()
            elif d == Chem.BondDir.BEGINDASH:
                kind, apex = "hash", rb.GetBeginAtomIdx()
        bond_info[bi] = {"kind": kind, "apex": apex, "order": order}

    return positions, bond_info


def _fallback_layout(mol: Molecule, planarity_tolerance: float = PLANARITY_TOLERANCE,
                     view_rot: Optional[np.ndarray] = None, projection: bool = False,
                     plane: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[Positions2D, BondInfo]:
    """Dependency-free layout: project the real 3D geometry onto its best-fit plane."""
    coords = np.array([[a.x, a.y, a.z] for a in mol.atoms], dtype=float)
    n = len(coords)

    if n == 1:
        positions = {0: (0.0, 0.0)}
        return positions, {}

    centroid = coords.mean(axis=0)
    centered = coords - centroid

    basis_u, basis_v, normal = _plane_frame(centered, np.eye(3) if view_rot is None else view_rot)

    xs = centered @ basis_u
    ys = centered @ basis_v
    zs = centered @ normal

    bond_lengths = [
        float(np.linalg.norm(coords[b.i] - coords[b.j]))
        for b in mol.bonds
        if np.linalg.norm(coords[b.i] - coords[b.j]) > 1e-6
    ]
    median_len = float(np.median(bond_lengths)) if bond_lengths else 1.5
    unit_scale = 1.0 / median_len if median_len > 1e-6 else 1.0
    if projection:
        rot = np.eye(3) if view_rot is None else view_rot
        cam = rot @ centered.T
        zf = PERSPECTIVE_CAMERA_Z / np.where(np.abs(PERSPECTIVE_CAMERA_Z - cam[2]) < 1e-6, 1e-6, PERSPECTIVE_CAMERA_Z - cam[2])
        xs, ys = cam[0] * zf, cam[1] * zf
        unit_scale = 1.0
        if plane is not None:
            n_w, p_w = plane
            zs = (coords - p_w) @ n_w
            if (rot @ n_w)[2] < 0:
                zs = -zs   # "toward the viewer" is positive

    positions: Positions2D = {
        i: (float(xs[i]) * unit_scale, float(ys[i]) * unit_scale) for i in range(n)
    }

    kek = _kekule_orders(mol) if any(b.order == 5 for b in mol.bonds) else {}
    oop_threshold = planarity_tolerance

    bond_info: BondInfo = {}
    for bi, b in enumerate(mol.bonds):
        order = b.order
        if order == 5:
            k = (b.i, b.j) if b.i < b.j else (b.j, b.i)
            order = kek.get(k, 1)

        zi, zj = zs[b.i], zs[b.j]
        dz = zj - zi
        if abs(dz) <= oop_threshold:
            kind, apex = "none", b.i
        elif abs(zi) <= abs(zj):
            apex = b.i
            kind = "wedge" if zj > zi else "hash"
        else:
            apex = b.j
            kind = "wedge" if zi > zj else "hash"

        bond_info[bi] = {"kind": kind, "apex": apex, "order": order}

    if not projection:
        _fan_out_terminal_wedges(mol, positions, bond_info)
    return positions, bond_info


def _fan_out_terminal_wedges(mol: Molecule, positions: Positions2D, bond_info: BondInfo) -> None:
    """
    A straight plane projection often lands two purely-axial substituents
    (e.g. both H's of a CH2 whose other neighbors are all in the projection
    plane) on almost the same 2D point. Spread terminal (degree-1) wedge/hash
    substituents into the free angular sector around their anchor atom so
    they read as distinct bonds instead of overlapping lines.
    """
    degree: Dict[int, int] = {}
    adjacency: Dict[int, List[int]] = {}
    for bi, b in enumerate(mol.bonds):
        degree[b.i] = degree.get(b.i, 0) + 1
        degree[b.j] = degree.get(b.j, 0) + 1
        adjacency.setdefault(b.i, []).append(bi)
        adjacency.setdefault(b.j, []).append(bi)

    for atom_idx, bond_indices in adjacency.items():
        ax, ay = positions[atom_idx]
        neighbors = []
        for bi in bond_indices:
            b = mol.bonds[bi]
            other = b.j if b.i == atom_idx else b.i
            dx, dy = positions[other][0] - ax, positions[other][1] - ay
            angle = math.atan2(dy, dx)
            is_oop = bond_info[bi]["kind"] != "none" and bond_info[bi]["apex"] == atom_idx
            neighbors.append((other, angle, is_oop, bi))

        movable_oop = [nd for nd in neighbors if nd[2] and degree.get(nd[0], 0) == 1]
        if not movable_oop:
            continue
        anchor_angles = [nd[1] for nd in neighbors if nd not in movable_oop]

        if anchor_angles:
            base_angle = math.atan2(
                sum(math.sin(a) for a in anchor_angles),
                sum(math.cos(a) for a in anchor_angles),
            ) + math.pi
        else:
            base_angle = -math.pi / 2

        n = len(movable_oop)
        spread = math.radians(55.0)
        for k, (other, _angle, _oop, bi) in enumerate(movable_oop):
            target = base_angle if n == 1 else base_angle - spread / 2 + spread * k / (n - 1)
            length = math.hypot(positions[other][0] - ax, positions[other][1] - ay)
            if length < 1e-6:
                length = 0.9
            positions[other] = (ax + math.cos(target) * length, ay + math.sin(target) * length)


# ─────────────────────────────────────────────────────────────────────────────
# SVG RENDERING
# ─────────────────────────────────────────────────────────────────────────────
#
# SCALING STRATEGY — every size below falls into exactly one of two tiers:
#
#  MOLECULE-SCALED — grows/shrinks with the molecule's own on-screen size,
#    i.e. with `scale` (= ChemCanvas.base_scale * ChemCanvas._zoom, the 3D view's px/Angstrom). Bond
#    length itself is the obvious example; anything measured as a fraction
#    of one specific bond's rendered length also belongs here — the short
#    inner line of a C=C double bond (DOUBLE_BOND_INNER_TRIM), the shorter
#    flanking lines of a triple bond (TRIPLE_BOND_FLANK_TRIM), and the
#    wedge/hash width cap on very short bonds.
#
#  ZOOM-ONLY — grows/shrinks with the interactive zoom level alone
#    (ChemCanvas._eff_zoom: the drawn typical bond length relative to CHEM_SIZE_REF_BOND_PX), so their ratio to bond length is the same for every molecule.
#    ChemCanvas._render_to_bytes multiplies each of these by self._eff_zoom
#    before calling render_chemvector, so they stay a legible, consistent
#    size right after loading any molecule no matter how compact its
#    auto-fit turned out, and still scale together as the user zooms in/out
#    from there:
#      * letters (font_size_px) — the label boxes (_label_box) and the
#        clear space between letters and bond ends (LABEL_BOND_GAP) are
#        sized from that already-resolved font size, so they follow suit;
#      * line thickness (bond_width_px);
#      * the one spacing unit of every multi-line or stereo bond,
#        DOUBLE_BOND_GAP_PX (center-to-center distance of parallel lines):
#        double and triple bonds use it as is, a wedge/hash's wide end is
#        1.5 gaps (+ line width) across, and hash marks are one gap apart.
#        Since the gap and the line width scale together, hashes keep the
#        same clear space between marks at every zoom instead of thickening
#        into a solid-looking wedge.
SNAP_RADIUS_PX = 14.0          # build-mode snapping reach (screen pixels)
STEREO_DEPTH = 1.0             # Angstrom a wedge/hash-drawn atom sits toward/away from the viewer in 3D
DOUBLE_BOND_GAP_PX = 7.0       # ZOOM-ONLY — the spacing unit of double/triple/wedge/hash bonds
DOUBLE_BOND_INNER_TRIM = 0.12  # MOLECULE-SCALED (fraction of the bond's own length, at a bare vertex)
TRIPLE_BOND_FLANK_TRIM = 0.12  # MOLECULE-SCALED (same, for a triple bond's flanking lines)

# A "painted" atom never overrides its letter's own element color — it gets
# a low-opacity color wash instead. A whole selected/painted ring instead
# shades its interior as one filled polygon (RING_PAINT_OPACITY); that
# polygon, like selection's per-atom highlight, is layered (in paint_group)
# behind bonds and atoms so it reads as a background highlight rather than
# tinting the structure drawn on top of it — selection just uses a much
# higher opacity than paint so it still can't be missed. A single painted
# atom's own wash is the one exception: it sits (in mask_group) in its
# label's box, right below the letter — bonds stop short of that box, so the
# wash shows cleanly around the letter instead of hiding under bond lines.
SELECTION_COLOR = "#2f8fff"      # selection stays blue
DEFAULT_PAINT_COLOR = "#e63946"   # paint defaults to a soft red, so it never reads as a selection
PAINT_OPACITY = 0.55
RING_PAINT_OPACITY = 0.30


Box = Tuple[float, float, float, float]   # (x0, x1, y0, y1) relative to the atom's center


def _label_box(elem: str, font_size: float, n_h: int = 0, h_side: str = "right") -> Box:
    """
    The one rectangular box an atom's whole label occupies — its element
    letters, plus (for a carbon with hidden hydrogens) an "H" and a
    subscript count in the same box. The element letter(s) stay centered on
    the atom itself, where its bonds attach; the H part goes to *h_side*:
    "right" ("CH3"), "left" ("H3C"), or "down"/"up" (stacked under/over the
    C) when bonds crowd the sides. Used for the background padding and for
    trimming wedge/hash bonds clear of the text.
    """
    fs = font_size
    pad = 0.12 * fs
    half_e = 0.34 * fs * len(elem)
    x0, x1 = -half_e - pad, half_e + pad
    y0, y1 = -0.46 * fs, 0.46 * fs
    if n_h > 0:
        row_w = 0.68 * fs + (0.44 * fs * len(str(n_h)) if n_h > 1 else 0.0)
        if h_side in ("right", "left"):
            if h_side == "left":
                x0 -= row_w
            else:
                x1 += row_w
            if n_h > 1:
                y1 = 0.62 * fs
        else:
            half_r = row_w / 2 + pad
            x0, x1 = min(x0, -half_r), max(x1, half_r)
            if h_side == "down":
                y1 = _H_ROW_DY * fs + (0.62 if n_h > 1 else 0.46) * fs
            else:
                y0 = -_H_ROW_DY * fs - 0.46 * fs
    return (x0, x1, y0, y1)


_H_ROW_DY = 0.92   # line spacing (in font sizes) of a stacked "H" row


def _pick_h_side(angles: List[float]) -> str:
    """
    Where a carbon's "H"/"Hn" goes given the canvas angles (radians, y down)
    of its visible bonds: right by default, left if the right is crowded,
    otherwise above/below — whichever direction leaves the most angular
    clearance from the bonds.
    """
    if not angles:
        return "right"

    def clearance(direction: float) -> float:
        return min(abs((a - direction + math.pi) % (2 * math.pi) - math.pi) for a in angles)

    cands = [("right", 0.0), ("left", math.pi), ("down", math.pi / 2), ("up", -math.pi / 2)]
    for name, d in cands[:2]:
        if clearance(d) >= math.radians(40):
            return name
    return max(cands, key=lambda c: clearance(c[1]) + (1e-6 if c[0] == "down" else 0.0))[0]


def _label_svg_text(dwg, elem: str, n_h: int, h_side: str, x: float, y: float,
                    font_size: float, font_family: str, fill: str, bold: bool = True) -> list:
    """The label's text element(s): the element, then "H" and a subscript count."""
    base_y = y + font_size * 0.35
    kw = dict(font_size=font_size, font_family=font_family,
              font_weight="bold" if bold else "normal", fill=fill)
    if n_h <= 0:
        return [dwg.text(elem, insert=(x, base_y), text_anchor="middle", **kw)]
    sub_size = font_size * 0.7
    sub_dy = font_size * 0.22
    if h_side in ("down", "up"):
        row_y = base_y + (_H_ROW_DY if h_side == "down" else -_H_ROW_DY) * font_size
        row = dwg.text("", insert=(x, row_y), text_anchor="middle", **kw)
        row.add(dwg.tspan("H"))
        if n_h > 1:
            row.add(dwg.tspan(str(n_h), font_size=sub_size, dy=[sub_dy]))
        return [dwg.text(elem, insert=(x, base_y), text_anchor="middle", **kw), row]
    half_e = 0.34 * font_size * len(elem)
    if h_side == "left":
        t = dwg.text("", insert=(x + half_e, base_y), text_anchor="end", **kw)
        t.add(dwg.tspan("H"))
        if n_h > 1:
            t.add(dwg.tspan(str(n_h), font_size=sub_size, dy=[sub_dy]))
            t.add(dwg.tspan(elem, dy=[-sub_dy]))
        else:
            t.add(dwg.tspan(elem))
    else:
        t = dwg.text("", insert=(x - half_e, base_y), text_anchor="start", **kw)
        t.add(dwg.tspan(elem))
        t.add(dwg.tspan("H"))
        if n_h > 1:
            t.add(dwg.tspan(str(n_h), font_size=sub_size, dy=[sub_dy]))
    return [t]


def _blend_hex(base: str, overlay: str, alpha: float) -> str:
    """*overlay* at *alpha* opacity composited over the opaque *base* color."""
    try:
        br, bg, bb = (int(base.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        orr, og, ob = (int(overlay.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        return base
    r = round(br * (1 - alpha) + orr * alpha)
    g = round(bg * (1 - alpha) + og * alpha)
    b = round(bb * (1 - alpha) + ob * alpha)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hydrogen_is_implicit(i: int, mol: Molecule, bond_info: BondInfo) -> bool:
    """
    Whether hydrogen atom *i* should be hidden in implicit-hydrogen mode.

    Only a hydrogen bonded exclusively to carbon is implicit — one on a
    heteroatom (O-H, N-H, ...) stays explicit, per organic-chemistry
    convention. A hydrogen carrying a wedge/hash (perspective) bond also
    stays explicit, but only when it could actually be conveying real
    stereochemistry — i.e. its carbon carries exactly one hydrogen (so it's
    a genuine potential stereocenter). A carbon with two or more hydrogens
    (a plain CH2 or CH3) can never be a stereocenter, so a wedge/hash there
    is purely cosmetic (from perceived 3D shape, not configuration) and
    hiding it just falls back to the normal implicit-CH2/CH3 convention —
    exactly like any other saturated ring carbon, with no dangling H atoms
    or wedge/hash lines drawn.
    """
    if mol.atoms[i].element != "H":
        return False

    own_bonds = [(bi, b) for bi, b in enumerate(mol.bonds) if b.i == i or b.j == i]
    if not own_bonds:
        return False  # isolated H atom — nothing to hide it from, keep visible

    neighbor_idx = [b.j if b.i == i else b.i for bi, b in own_bonds]
    if not all(mol.atoms[n].element == "C" for n in neighbor_idx):
        return False

    carbon_idx = neighbor_idx[0]
    h_siblings = sum(
        1 for b in mol.bonds
        if (b.i == carbon_idx and mol.atoms[b.j].element == "H")
        or (b.j == carbon_idx and mol.atoms[b.i].element == "H")
    )
    if h_siblings == 1:
        for bi, b in own_bonds:
            if bond_info.get(bi, {}).get("kind", "none") in ("wedge", "hash"):
                return False

    return True


# Clear space between a label's letters and the end of any bond reaching it,
# in font sizes — the same for every bond line, whatever its angle or type.
LABEL_BOND_GAP = 0.16

# A label's keep-out zone: the letters' own rectangle (canvas coordinates
# x0, x1, y0, y1) grown by a radius r, i.e. a rounded rectangle. Bond lines
# are clipped to stop at its edge, so the gap to the text is the same in
# every direction (a plain box would leave bigger gaps at its corners).
KeepOut = Tuple[float, float, float, float, float]


def _label_keepout(x: float, y: float, box: Box, font_size: float, lw: float) -> KeepOut:
    pad = 0.12 * font_size   # the padding _label_box adds around the letters
    x0, x1, y0, y1 = box
    return (x + x0 + pad, x + x1 - pad, y + y0 + pad, y + y1 - pad,
            LABEL_BOND_GAP * font_size + lw / 2.0)   # + half the stroke: round caps reach that far


def _keepout_excess(px: float, py: float, ko: KeepOut) -> float:
    """Distance from (px, py) to the keep-out's core rectangle, minus its radius (<= 0: inside)."""
    dx = max(ko[0] - px, 0.0, px - ko[1])
    dy = max(ko[2] - py, 0.0, py - ko[3])
    return math.hypot(dx, dy) - ko[4]


def _keepout_interval(p0, p1, ko: KeepOut) -> Optional[Tuple[float, float]]:
    """
    The part [t0, t1] of segment p0→p1 (t in 0..1) inside keep-out *ko*, or
    None. The distance to a convex shape is convex along a line, so its
    minimum is found by ternary search and both crossings by bisection.
    """
    def f(t):
        return _keepout_excess(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t, ko)

    lo, hi = 0.0, 1.0
    for _ in range(40):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if f(m1) <= f(m2):
            hi = m2
        else:
            lo = m1
    tm = (lo + hi) / 2
    if f(tm) > 0:
        return None

    def crossing(outside, inside):
        for _ in range(30):
            mid = (outside + inside) / 2
            if f(mid) > 0:
                outside = mid
            else:
                inside = mid
        return (outside + inside) / 2

    t0 = 0.0 if f(0.0) <= 0 else crossing(0.0, tm)
    t1 = 1.0 if f(1.0) <= 0 else crossing(1.0, tm)
    return t0, t1


def _clip_to_keepouts(p0, p1, ko_start: Optional[KeepOut], ko_end: Optional[KeepOut]):
    """
    Segment p0→p1 shortened so it starts outside *ko_start* (the label at
    its start) and ends outside *ko_end*; None if nothing is left.
    """
    s, e = 0.0, 1.0
    if ko_start is not None:
        iv = _keepout_interval(p0, p1, ko_start)
        if iv is not None and iv[0] < 0.5:
            s = iv[1]
    if ko_end is not None:
        iv = _keepout_interval(p0, p1, ko_end)
        if iv is not None and iv[1] > 0.5:
            e = iv[0]
    if e - s < 1e-3:
        return None
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return (p0[0] + dx * s, p0[1] + dy * s), (p0[0] + dx * e, p0[1] + dy * e)


def _draw_bond(dwg, group, ax, ay, bx, by, order: int, kind: str, apex_is_i: bool,
               bond_color: str, lw: float, gap: float,
               inside_canvas: Optional[Tuple[float, float]] = None,
               is_cc: bool = False,
               ko_i: Optional[KeepOut] = None,
               ko_j: Optional[KeepOut] = None):
    """
    Draw one bond from atom i (ax, ay) to atom j (bx, by). Every line is
    clipped at the keep-out of a labeled end (ko_i / ko_j), so all bonds stop
    the same distance from the letters. All spacing derives from one unit,
    *gap* — the center-to-center distance of parallel lines:
      * double bond: two lines *gap* apart (C=C: the second one inset);
      * triple bond: three lines *gap* apart;
      * wedge / hash: the wide end is 1.5 gaps across (+ the line width),
        and hash marks are one gap apart, like a double bond's two lines.
    """
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux

    def line(p0, p1, width=lw):
        seg = _clip_to_keepouts(p0, p1, ko_i, ko_j)
        if seg is not None:
            group.add(dwg.line(start=seg[0], end=seg[1], stroke=bond_color,
                                stroke_width=width, stroke_linecap="round"))

    def offset_line(off, trim_frac):
        # A parallel line *off* to the side, pulled in by trim_frac of the bond
        # at a bare vertex (so it never runs into the neighbors' bonds); at a
        # labeled end it is clipped by the keep-out like any other line.
        ti = 0.0 if ko_i is not None else length * trim_frac
        tj = 0.0 if ko_j is not None else length * trim_frac
        line((ax + ux * ti + px * off, ay + uy * ti + py * off),
             (bx - ux * tj + px * off, by - uy * tj + py * off))

    if kind in ("wedge", "hash"):
        if apex_is_i:
            apex_pt, base_pt, ko_apex, ko_base = (ax, ay), (bx, by), ko_i, ko_j
        else:
            apex_pt, base_pt, ko_apex, ko_base = (bx, by), (ax, ay), ko_j, ko_i
        if ko_apex is not None:
            seg = _clip_to_keepouts(apex_pt, base_pt, ko_apex, None)
            if seg is None:
                return
            apex_pt = seg[0]
        adx, ady = base_pt[0] - apex_pt[0], base_pt[1] - apex_pt[1]
        alen = math.hypot(adx, ady)
        if alen < 1e-3:
            return
        aux, auy = adx / alen, ady / alen
        apx, apy = -auy, aux
        half_w = min(0.75 * gap + lw / 2.0, alen * 0.2)
        # Shorten the whole tapered shape (not just its axis) so even the
        # wide end's corners stay out of a labeled base atom's keep-out.
        reach = 1.0
        if ko_base is not None:
            for side in (0.0, 1.0, -1.0):
                corner = (base_pt[0] + apx * half_w * side, base_pt[1] + apy * half_w * side)
                iv = _keepout_interval(apex_pt, corner, ko_base)
                if iv is not None:
                    reach = min(reach, iv[0])
        if reach < 0.05:
            return
        w_end = half_w * reach
        end = (apex_pt[0] + adx * reach, apex_pt[1] + ady * reach)
        if kind == "wedge":
            group.add(dwg.polygon(points=[apex_pt,
                                          (end[0] + apx * w_end, end[1] + apy * w_end),
                                          (end[0] - apx * w_end, end[1] - apy * w_end)],
                                  fill=bond_color, stroke="none"))
        else:
            # Evenly spaced marks, one gap apart (like a double bond's lines), widening toward the end.
            n_hash = max(2, round(alen * reach / gap))
            for k in range(1, n_hash + 1):
                t = k / n_hash
                w = w_end * t
                hx, hy = apex_pt[0] + (end[0] - apex_pt[0]) * t, apex_pt[1] + (end[1] - apex_pt[1]) * t
                group.add(dwg.line(start=(hx - apx * w, hy - apy * w), end=(hx + apx * w, hy + apy * w),
                                    stroke=bond_color, stroke_width=lw * 0.8, stroke_linecap="round"))
        return

    if order not in (1, 2, 3):
        order = 1

    if order == 1:
        line((ax, ay), (bx, by))
        return

    if order == 2 and is_cc:
        # C=C: the endpoint-to-endpoint line carries the connectivity; the
        # second line is shorter, inset toward the ring interior (or away
        # from the other substituents on a chain).
        line((ax, ay), (bx, by))
        if inside_canvas is not None and (inside_canvas[0] or inside_canvas[1]):
            inside_sign = 1.0 if (px * inside_canvas[0] + py * inside_canvas[1]) >= 0 else -1.0
        else:
            inside_sign = 1.0
        offset_line(inside_sign * gap, DOUBLE_BOND_INNER_TRIM)
        return

    if order == 2:
        # Any other double bond (C=O, C=N, N=O, S=O, ...): two full-length
        # lines centered on the bond.
        offset_line(-gap / 2.0, 0.0)
        offset_line(gap / 2.0, 0.0)
        return

    # Triple bond: the center line carries the connectivity, flanked by two
    # lines one gap away on each side.
    line((ax, ay), (bx, by))
    offset_line(-gap, TRIPLE_BOND_FLANK_TRIM)
    offset_line(gap, TRIPLE_BOND_FLANK_TRIM)


def render_chemvector(
    mol: Molecule,
    positions: Positions2D,
    bond_info: BondInfo,
    canvas_w: int = 700,
    canvas_h: int = 600,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
    scale: float = 90.0,
    background: str = "#ffffff",
    color_overrides: Optional[Dict[str, str]] = None,
    bond_color: str = "#2b2b2b",
    bond_width_px: float = 3.4,
    double_bond_gap_px: float = DOUBLE_BOND_GAP_PX,
    font_family: str = "Helvetica, Arial, sans-serif",
    font_size_px: float = 26.0,
    show_carbon_labels: bool = False,
    show_hydrogens: bool = False,
    label_color_mode: str = "cpk",
    selected_indices: Optional[set] = None,
    paint_colors: Optional[Dict[int, str]] = None,
    aromatic_bond_idxs: Optional[set] = None,
    export_mode: bool = False,
    center: Optional[Tuple[float, float]] = None,
    ring_bonds: Optional[List[dict]] = None,
    label_boxes_out: Optional[Dict[int, Box]] = None,
) -> str:
    """
    Render *mol* (with a precomputed 2D layout) as an SVG string.

    *scale* sets the molecule's own on-screen size (bond length, and
    anything measured as a fraction of a specific bond) — see the "SCALING
    STRATEGY" note above _label_box. *bond_width_px*, *font_size_px* and
    *double_bond_gap_px* are all taken as-is: the caller
    (ChemCanvas) is expected to have already resolved them against the
    interactive zoom level, independent of *scale*.

    *center*, in layout space, is what gets drawn at the canvas center
    (before pan). Left as None, it defaults to *positions*' own bounding-box
    center — fine for a one-off render, but wrong for an interactive canvas:
    recomputing it from the live bounding box on every frame means the
    whole drawing recenters itself around whatever was just added or moved
    (a single newly-placed atom is its own bounding box, so it would jump
    straight to the canvas center). ChemCanvas instead freezes a *center*
    once per fit/reset and passes that same value on every render, so
    editing the molecule doesn't shift anything already on screen.

    When show_carbon_labels/show_hydrogens are False (the default), carbons
    are drawn as unlabeled vertices (bonds simply meet at their position)
    and hydrogens are omitted entirely along with their bonds — standard
    organic-chemistry skeletal-formula shorthand. A carbon whose valence
    isn't 4 (over 4 always; under 4 whenever the molecule has explicit H
    atoms, e.g. a dangling valence-1 C) keeps its letter, since a bare
    vertex would suggest implicit hydrogens that aren't there. A carbon
    with no visible bonds at all is also labeled so it doesn't vanish.

    *label_color_mode* — "cpk" draws each atom letter in its element's CPK
    color; "black" draws all letters in one plain color (black, or white on a
    dark background).

    *paint_colors* ({atom_idx: hex_color}) is the 2D view's own paint layer —
    entirely separate from Atom.color (which only the 3D view reads/writes),
    so painting in one view never touches the other. Whenever every atom of
    one ring is selected (or painted the same color), that whole ring is
    shaded as a single filled polygon instead of per-atom dots.

    *ring_bonds* ({"ring": [atom indices], "atom": idx}) are drawn as a line
    from the ring's center out to that atom — a substituent whose ring
    position (ortho/meta/para) is left undetermined.

    Each atom's label uses its own Atom.label_color / Atom.label_bold when
    set. *label_boxes_out*, if given, is filled with every drawn label's
    canvas box (x0, x1, y0, y1) by atom index — for hit-testing labels.
    """
    dwg = svgwrite.Drawing("__mem__", size=(canvas_w, canvas_h))
    if not export_mode:
        dwg.add(dwg.rect(insert=(0, 0), size=(canvas_w, canvas_h), fill=background))

    if not mol.atoms or not positions:
        return dwg.tostring()

    if center is not None:
        ccx, ccy = center
    elif positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        ccx = (min(xs) + max(xs)) / 2.0
        ccy = (min(ys) + max(ys)) / 2.0
    else:
        ccx = ccy = 0.0
    cx = canvas_w / 2 + pan_x
    cy = canvas_h / 2 + pan_y

    def to_canvas_xy(x: float, y: float) -> Tuple[float, float]:
        return (cx + (x - ccx) * scale, cy - (y - ccy) * scale)

    def to_canvas(i: int) -> Tuple[float, float]:
        return to_canvas_xy(*positions[i])

    base_colors = dict(CPK_BASE)
    if color_overrides:
        base_colors.update(color_overrides)

    # ZOOM-ONLY — all pre-resolved by the caller against ChemCanvas._eff_zoom
    lw = max(1.1, bond_width_px)
    font_size = max(6.0, font_size_px)
    double_gap = double_bond_gap_px

    visible = {i for i in range(len(mol.atoms)) if i in positions}
    if not show_hydrogens:
        visible = {i for i in visible if not _hydrogen_is_implicit(i, mol, bond_info)}

    visible_degree: Dict[int, int] = {i: 0 for i in visible}
    for bond in mol.bonds:
        if bond.i in visible and bond.j in visible:
            visible_degree[bond.i] += 1
            visible_degree[bond.j] += 1

    # Ring bonds whose ring and substituent are all on screen, with the
    # ring's canvas center. They count as a real bond on the substituent
    # (degree, valence, H placement), so e.g. a bare methyl stays a line end.
    shown_ring_bonds: List[Tuple[int, Tuple[float, float]]] = []
    for rb in ring_bonds or []:
        ring, a = rb.get("ring", []), rb.get("atom")
        if a not in visible or len(ring) < 3 or not all(r in visible for r in ring):
            continue
        pts = [to_canvas(r) for r in ring]
        shown_ring_bonds.append((a, (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))))
        visible_degree[a] += 1

    # A carbon can only be a bare implicit vertex when its valence checks out.
    # Over 4 (invalid structure) always keeps its "C" letter. Under 4 depends
    # on whether hydrogens are known: when the molecule carries explicit H
    # atoms (loaded files, the 3D builder's auto-H), they are all accounted
    # for, so a carbon that doesn't total exactly 4 (e.g. a dangling
    # valence-1 C) is drawn with its letter — a bare line end would wrongly
    # read as CH3. In a heavy-atom-only sketch (2D build mode adds no H
    # atoms) the missing valence is by convention implicit H, so under-4 is
    # fine there. Order-5 (aromatic) bonds count as 1.5.
    explicit_h_known = any(a.element == "H" for a in mol.atoms)
    valence_sum: Dict[int, float] = {}
    for b in mol.bonds:
        o = 1.5 if b.order == 5 else b.order
        valence_sum[b.i] = valence_sum.get(b.i, 0) + o
        valence_sum[b.j] = valence_sum.get(b.j, 0) + o
    for a, _center in shown_ring_bonds:
        valence_sum[a] = valence_sum.get(a, 0) + 1

    # Needed up front (not just in the atom-drawing loop below) so wedge/hash
    # bonds can trim themselves clear of whichever end has a visible label.
    # Hydrogens hidden as implicit (bonded only to carbon) are folded into
    # their carbon's label when it has one: "CH", "CH2", "CH3" — the count as
    # a subscript in the same text box. The H part goes on whichever side
    # (right/left/below/above) the carbon's bonds leave most free.
    hidden_h_count: Dict[int, int] = {}
    bond_angles: Dict[int, List[float]] = {}
    for b in mol.bonds:
        for a, o in ((b.i, b.j), (b.j, b.i)):
            if a not in visible:
                continue
            if o not in visible and mol.atoms[o].element == "H" and mol.atoms[a].element == "C":
                hidden_h_count[a] = hidden_h_count.get(a, 0) + 1
            elif o in visible:
                ax_, ay_ = to_canvas(a)
                ox_, oy_ = to_canvas(o)
                d = math.hypot(ox_ - ax_, oy_ - ay_)
                if d > 1e-6:
                    bond_angles.setdefault(a, []).append(math.atan2(oy_ - ay_, ox_ - ax_))
    for a, (rcx, rcy) in shown_ring_bonds:
        ax_, ay_ = to_canvas(a)
        bond_angles.setdefault(a, []).append(math.atan2(rcy - ay_, rcx - ax_))

    label_box_by_atom: Dict[int, Box] = {}
    label_h_side: Dict[int, str] = {}
    for i in visible:
        elem = mol.atoms[i].element
        v = valence_sum.get(i, 0)
        bad_valence = elem == "C" and (v > 4.6 or (explicit_h_known and abs(v - 4) > 0.6))
        is_implicit_carbon = (elem == "C" and not show_carbon_labels and not bad_valence)
        needs_label = (not is_implicit_carbon) or visible_degree.get(i, 0) == 0
        if needs_label:
            label_h_side[i] = _pick_h_side(bond_angles.get(i, []))
            label_box_by_atom[i] = _label_box(
                elem, font_size, hidden_h_count.get(i, 0), label_h_side[i])

    # A ring that's fully aromatic — genuinely order 5 on mol, or flagged in
    # the caller's own aromatic_bond_idxs overlay (Check Aromaticity, which
    # never touches mol.bonds/the 3D view) — is drawn with a circle in its
    # interior instead of Kekulized double-bond lines; those bonds render as
    # plain single lines since the circle alone conveys the aromaticity.
    aromatic_rings = [
        ring for ring in _find_aromatic_rings(mol, aromatic_bond_idxs)
        if all(a in visible for a in ring)
    ]
    aromatic_ring_bond_idxs: set = set()
    if aromatic_rings:
        pair_to_bi = {(b.i, b.j): bi for bi, b in enumerate(mol.bonds)}
        pair_to_bi.update({(b.j, b.i): bi for bi, b in enumerate(mol.bonds)})
        for ring in aromatic_rings:
            n = len(ring)
            for k in range(n):
                aromatic_ring_bond_idxs.add(pair_to_bi[(ring[k], ring[(k + 1) % n])])

    keepout: Dict[int, KeepOut] = {
        i: _label_keepout(*to_canvas(i), box, font_size, lw) for i, box in label_box_by_atom.items()
    }

    paint_group = dwg.g(id="chem_paint")  # added first — paint sits behind bonds and atoms
    bonds_group = dwg.g(id="chem_bonds")
    mask_group = dwg.g(id="chem_label_tints")
    text_group = dwg.g(id="chem_labels")

    for bi, bond in enumerate(mol.bonds):
        if bond.i not in visible or bond.j not in visible:
            continue
        info = bond_info.get(bi, {"kind": "none", "apex": bond.i, "order": bond.order})
        ax, ay = to_canvas(bond.i)
        bx, by = to_canvas(bond.j)
        order = 1 if bi in aromatic_ring_bond_idxs else info.get("order", bond.order)
        if order == 5:
            order = 1

        inside_canvas = None
        layout_inside = info.get("inside")
        if layout_inside is not None:
            # to_canvas mirrors y (chem-space up vs. SVG-space down); a
            # direction vector transforms the same way, minus translation.
            inside_canvas = (layout_inside[0], -layout_inside[1])

        is_cc = mol.atoms[bond.i].element == "C" and mol.atoms[bond.j].element == "C"

        _draw_bond(
            dwg, bonds_group, ax, ay, bx, by,
            order=order, kind=info.get("kind", "none"),
            apex_is_i=(info.get("apex", bond.i) == bond.i),
            bond_color=bond_color, lw=lw, gap=double_gap,
            inside_canvas=inside_canvas, is_cc=is_cc,
            ko_i=keepout.get(bond.i), ko_j=keepout.get(bond.j),
        )

    for a, (rcx, rcy) in shown_ring_bonds:
        # Plain line from the ring center, stopping short of the substituent's label.
        seg = _clip_to_keepouts((rcx, rcy), to_canvas(a), None, keepout.get(a))
        if seg is not None:
            bonds_group.add(dwg.line(start=seg[0], end=seg[1], stroke=bond_color,
                                      stroke_width=lw, stroke_linecap="round"))

    for ring in aromatic_rings:
        pts = [to_canvas(i) for i in ring]
        cx_r = sum(p[0] for p in pts) / len(pts)
        cy_r = sum(p[1] for p in pts) / len(pts)
        n = len(pts)
        # Radius from the nearest edge midpoint (the ring's apothem), pulled
        # in a bit so the circle sits clear of the bond lines around it.
        apothem = min(
            math.hypot((pts[k][0] + pts[(k + 1) % n][0]) / 2 - cx_r,
                       (pts[k][1] + pts[(k + 1) % n][1]) / 2 - cy_r)
            for k in range(n)
        )
        bonds_group.add(dwg.circle(center=(cx_r, cy_r), r=apothem * 0.78, fill="none",
                                    stroke=bond_color, stroke_width=lw * 0.7))

    # Whenever a whole ring is selected (or painted one solid color), shade
    # its interior as a single filled polygon instead of per-atom dots; only
    # atoms left over from that (a partial/non-ring selection or paint) fall
    # back to the per-atom treatment below. A ring-member atom's own label
    # padding is deliberately left untinted here (see pad_fill below) — a
    # ring selection should read as "this ring is selected", not also as
    # "and this one atom on it is individually selected".
    sel_rings: List[List[int]] = []
    sel_leftover: set = set()
    if selected_indices:
        sel_rings, sel_leftover = _extract_full_rings(mol, set(selected_indices) & visible)
    for ring in sel_rings:
        pts = [to_canvas(i) for i in ring]
        paint_group.add(dwg.polygon(points=pts, fill=SELECTION_COLOR,
                                     opacity=RING_PAINT_OPACITY, stroke="none"))

    paint_ring_atoms: set = set()
    paint_ring_color_by_atom: Dict[int, str] = {}
    paint_leftover: Dict[int, str] = {}
    if paint_colors:
        by_color: Dict[str, set] = {}
        for idx, col in paint_colors.items():
            if idx in visible:
                by_color.setdefault(col, set()).add(idx)
        for col, idxs in by_color.items():
            rings, leftover = _extract_full_rings(mol, idxs)
            for ring in rings:
                pts = [to_canvas(i) for i in ring]
                paint_group.add(dwg.polygon(points=pts, fill=col, opacity=RING_PAINT_OPACITY, stroke="none"))
                paint_ring_atoms.update(ring)
                for a in ring:
                    paint_ring_color_by_atom[a] = col
            for idx in leftover:
                paint_leftover[idx] = col

    try:
        bg = background.lstrip("#")
        lum = (0.299 * int(bg[0:2], 16) + 0.587 * int(bg[2:4], 16) + 0.114 * int(bg[4:6], 16)) / 255.0
    except (ValueError, IndexError):
        lum = 1.0
    mono_label_color = "#000000" if lum >= 0.4 else "#ffffff"
    for i, (bx0, bx1, by0, by1) in label_box_by_atom.items():
        elem = mol.atoms[i].element
        x, y = to_canvas(i)
        # The letter always keeps its own element color — painting an atom
        # never overrides it; see the low-opacity wash below instead.
        text_color = mono_label_color if label_color_mode == "black" else base_colors.get(elem, DEFAULT_BASE)
        atom = mol.atoms[i]
        if getattr(atom, "label_color", None):
            text_color = atom.label_color
        if label_boxes_out is not None:
            label_boxes_out[i] = (x + bx0, x + bx1, y + by0, y + by1)

        # Bonds already stop short of the letters (see _label_keepout), so
        # nothing needs erasing here: a label gets a box behind it only to
        # show an individual ("leftover") selection — a ring-selected atom is
        # shown by the ring polygon alone (see the note above sel_rings).
        if i in sel_leftover:
            mask_group.add(dwg.rect(insert=(x + bx0, y + by0), size=(bx1 - bx0, by1 - by0),
                                    rx=0.23 * font_size, ry=0.23 * font_size,
                                    fill=_blend_hex(background, SELECTION_COLOR, 0.6), stroke="none"))

        # The paint wash goes on top of that, then the letter on top of the paint.
        paint_col = paint_leftover.get(i)
        if paint_col and i not in paint_ring_atoms:
            mask_group.add(dwg.rect(insert=(x + bx0, y + by0), size=(bx1 - bx0, by1 - by0),
                                    rx=0.23 * font_size, ry=0.23 * font_size, fill=paint_col,
                                    opacity=PAINT_OPACITY, stroke="none"))

        for el in _label_svg_text(
            dwg, elem, hidden_h_count.get(i, 0), label_h_side.get(i, "right"),
            x, y, font_size, font_family, text_color,
            bold=getattr(atom, "label_bold", None) is not False,
        ):
            text_group.add(el)

    for i in visible:
        if i in label_box_by_atom or i not in positions:
            continue
        # An implicit (unlabeled) vertex never goes through the loop above —
        # still show its paint wash / selection dot at a small fixed radius
        # around the bare vertex point (when not already covered by a ring
        # polygon above).
        is_selected_leftover = i in sel_leftover
        paint_col = paint_leftover.get(i) if i not in paint_ring_atoms else None
        if not (is_selected_leftover or paint_col):
            continue
        x, y = to_canvas(i)
        vertex_r = max(7.0, font_size * 0.3)
        if is_selected_leftover:
            paint_group.add(dwg.circle(center=(x, y), r=vertex_r, fill=SELECTION_COLOR,
                                        opacity=0.9, stroke="none"))
        if paint_col:
            # Under the bond lines here (an implicit vertex has no label box
            # keeping bonds away), so sized generously to peek out around them.
            paint_group.add(dwg.circle(center=(x, y), r=vertex_r * 1.3, fill=paint_col,
                                        opacity=PAINT_OPACITY, stroke="none"))

    dwg.add(paint_group)
    dwg.add(bonds_group)
    dwg.add(mask_group)
    dwg.add(text_group)


    if not export_mode:
        dwg.add(dwg.text(
            mol.name, insert=(16, 26), font_size=15, font_family="monospace",
            fill="#88ccff" if background.lower() in ("#0a0a12", "#000000") else "#7799bb",
            opacity=0.75,
        ))

    return dwg.tostring()


# ─────────────────────────────────────────────────────────────────────────────
# QT WIDGET
# ─────────────────────────────────────────────────────────────────────────────

def _apply_drawn_stereo(mol: Molecule, bond_info: BondInfo) -> None:
    """Bonds drawn as wedge/hash in 2D build mode keep that look over whatever the layout perceived."""
    for bi, b in enumerate(mol.bonds):
        stereo = getattr(b, "stereo", None)
        if stereo in ("wedge", "hash"):
            bond_info[bi] = {"kind": stereo, "apex": b.i, "order": 1}


def _rot_z(a: float) -> np.ndarray:
    """3D rotation about the viewing axis by *a* radians (counter-clockwise, y up)."""
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


class _LabelEdit(QLineEdit):
    """Line edit of the label editor: Enter commits, Escape cancels (focus-out
    does not — the color dialog takes focus while the edit is open)."""

    cancelled = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            return
        super().keyPressEvent(event)


class _TextEditor(QFrame):
    """
    Inline editor for an atom label (Text tool): the text field, with a
    minimal toolbar right above it (color swatch, bold toggle). Enter
    commits, Escape cancels; clicking elsewhere on the canvas commits too
    (ChemCanvas does that — not focus-out, since the color dialog takes focus).
    """

    committed = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, text: str, color: str, bold: bool, font: QFont, parent=None):
        super().__init__(parent)
        self.color = color
        self._invalid = False
        self.setObjectName("chemTextEditor")
        self.setStyleSheet(
            "#chemTextEditor { background: transparent; }"
            "QToolButton { background: #ffffff; color: #222; border: 1px solid #c8ccd2; border-radius: 4px;"
            " min-width: 22px; min-height: 20px; padding: 0 3px; font-weight: bold; }"
            "QToolButton:checked { background: #dbe9ff; border-color: #2f8fff; }")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)
        bar = QHBoxLayout()
        bar.setSpacing(3)
        self._color_btn = QToolButton()
        self._color_btn.setToolTip("Text color")
        self._color_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._color_btn.clicked.connect(self._pick_color)
        self._bold_btn = QToolButton()
        self._bold_btn.setText("B")
        self._bold_btn.setToolTip("Bold")
        self._bold_btn.setCheckable(True)
        self._bold_btn.setChecked(bold)
        self._bold_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._bold_btn.toggled.connect(self._restyle)
        bar.addWidget(self._color_btn)
        bar.addWidget(self._bold_btn)
        bar.addStretch()
        outer.addLayout(bar)
        self.edit = _LabelEdit(text)
        self.edit.setFont(font)
        self.edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edit.cancelled.connect(self.cancelled)
        self.edit.returnPressed.connect(self.committed)
        self.edit.textChanged.connect(lambda _t: self.set_invalid(False))
        outer.addWidget(self.edit)
        self._restyle()

    @property
    def bold(self) -> bool:
        return self._bold_btn.isChecked()

    def set_invalid(self, invalid: bool):
        self._invalid = invalid
        self._restyle()

    def _pick_color(self):
        col = QColorDialog.getColor(QColor(self.color), self, "Text color")
        if col.isValid():
            self.color = col.name()
            self._restyle()
        self.edit.setFocus()

    def _restyle(self, *_):
        self._color_btn.setStyleSheet(
            f"QToolButton {{ background: {self.color}; border: 2px solid #ffffff; outline: 1px solid #c8ccd2; }}")
        font = self.edit.font()
        font.setBold(self.bold)
        self.edit.setFont(font)
        border, bg = ("#e03131", "#fff0f0") if self._invalid else ("#2f8fff", "white")
        self.edit.setStyleSheet(
            f"QLineEdit {{ border: 2px solid {border}; border-radius: 4px; background: {bg}; color: {self.color}; }}")
        self._fit_width()

    def _fit_width(self, *_):
        fm = QFontMetricsF(self.edit.font())
        w = max(90, int(fm.horizontalAdvance(self.edit.text() or "M") + 30))
        self.edit.setFixedWidth(w)
        self.adjustSize()
        center = self.property("anchor_center")
        if center is not None:
            x = int(center[0] - self.width() / 2)
            y = int(center[1] - self.height() + self.edit.height() / 2)
            parent = self.parentWidget()
            if parent is not None:   # keep it fully on the canvas near the edges
                x = max(2, min(x, parent.width() - self.width() - 2))
                y = max(2, min(y, parent.height() - self.height() - 2))
            self.move(x, y)


class ChemCanvas(QSvgWidget):
    """2D 'whiteboard' molecule view — pan/zoom, plus a build mode for sketching directly in 2D."""

    fileDropped = pyqtSignal(str)
    layoutRegenerated = pyqtSignal()
    moleculeChanged = pyqtSignal()      # emitted after atoms/bonds change
    zoomChanged = pyqtSignal(float)     # zoom multiplier changed here (linked with the 3D view's)
    viewChanged = pyqtSignal(object)    # new 3D camera rotation (3x3) after the user turned the depiction here
    requestHistorySave = pyqtSignal()   # emitted BEFORE a change, for undo

    def __init__(self, parent=None):
        super().__init__(parent)
        self.molecule: Optional[Molecule] = None
        self._positions: Positions2D = {}
        self._bond_info: BondInfo = {}

        self._zoom = 1.0   # user-facing zoom, shared with the 3D view (1.0 = "100%")
        self._pan = np.array([0.0, 0.0])
        self._center: Tuple[float, float] = (0.0, 0.0)  # frozen at each _fit_view(); see render_chemvector's `center`
        self._drag_start: Optional[QPoint] = None
        self._drag_mode = "none"

        self.build_mode = False
        self.build_element = "C"
        self._bonding_from: Optional[int] = None
        self._mouse_pos: Optional[QPoint] = None

        self.selection_mode = False
        self.paint_mode = False
        self._paint_color = DEFAULT_PAINT_COLOR
        # 2D's own paint layer — entirely separate from Atom.color (which
        # only the 3D view reads/writes), so painting here never touches it.
        self.atom_paint: Dict[int, str] = {}
        # 2D's own aromaticity overlay from Check Aromaticity — bond indices
        # drawn with a ring circle instead of Kekulized lines. Never written
        # to bond.order, so it never touches the 3D view either.
        self.aromatic_bonds: set = set()

        self.setMinimumSize(500, 450)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # Render parameters — public, mirrored from MainWindow's 3D settings where sensible.
        # See the "SCALING STRATEGY" note in render_chemvector for how each behaves on zoom.
        # px per Angstrom at 100% zoom — set from the 3D view's own scale so both views match
        self.base_scale = 60.0
        self._ref_len = 1.5   # typical bond length of the layout (Angstrom), see _refresh_ref_len
        self.background = "#ffffff"
        self.bond_color = "#2b2b2b"
        self.font_family = "Helvetica, Arial, sans-serif"
        # ZOOM-ONLY — all scaled by self._zoom on render, see _render_to_bytes
        self.bond_width_px = 3.4
        self.font_size_px = 26.0
        self.double_bond_gap_px = DOUBLE_BOND_GAP_PX
        self.show_carbon_labels = False
        self.show_hydrogens = False
        self.label_color_mode = "cpk"   # "cpk" | "black" — atom letter colors
        self.color_overrides: Dict[str, str] = {}
        self.planarity_tolerance = PLANARITY_TOLERANCE
        self.layout_mode = "match3d"   # "match3d" (atoms exactly where the 3D view has them) | "clean" (idealized depiction)
        self._plane = None              # (normal, point) chosen with Align plane; kept while the molecule is
        self.align_mode = False
        self._green_atoms: set = set()   # atoms marked (Alt+click) to define a plane
        self.measure_mode = False
        # Distances/angles are real 3D quantities (from Atom.x/y/z), drawn at
        # the atoms' current 2D screen positions — same measurements the 3D
        # view reports, whichever layout (or projection) is on screen here.
        self._measure_atoms: List[List[int]] = []
        self._measure_from: Optional[int] = None
        self._angle_atoms: List[List[int]] = []
        self._angle_leg1: Optional[Tuple[int, int]] = None
        # The 3D view this depiction is a projection of: camera rotation, and
        # the 3D point (the atoms' centroid when last aligned) that sits at
        # layout (0, 0). Together they map a 2D layout point back to 3D.
        self.view_rot = np.eye(3)
        self._frame_origin = np.zeros(3)
        self.selected_atoms: set = set()

        # Build mode subtools: what kind of bond a drag draws / a bond click
        # sets ("single" | "double" | "triple" | "wedge" | "hash" | "ring"),
        # and whether the cursor snaps to atoms, ring centers, bond midpoints
        # and ideal next-atom positions.
        self.bond_type = "single"
        self.snapping = False
        self._snap_hint: Optional[Tuple[float, float, str]] = None   # canvas x, y, kind
        self._ring_bond_from: Optional[List[int]] = None   # ring a ring bond is being dragged out of

        # Text tool (a build subtool): edit the label of an atom drawn as
        # letters — its text, plus color and boldness (Atom.label_color/_bold)
        self.text_mode = False
        self._text_editor: Optional[_TextEditor] = None
        self._editing_atom: Optional[int] = None
        self._editing_initial: Tuple[str, bool] = ("#000000", True)
        self._label_boxes: Dict[int, Box] = {}   # canvas box of each drawn label, from the last render

        self._render_timer = QTimer()
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._do_render)

    # ── public API ───────────────────────────────────────────────────────────

    def load_molecule(self, mol: Optional[Molecule], regenerate: bool = True, keep_view: bool = False,
                      view_rot: Optional[np.ndarray] = None):
        """
        Show *mol*. With *keep_view*, zoom/pan/scale are left exactly as they
        are (only the layout is redone, recentered on the same screen spot) —
        for edits made while this view is showing (clear, undo, clean, ...),
        as opposed to opening a different molecule, which refits the view.
        """
        self._close_text_editor()
        self.molecule = mol
        if view_rot is not None:
            self.view_rot = np.array(view_rot, dtype=float)
        if regenerate:
            # These are keyed by atom/bond index of the previous molecule.
            self.selected_atoms = set()
            self._green_atoms = set()
            if not keep_view:
                self._plane = None
            self.atom_paint = {}
            self.aromatic_bonds = set()
            self.regenerate_layout(keep_view=keep_view)
        else:
            self.request_render()

    def regenerate_layout(self, keep_view: bool = False):
        if self.molecule is None or not self.molecule.atoms:
            self._positions, self._bond_info = {}, {}
            if not keep_view:
                self._fit_view()  # clean slate for a fresh build
            self.request_render()
            return
        self._positions, self._bond_info = self._make_layout()
        self._frame_origin = self._atoms_centroid()
        if keep_view:
            self._center = self._layout_centroid()
            self._refresh_ref_len()
        else:
            self._fit_view()
        self.layoutRegenerated.emit()
        self.request_render()

    def _make_layout(self):
        positions, bond_info = generate_2d_layout(
            self.molecule, self.planarity_tolerance, self.view_rot,
            projection=(self.layout_mode == "match3d"), plane=self._plane)
        _apply_drawn_stereo(self.molecule, bond_info)
        return positions, bond_info

    def set_layout_mode(self, mode: str):
        """Switch between "match3d" and "clean" layouts and redraw (view unchanged)."""
        if mode == self.layout_mode:
            return
        self.layout_mode = mode
        if self.molecule is not None and self.molecule.atoms:
            self._positions, self._bond_info = self._make_layout()
            self._frame_origin = self._atoms_centroid()
            self._center = self._layout_centroid()
            self._refresh_ref_len()
            self.request_render()

    @property
    def _scale_px(self) -> float:
        """Pixels per layout unit (Angstrom): the 3D view's scale times the shared zoom."""
        return self.base_scale * self._zoom

    @property
    def _eff_zoom(self) -> float:
        """Factor for line thickness, letters and spacing: the drawn bond length relative to CHEM_SIZE_REF_BOND_PX."""
        return self._ref_len * self._scale_px / CHEM_SIZE_REF_BOND_PX

    def _refresh_ref_len(self):
        self._ref_len = self._reference_bond_length()

    def _atoms_centroid(self) -> np.ndarray:
        if self.molecule is None or not self.molecule.atoms:
            return np.zeros(3)
        return np.array([[a.x, a.y, a.z] for a in self.molecule.atoms], dtype=float).mean(axis=0)

    def align_to_view(self, rot: np.ndarray):
        """
        Turn the existing 2D layout (sketch positions and all) to match the
        3D view with camera rotation *rot* — rotating it in the plane, or
        mirroring it (swapping wedge/hash) when it fits better seen from
        behind. Zoom/pan are left alone.
        """
        self.view_rot = np.array(rot, dtype=float)
        if self.molecule is None or not self._positions:
            self._frame_origin = self._atoms_centroid()
            return
        if self.layout_mode == "match3d":
            # Every atom is where the 3D view draws it, so just redo the projection.
            self._positions, self._bond_info = self._make_layout()
            self._frame_origin = self._atoms_centroid()
            self._center = self._layout_centroid()
            self._refresh_ref_len()
            self.request_render()
            return
        self._positions, reflected = _align_layout_to_view(
            self.molecule, self._positions, self.view_rot, rescale=False)
        if reflected:
            _swap_wedge_hash(self._bond_info)
        self._frame_origin = self._atoms_centroid()
        _annotate_double_bond_insets(self.molecule, self._positions, self._bond_info)
        self._center = self._layout_centroid()
        self._refresh_ref_len()
        self.request_render()

    def reset_view(self):
        self._fit_view()
        self.request_render()

    def request_render(self, delay_ms: int = 0):
        self._render_timer.start(delay_ms)

    def get_svg_bytes(self, export_mode: bool = False) -> bytes:
        return self._render_to_bytes(export_mode=export_mode)

    # ── internal ─────────────────────────────────────────────────────────────

    def _reference_bond_length(self) -> float:
        """Typical bond length of the current layout (median over non-hydrogen bonds), in layout units."""
        mol = self.molecule
        lens: List[float] = []
        if mol is not None:
            for heavy_only in (True, False):
                for b in mol.bonds:
                    if b.i not in self._positions or b.j not in self._positions:
                        continue
                    if heavy_only and "H" in (mol.atoms[b.i].element, mol.atoms[b.j].element):
                        continue
                    d = math.dist(self._positions[b.i], self._positions[b.j])
                    if d > 1e-6:
                        lens.append(d)
                if lens:
                    break
        return float(np.median(lens)) if lens else 1.5

    def _fit_view(self):
        """Reset zoom (shared with the 3D view) and pan; the layout is centered on the 3D centroid's projection."""
        if self._zoom != 1.0:
            self._zoom = 1.0
            self.zoomChanged.emit(1.0)
        self._pan = np.array([0.0, 0.0])
        self._center = self._layout_centroid()
        self._refresh_ref_len()

    def _render_to_bytes(self, w=None, h=None, export_mode: bool = False) -> bytes:
        if self.molecule is None or not self.molecule.atoms:
            self._label_boxes = {}
            return b""
        w = w or max(self.width(), 500)
        h = h or max(self.height(), 450)
        pan_x, pan_y, center = self._pan[0], self._pan[1], self._center
        label_boxes: Dict[int, Box] = {}
        if export_mode:
            # Export what is drawn, not what fits the window: size the canvas
            # to the visible atoms (plus room for labels) at the current
            # scale, so a molecule larger than the window isn't cropped.
            hidden = set(self._hidden_h_atoms())
            pts = [p for i, p in self._positions.items() if i not in hidden]
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                scale_px = self._scale_px
                pad = self.font_size_px * self._eff_zoom * 1.3
                w = int(math.ceil((max(xs) - min(xs)) * scale_px + 2 * pad))
                h = int(math.ceil((max(ys) - min(ys)) * scale_px + 2 * pad))
                center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
                pan_x = pan_y = 0.0
        svg = render_chemvector(
            self.molecule, self._positions, self._bond_info,
            canvas_w=w, canvas_h=h,
            pan_x=pan_x, pan_y=pan_y,
            scale=self._scale_px,
            center=center,
            background=self.background,
            color_overrides=self.color_overrides or None,
            bond_color=self.bond_color,
            # Line thickness and parallel-line spacing scale with the
            # interactive zoom level only — not with the molecule's auto-fit
            # scale — so they stay a consistent size right after loading
            # regardless of how compact the fit is, and grow/shrink together
            # (along with letters) as the user zooms in/out from there.
            bond_width_px=self.bond_width_px * self._eff_zoom,
            double_bond_gap_px=self.double_bond_gap_px * self._eff_zoom,
            font_family=self.font_family,
            font_size_px=self.font_size_px * self._eff_zoom,
            show_carbon_labels=self.show_carbon_labels,
            show_hydrogens=self.show_hydrogens,
            label_color_mode=self.label_color_mode,
            selected_indices=None if export_mode else self.selected_atoms,  # selection is UI, paint is content
            paint_colors=self.atom_paint or None,
            aromatic_bond_idxs=self.aromatic_bonds or None,
            export_mode=export_mode,
            ring_bonds=self.molecule.ring_bonds,
            label_boxes_out=None if export_mode else label_boxes,
        )
        if not export_mode:
            self._label_boxes = label_boxes
        return svg.encode("utf-8")

    def _do_render(self):
        data = self._render_to_bytes()
        if not data:
            # Nothing to draw (e.g. after Clear All) — show a blank whiteboard
            # instead of leaving the previous drawing on screen.
            data = render_chemvector(
                Molecule("empty"), {}, {}, canvas_w=max(self.width(), 500),
                canvas_h=max(self.height(), 450), background=self.background,
            ).encode("utf-8")
        self.load(QByteArray(data))

    # ── build mode: coordinate transforms ───────────────────────────────────
    # Mirror render_chemvector's own to_canvas() exactly (including its
    # y-flip), so hit-testing and new-atom placement line up with what's drawn.

    def _layout_centroid(self) -> Tuple[float, float]:
        """
        Layout point drawn at the canvas center: the origin, i.e. the
        projection of the molecule's 3D centroid — exactly what the 3D view
        centers on, so switching views doesn't shift the drawing.
        """
        return (0.0, 0.0)

    def _to_canvas_point(self, x: float, y: float) -> Tuple[float, float]:
        ccx, ccy = self._center
        scale = self._scale_px
        cx = self.width() / 2 + self._pan[0]
        cy = self.height() / 2 + self._pan[1]
        return (cx + (x - ccx) * scale, cy - (y - ccy) * scale)

    def _from_canvas_point(self, px: float, py: float) -> Tuple[float, float]:
        ccx, ccy = self._center
        scale = self._scale_px
        cx = self.width() / 2 + self._pan[0]
        cy = self.height() / 2 + self._pan[1]
        return (ccx + (px - cx) / scale, ccy - (py - cy) / scale)

    def _hidden_h_atoms(self, idx: Optional[int] = None) -> List[int]:
        """Hydrogen atoms currently hidden as implicit (all of them, or only those on carbon *idx*)."""
        mol = self.molecule
        if mol is None or self.show_hydrogens:
            return []
        out = []
        for i, a in enumerate(mol.atoms):
            if a.element != "H" or not _hydrogen_is_implicit(i, mol, self._bond_info):
                continue
            if idx is None or any((b.i == i and b.j == idx) or (b.j == i and b.i == idx) for b in mol.bonds):
                out.append(i)
        return out

    def _get_hit_atom(self, pos: QPoint, radius_px: float = 16.0) -> Optional[int]:
        hidden = set(self._hidden_h_atoms())
        for i, (x, y) in self._positions.items():
            if i in hidden:
                continue
            ax, ay = self._to_canvas_point(x, y)
            if math.hypot(pos.x() - ax, pos.y() - ay) < radius_px:
                return i
        return None

    def _get_hit_bond(self, pos: QPoint, threshold_px: float = 8.0) -> Optional[int]:
        if self.molecule is None:
            return None
        for bi, b in enumerate(self.molecule.bonds):
            if b.i not in self._positions or b.j not in self._positions:
                continue
            ax, ay = self._to_canvas_point(*self._positions[b.i])
            bx, by = self._to_canvas_point(*self._positions[b.j])
            l2 = (bx - ax) ** 2 + (by - ay) ** 2
            if l2 < 1e-6:
                continue
            t = max(0.0, min(1.0, ((pos.x() - ax) * (bx - ax) + (pos.y() - ay) * (by - ay)) / l2))
            projx, projy = ax + t * (bx - ax), ay + t * (by - ay)
            if math.hypot(pos.x() - projx, pos.y() - projy) < threshold_px:
                return bi
        return None

    def _get_hit_ring(self, pos: QPoint) -> Optional[List[int]]:
        """The smallest ring whose interior (as a polygon over its atoms'
        canvas positions) contains pos — lets a plain click select a ring by
        landing anywhere inside it, not just on one of its atoms/bonds."""
        if self.molecule is None:
            return None
        best = None
        for ring in self._rings():
            pts = [self._to_canvas_point(*self._positions[a]) for a in ring if a in self._positions]
            if len(pts) >= 3 and _point_in_polygon(pos.x(), pos.y(), pts):
                if best is None or len(ring) < len(best):
                    best = ring
        return best

    def _rings(self) -> List[List[int]]:
        """Every atom's smallest ring, deduplicated."""
        if self.molecule is None:
            return []
        seen: Dict[frozenset, List[int]] = {}
        for i in range(len(self.molecule.atoms)):
            ring = _find_ring_for_atom(self.molecule, i)
            if ring:
                seen.setdefault(frozenset(ring), ring)
        return list(seen.values())

    def _ring_center_canvas(self, ring: List[int]) -> Optional[Tuple[float, float]]:
        pts = [self._to_canvas_point(*self._positions[a]) for a in ring if a in self._positions]
        if len(pts) < 3:
            return None
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def _ring_at(self, pos: QPoint) -> Optional[List[int]]:
        """The ring whose center is at *pos* (within the snap radius), else the one *pos* is inside."""
        for ring in self._rings():
            c = self._ring_center_canvas(ring)
            if c and math.hypot(pos.x() - c[0], pos.y() - c[1]) < SNAP_RADIUS_PX:
                return ring
        return self._get_hit_ring(pos)

    def _ideal_bond_px(self) -> float:
        return self._ref_len * self._scale_px

    def _snap(self, pos: QPoint, origin: Optional[int] = None) -> Tuple[QPoint, Optional[str]]:
        """
        With snapping on, pull *pos* onto the nearest important location
        within SNAP_RADIUS_PX: an atom, a ring center, a bond midpoint, or —
        while drawing a bond from atom *origin* — an ideal next-atom spot
        (typical bond length, on a 30° step). Atoms win over ring centers,
        which win over the rest. Returns the point and what it snapped to.
        """
        if not self.snapping or self.molecule is None:
            return pos, None
        cands: List[Tuple[float, float, str]] = []
        hidden = set(self._hidden_h_atoms())
        for i, (x, y) in self._positions.items():
            if i not in hidden:
                cands.append((*self._to_canvas_point(x, y), "atom"))
        for ring in self._rings():
            c = self._ring_center_canvas(ring)
            if c:
                cands.append((*c, "ring"))
        for b in self.molecule.bonds:
            if b.i in self._positions and b.j in self._positions and not ({b.i, b.j} & hidden):
                ax, ay = self._to_canvas_point(*self._positions[b.i])
                bx, by = self._to_canvas_point(*self._positions[b.j])
                cands.append(((ax + bx) / 2, (ay + by) / 2, "bond"))
        if origin is not None and origin in self._positions:
            ox, oy = self._to_canvas_point(*self._positions[origin])
            ref = self._existing_bond_angle_canvas(origin) or 0.0
            length = self._ideal_bond_px()
            for k in range(12):
                a = ref + k * math.pi / 6
                cands.append((ox + length * math.cos(a), oy + length * math.sin(a), "ideal"))
        rank = {"atom": 0, "ring": 1, "bond": 2, "ideal": 2}
        best = None
        for x, y, kind in cands:
            d = math.hypot(pos.x() - x, pos.y() - y)
            if d < SNAP_RADIUS_PX:
                key = (rank[kind], d)
                if best is None or key < best[0]:
                    best = (key, x, y, kind)
        if best is None:
            return pos, None
        return QPoint(round(best[1]), round(best[2])), best[3]

    def _snap_ring_bond_end(self, ring: List[int], pos: QPoint) -> QPoint:
        """
        Snapped end of a ring bond dragged to *pos*: onto an atom outside
        the ring if one is in reach, else straight out over the nearest ring
        bond (through its midpoint), and at a typical bond length past the
        ring's edge when close to it.
        """
        snapped, kind = self._snap(pos)
        if kind == "atom" and self._get_hit_atom(snapped) not in ring:
            return snapped
        if not self.snapping:
            return pos
        c = self._ring_center_canvas(ring)
        if c is None:
            return pos
        dx, dy = pos.x() - c[0], pos.y() - c[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return pos
        best = None
        n = len(ring)
        for k in range(n):
            a, b = ring[k], ring[(k + 1) % n]
            if a not in self._positions or b not in self._positions:
                continue
            ax, ay = self._to_canvas_point(*self._positions[a])
            bx, by = self._to_canvas_point(*self._positions[b])
            mx, my = (ax + bx) / 2 - c[0], (ay + by) / 2 - c[1]
            m = math.hypot(mx, my)
            if m < 1e-6:
                continue
            cosang = (mx * dx + my * dy) / (m * dist)
            if best is None or cosang > best[0]:
                best = (cosang, mx / m, my / m, m)
        if best is None:
            return pos
        _, ux, uy, apothem = best
        along = max(dist, apothem * 1.2)
        ideal = apothem + self._ideal_bond_px()
        if abs(along - ideal) < SNAP_RADIUS_PX * 1.5:
            along = ideal
        return QPoint(round(c[0] + ux * along), round(c[1] + uy * along))

    def _get_hit_ring_bond(self, pos: QPoint, threshold_px: float = 8.0) -> Optional[int]:
        """Index into molecule.ring_bonds of the ring bond under *pos*."""
        if self.molecule is None:
            return None
        for k, rb in enumerate(self.molecule.ring_bonds):
            c = self._ring_center_canvas(rb["ring"])
            if c is None or rb["atom"] not in self._positions:
                continue
            bx, by = self._to_canvas_point(*self._positions[rb["atom"]])
            ax, ay = c
            l2 = (bx - ax) ** 2 + (by - ay) ** 2
            if l2 < 1e-6:
                continue
            t = max(0.0, min(1.0, ((pos.x() - ax) * (bx - ax) + (pos.y() - ay) * (by - ay)) / l2))
            if math.hypot(pos.x() - (ax + t * (bx - ax)), pos.y() - (ay + t * (by - ay))) < threshold_px:
                return k
        return None

    def _existing_bond_angle_canvas(self, atom_idx: int) -> Optional[float]:
        """Canvas-space angle (radians) of one existing bond at atom_idx, or None if it has none."""
        if self.molecule is None or atom_idx not in self._positions:
            return None
        for b in self.molecule.bonds:
            other = b.j if b.i == atom_idx else (b.i if b.j == atom_idx else None)
            if other is None or other not in self._positions:
                continue
            ax, ay = self._to_canvas_point(*self._positions[atom_idx])
            bx, by = self._to_canvas_point(*self._positions[other])
            if math.hypot(bx - ax, by - ay) < 1e-6:
                continue
            return math.atan2(by - ay, bx - ax)
        return None

    def _snap_to_60deg(self, origin_idx: int, target: QPoint) -> QPoint:
        """
        Snap *target* so the drag from origin_idx sits on a 60° increment,
        measured from one of origin_idx's existing bonds (or from the
        horizontal, if it has none yet) — keeps hand-drawn angles matching
        standard sp2/sp3/aromatic bond geometry.
        """
        if origin_idx not in self._positions:
            return target
        ax, ay = self._to_canvas_point(*self._positions[origin_idx])
        dx, dy = target.x() - ax, target.y() - ay
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return target
        reference = self._existing_bond_angle_canvas(origin_idx)
        if reference is None:
            reference = 0.0
        step = math.radians(60.0)
        k = round((math.atan2(dy, dx) - reference) / step)
        snapped_angle = reference + k * step
        sx = ax + dist * math.cos(snapped_angle)
        sy = ay + dist * math.sin(snapped_angle)
        return QPoint(round(sx), round(sy))

    # ── build mode: editing ─────────────────────────────────────────────────

    def _add_atom(self, pos: QPoint, depth_from: Optional[int] = None, depth_offset: float = 0.0) -> int:
        """
        Add an atom at canvas point *pos*. Its 3D position is the inverse of
        this view's projection: the picked point in the screen plane, at the
        depth of atom *depth_from* (or at the molecule's center depth), moved
        *depth_offset* Angstrom toward the viewer (negative: away).
        """
        if self.molecule is None:
            self.molecule = Molecule("New Molecule", atoms=[])
        x, y = self._from_canvas_point(pos.x(), pos.y())
        depth = 0.0
        if depth_from is not None and depth_from < len(self.molecule.atoms):
            a = self.molecule.atoms[depth_from]
            depth = float((self.view_rot @ (np.array([a.x, a.y, a.z]) - self._frame_origin))[2])
        depth += depth_offset
        zf = PERSPECTIVE_CAMERA_Z / (PERSPECTIVE_CAMERA_Z - depth) if abs(PERSPECTIVE_CAMERA_Z - depth) > 1e-6 else 1.0
        world = self._frame_origin + self.view_rot.T @ np.array([x / zf, y / zf, depth])
        self.molecule.atoms.append(Atom(self.build_element, float(world[0]), float(world[1]), float(world[2])))
        idx = len(self.molecule.atoms) - 1
        self._positions[idx] = (x, y)
        return idx

    def _add_bond(self, i: int, j: int, order: int = 1, stereo: Optional[str] = None):
        self.molecule.bonds.append(Bond(i, j, order, stereo))
        bi = len(self.molecule.bonds) - 1
        self._bond_info[bi] = {"kind": stereo or "none", "apex": i, "order": order}
        # A new bond can close a ring that an existing double bond is part
        # of (or open/close another one), so refresh which double bonds get
        # the ring-aware inner line and on which side.
        _annotate_double_bond_insets(self.molecule, self._positions, self._bond_info)

    def _new_bond_args(self) -> Tuple[int, Optional[str]]:
        """(order, stereo) of a bond drawn with the current bond type."""
        return {"double": (2, None), "triple": (3, None),
                "wedge": (1, "wedge"), "hash": (1, "hash")}.get(self.bond_type, (1, None))

    def _apply_bond_type(self, bi: int) -> bool:
        """
        A click on bond *bi* in build mode: Regular cycles its order (or
        first clears a wedge/hash); Double/Triple set that order; Wedge/Hash
        make it one, or flip its direction when it already is. Returns
        whether anything changed.
        """
        bond = self.molecule.bonds[bi]
        bt = self.bond_type
        if bt == "ring":
            return False
        info = self._bond_info.setdefault(bi, {"kind": "none", "apex": bond.i, "order": bond.order})
        if bt in ("wedge", "hash"):
            if bond.stereo == bt:
                bond.i, bond.j = bond.j, bond.i   # narrow end goes to the other atom
            bond.stereo = bt
            bond.order = 1
            info.update(kind=bt, apex=bond.i, order=1)
            info.pop("inside", None)
            _annotate_double_bond_insets(self.molecule, self._positions, self._bond_info)
            return True
        if bond.stereo:
            bond.stereo = None
            info["kind"] = "none"
            if bt == "single":
                _annotate_double_bond_insets(self.molecule, self._positions, self._bond_info)
                return True
        if bt == "single":
            self._cycle_bond_order(bi)
            return True
        order = 2 if bt == "double" else 3
        bond.order = order
        info.update(kind="none", order=order)
        _annotate_double_bond_insets(self.molecule, self._positions, self._bond_info)
        return True

    def _cycle_bond_order(self, bi: int):
        bond = self.molecule.bonds[bi]
        bond.order = (bond.order % 3) + 1
        info = self._bond_info.setdefault(bi, {"kind": "none", "apex": bond.i, "order": bond.order})
        info["order"] = bond.order
        if bond.order != 2:
            info["kind"] = "none"
            info.pop("inside", None)
        _annotate_double_bond_insets(self.molecule, self._positions, self._bond_info)

    def _delete_bond(self, bi: int):
        self.molecule.bonds.pop(bi)
        self._bond_info = {
            (k - 1 if k > bi else k): v for k, v in self._bond_info.items() if k != bi
        }
        self.aromatic_bonds = {(k - 1 if k > bi else k) for k in self.aromatic_bonds if k != bi}
        _annotate_double_bond_insets(self.molecule, self._positions, self._bond_info)

    def _delete_atom(self, idx: int):
        """Delete atom *idx*, and any now-lone terminal H it was carrying, reindexing as we go."""
        mol = self.molecule
        h_to_remove = []
        for b in mol.bonds:
            if b.i == idx:
                nb_idx = b.j
            elif b.j == idx:
                nb_idx = b.i
            else:
                continue
            nb_atom = mol.atoms[nb_idx]
            if nb_atom.element == "H":
                other_bonds = sum(1 for b2 in mol.bonds if b2.i == nb_idx or b2.j == nb_idx)
                if other_bonds == 1:
                    h_to_remove.append(nb_idx)

        for d_idx in sorted(set(h_to_remove + [idx]), reverse=True):
            mol.atoms.pop(d_idx)
            new_bonds: List[Bond] = []
            new_bond_info: BondInfo = {}
            new_aromatic_bonds: set = set()
            for bi, b in enumerate(mol.bonds):
                if b.i == d_idx or b.j == d_idx:
                    continue
                ni = b.i - 1 if b.i > d_idx else b.i
                nj = b.j - 1 if b.j > d_idx else b.j
                new_bonds.append(Bond(ni, nj, b.order, b.stereo))
                info = dict(self._bond_info.get(bi, {"kind": "none", "apex": b.i, "order": b.order}))
                old_apex = info.get("apex", b.i)
                info["apex"] = old_apex - 1 if old_apex > d_idx else old_apex
                new_bond_info[len(new_bonds) - 1] = info
                if bi in self.aromatic_bonds:
                    new_aromatic_bonds.add(len(new_bonds) - 1)
            mol.bonds = new_bonds
            self._bond_info = new_bond_info
            self.aromatic_bonds = new_aromatic_bonds
            self._positions = {
                (i - 1 if i > d_idx else i): p for i, p in self._positions.items() if i != d_idx
            }
            # selected_atoms/atom_paint are keyed by atom index too — reshift
            # them the same way on every deletion, not just filter afterward,
            # or a later atom's selection/paint would silently land on
            # whichever different atom slid down into that freed index.
            self.selected_atoms = {(i - 1 if i > d_idx else i) for i in self.selected_atoms if i != d_idx}
            self.atom_paint = {
                (i - 1 if i > d_idx else i): c for i, c in self.atom_paint.items() if i != d_idx
            }
            self._green_atoms = {(i - 1 if i > d_idx else i) for i in self._green_atoms if i != d_idx}
            shift = lambda i: i - 1 if i > d_idx else i
            mol.ring_bonds = [
                {"ring": [shift(r) for r in rb["ring"]], "atom": shift(rb["atom"])}
                for rb in mol.ring_bonds if rb["atom"] != d_idx and d_idx not in rb["ring"]
            ]
        _annotate_double_bond_insets(mol, self._positions, self._bond_info)

    # ── drag and drop ────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.fileDropped.emit(urls[0].toLocalFile())
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    # ── mouse: pan + zoom (no rotation — this is a flat 2D view), plus build/selection/paint modes ─

    def mousePressEvent(self, event):
        pos = event.position().toPoint()

        if self.text_mode and self._text_press(event, pos):
            return

        if self.align_mode and event.button() == Qt.MouseButton.LeftButton and self.molecule:
            if event.modifiers() & Qt.KeyboardModifier.AltModifier:
                idx = self._get_hit_atom(pos)
                if idx is not None:
                    self._green_atoms ^= {idx}
                    self.update()
                    return
            else:
                bi = self._get_hit_bond(pos)
                if bi is not None:
                    self._align_bond(bi, bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
                    return

        if self.measure_mode and event.button() == Qt.MouseButton.LeftButton and self.molecule:
            idx = self._get_hit_atom(pos)
            if idx is not None:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    if self._angle_leg1 is None and self._measure_from is None:
                        self._measure_from = idx
                        self._mouse_pos = pos
                        self.update()
                        return
                    elif self._measure_from is not None and self._angle_leg1 is None:
                        if idx != self._measure_from:
                            self._angle_leg1 = (self._measure_from, idx)
                        self._measure_from = None
                        self.update()
                        return
                    elif self._angle_leg1 is not None:
                        i1, i2 = self._angle_leg1
                        if idx != i1:
                            self._angle_atoms.append([i1, i2, idx])
                        self._angle_leg1 = None
                        self._measure_from = None
                        self._mouse_pos = None
                        self.update()
                        return
                    self.update()
                    return
                if self._measure_from is not None and idx != self._measure_from:
                    self._measure_atoms.append([self._measure_from, idx])
                    self._measure_from = None
                    self._mouse_pos = None
                    self.update()
                    return
                self._measure_from = idx
                self._mouse_pos = pos
                self.update()
                return
            self._measure_from = None
            self._mouse_pos = None
            self.update()
            return

        if self.paint_mode and event.button() == Qt.MouseButton.LeftButton and self.molecule:
            idx = self._get_hit_atom(pos)
            if idx is not None:
                # 2D's own paint layer (atom_paint) — separate from Atom.color,
                # so this never touches or is touched by the 3D view's paint.
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.atom_paint.pop(idx, None)
                else:
                    self.atom_paint[idx] = self._paint_color
                self.request_render()
                return

        if self.selection_mode and event.button() == Qt.MouseButton.LeftButton and self.molecule:
            mod = event.modifiers()
            idx = self._get_hit_atom(pos)
            if idx is not None:
                if mod & Qt.KeyboardModifier.ShiftModifier:
                    if idx in self.selected_atoms:
                        self.selected_atoms.discard(idx)
                    else:
                        self.selected_atoms.add(idx)
                else:
                    self.selected_atoms = {idx} if idx not in self.selected_atoms else set()
                self.request_render()
                return
            # No modifier needed for a ring: clicking one of its bonds, or
            # anywhere in its (otherwise empty) interior, selects it whole.
            bi = self._get_hit_bond(pos)
            ring = _find_ring_for_bond(self.molecule, bi) if bi is not None else None
            if ring is None:
                ring = self._get_hit_ring(pos)
            if ring:
                if mod & Qt.KeyboardModifier.ShiftModifier:
                    self.selected_atoms.update(ring)
                else:
                    self.selected_atoms = set(ring)
                self.request_render()
                return
            self.selected_atoms.clear()
            self.request_render()
            return

        if self.build_mode and event.button() == Qt.MouseButton.LeftButton:
            raw = pos
            pos, _kind = self._snap(pos)
            idx = self._get_hit_atom(pos)
            if self.bond_type == "ring" and idx is None and self.molecule:
                ring = self._ring_at(pos)
                if ring:
                    # Ring bond: dragged out from the ring's center
                    self._ring_bond_from = ring
                    self._mouse_pos = pos
                    self.update()
                    return
            if idx is not None:
                self._bonding_from = idx
                self._mouse_pos = pos
                self.update()
                return
            bi = self._get_hit_bond(raw)
            if bi is not None:
                self.requestHistorySave.emit()
                if self._apply_bond_type(bi):
                    self.moleculeChanged.emit()
                    self.request_render()
                return
            self.requestHistorySave.emit()
            self._add_atom(pos)
            self.moleculeChanged.emit()
            self.request_render()
            return

        if self.build_mode and event.button() == Qt.MouseButton.RightButton and self.molecule:
            idx = self._get_hit_atom(pos)
            if idx is not None:
                self.requestHistorySave.emit()
                self._delete_atom(idx)
                self.moleculeChanged.emit()
                self.request_render()
                return
            bi = self._get_hit_bond(pos)
            if bi is not None:
                self.requestHistorySave.emit()
                self._delete_bond(bi)
                self.moleculeChanged.emit()
                self.request_render()
                return
            rbi = self._get_hit_ring_bond(pos)
            if rbi is not None:
                self.requestHistorySave.emit()
                self.molecule.ring_bonds.pop(rbi)
                self.moleculeChanged.emit()
                self.request_render()
                return

        if (event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
                and event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
                and self.molecule and self._positions
                and not (self.build_mode or self.selection_mode or self.paint_mode or self.measure_mode
                         or self.text_mode)):
            # No tool active: Ctrl (Cmd or Ctrl on macOS) + drag spins the
            # depiction in the plane, about where it currently sits on screen.
            px, py = self._to_canvas_point(0.0, 0.0)  # the 3D centroid's projection
            self._rot_pivot = (px, py)
            self._rot_last = math.atan2(pos.y() - py, pos.x() - px)
            self._drag_mode = "rotate"
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            return

        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton,
                               Qt.MouseButton.MiddleButton):
            self._drag_start = pos
            self._drag_mode = "pan"
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def mouseMoveEvent(self, event):
        if self.build_mode and self._ring_bond_from is not None:
            self._mouse_pos = self._snap_ring_bond_end(self._ring_bond_from, event.position().toPoint())
            self._snap_hint = (float(self._mouse_pos.x()), float(self._mouse_pos.y()), "ideal") if self.snapping else None
            self.update()
            return
        if self.build_mode and self._bonding_from is not None:
            pos = event.position().toPoint()
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                pos = self._snap_to_60deg(self._bonding_from, pos)
            else:
                pos, _kind = self._snap(pos, origin=self._bonding_from)
            self._mouse_pos = pos
            self._update_snap_hint(pos)
            return
        if self.build_mode and self.snapping and not self.text_mode and self._drag_mode == "none":
            self._update_snap_hint(event.position().toPoint())
        if self.measure_mode and (self._measure_from is not None or self._angle_leg1 is not None):
            self._mouse_pos = event.position().toPoint()
            self.update()
            return
        if self._drag_mode == "rotate":
            pos = event.position().toPoint()
            px, py = self._rot_pivot
            ang = math.atan2(pos.y() - py, pos.x() - px)
            self._rotate_layout(ang - self._rot_last)
            self._rot_last = ang
            return
        if self._drag_mode == "pan" and self._drag_start is not None:
            pos = event.position().toPoint()
            dx = pos.x() - self._drag_start.x()
            dy = pos.y() - self._drag_start.y()
            self._drag_start = pos
            self._pan[0] += dx
            self._pan[1] += dy
            self.request_render(delay_ms=0)

    def mouseReleaseEvent(self, event):
        if self.measure_mode and self._measure_from is not None:
            end_idx = self._get_hit_atom(event.position().toPoint())
            if end_idx is not None and end_idx != self._measure_from and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self._measure_atoms.append([self._measure_from, end_idx])
                self._measure_from = None
                self._mouse_pos = None
            self.update()
            return
        if self.build_mode and self._ring_bond_from is not None:
            self._finish_ring_bond(event.position().toPoint())
            return
        if self.build_mode and self._bonding_from is not None:
            pos = event.position().toPoint()
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                pos = self._snap_to_60deg(self._bonding_from, pos)
            else:
                pos, _kind = self._snap(pos, origin=self._bonding_from)
            self._snap_hint = None
            order, stereo = self._new_bond_args()
            end_idx = self._get_hit_atom(pos)
            if end_idx is not None and end_idx == self._bonding_from:
                self.requestHistorySave.emit()
                self.molecule.atoms[self._bonding_from].element = self.build_element
                self.moleculeChanged.emit()
            elif end_idx is not None:
                self.requestHistorySave.emit()
                self._add_bond(self._bonding_from, end_idx, order, stereo)
                self.moleculeChanged.emit()
            else:
                self.requestHistorySave.emit()
                # A wedge/hash also puts the new atom toward/away from the
                # viewer in 3D, so the structure matches what was drawn.
                offset = {"wedge": STEREO_DEPTH, "hash": -STEREO_DEPTH}.get(stereo, 0.0)
                new_idx = self._add_atom(pos, depth_from=self._bonding_from, depth_offset=offset)
                self._add_bond(self._bonding_from, new_idx, order, stereo)
                self.moleculeChanged.emit()
            self._bonding_from = None
            self._mouse_pos = None
            self.request_render()
            self.update()
            return
        self._drag_start = None
        self._drag_mode = "none"
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def _update_snap_hint(self, pos: Optional[QPoint]):
        """Show (or clear) the marker over the location the cursor would snap to."""
        hint = None
        if pos is not None and self.snapping:
            snapped, kind = self._snap(pos, origin=self._bonding_from)
            if kind is not None:
                hint = (float(snapped.x()), float(snapped.y()), kind)
        if hint != self._snap_hint or self._bonding_from is not None:
            self._snap_hint = hint
            self.update()

    def _finish_ring_bond(self, pos: QPoint):
        ring = self._ring_bond_from
        self._ring_bond_from = None
        self._snap_hint = None
        self._mouse_pos = None
        end = self._snap_ring_bond_end(ring, pos)
        end_idx = self._get_hit_atom(end)
        outline = [self._to_canvas_point(*self._positions[a]) for a in ring if a in self._positions]
        if end_idx in ring or (end_idx is None and _point_in_polygon(end.x(), end.y(), outline)):
            self.update()   # released inside the ring: nothing to draw
            return
        self.requestHistorySave.emit()
        if end_idx is None:
            end_idx = self._add_atom(end, depth_from=ring[0])
        rb = {"ring": list(ring), "atom": end_idx}
        if rb not in self.molecule.ring_bonds:
            self.molecule.ring_bonds.append(rb)
        self.moleculeChanged.emit()
        self.request_render()
        self.update()

    # ── text tool: edit atom labels ──────────────────────────────────────────

    def _get_hit_label(self, pos: QPoint) -> Optional[int]:
        """The atom whose drawn label (letters) is under *pos*, if any."""
        for i, (x0, x1, y0, y1) in self._label_boxes.items():
            if x0 <= pos.x() <= x1 and y0 <= pos.y() <= y1:
                return i
        return None

    def _default_label_color(self, idx: int) -> str:
        """The label color atom *idx* gets without a color of its own."""
        if self.label_color_mode == "black":
            bg = self.background.lstrip("#")
            try:
                lum = (0.299 * int(bg[0:2], 16) + 0.587 * int(bg[2:4], 16) + 0.114 * int(bg[4:6], 16)) / 255.0
            except (ValueError, IndexError):
                lum = 1.0
            return "#000000" if lum >= 0.4 else "#ffffff"
        elem = self.molecule.atoms[idx].element
        return self.color_overrides.get(elem) or CPK_BASE.get(elem, DEFAULT_BASE)

    def set_text_mode(self, enabled: bool):
        if not enabled:
            self._commit_text_editor()
        self.text_mode = enabled
        self.setCursor(QCursor(Qt.CursorShape.IBeamCursor if enabled else Qt.CursorShape.OpenHandCursor))

    def _text_press(self, event, pos: QPoint) -> bool:
        """
        Text tool: click an atom drawn as letters to edit its label — text,
        color and boldness. Clicking elsewhere confirms the open edit; it
        never adds atoms. Right-clicks are left to build mode (delete).
        Returns whether the press was handled.
        """
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if self._text_editor is not None:
            self._commit_text_editor()
            return True
        idx = self._get_hit_label(pos)
        if idx is not None:
            self._open_text_editor(idx)
        return True

    def _open_text_editor(self, idx: int):
        self._close_text_editor()
        atom = self.molecule.atoms[idx]
        bold = atom.label_bold is not False
        color = atom.label_color or self._default_label_color(idx)
        font = QFont(self.font_family.split(",")[0].strip())
        font.setPixelSize(max(10, round(self.font_size_px * self._eff_zoom)))
        font.setBold(bold)
        editor = _TextEditor(self._atom_label_text(idx), color, bold, font, self)
        editor.edit.setMaxLength(6)
        editor.setProperty("anchor_center", self._to_canvas_point(*self._positions[idx]))
        editor.committed.connect(self._commit_text_editor)
        editor.cancelled.connect(self._close_text_editor)
        editor._fit_width()
        editor.show()
        editor.edit.setFocus()
        editor.edit.selectAll()
        self._text_editor = editor
        self._editing_atom = idx
        self._editing_initial = (color, bold)

    def _commit_text_editor(self) -> bool:
        """Apply the open label edit. An unreadable label keeps the editor open (marked red)."""
        editor = self._text_editor
        if editor is None:
            return True
        parsed = self._parse_atom_label(editor.edit.text())
        if parsed is None:
            editor.set_invalid(True)
            return False
        idx = self._editing_atom
        atom = self.molecule.atoms[idx]
        # Only store a style that was actually changed here (or already set),
        # so untouched labels keep following the element / theme default.
        new_color = editor.color if (editor.color != self._editing_initial[0] or atom.label_color) else None
        new_bold = editor.bold if (editor.bold != self._editing_initial[1] or atom.label_bold is not None) else None
        self._close_text_editor()
        symbol, n_h = parsed
        want_h, cur_h = self._label_h_target(idx, symbol, n_h)
        label_changed = symbol != atom.element or want_h != len(cur_h)
        if not label_changed and (new_color, new_bold) == (atom.label_color, atom.label_bold):
            return True
        self.requestHistorySave.emit()
        atom.label_color, atom.label_bold = new_color, new_bold
        if label_changed:
            self._apply_atom_label(idx, symbol, want_h, cur_h)
            self.moleculeChanged.emit()
        self.request_render()
        return True

    def _close_text_editor(self):
        editor = self._text_editor
        self._text_editor = None
        self._editing_atom = None
        if editor is not None:
            editor.blockSignals(True)
            editor.hide()
            editor.deleteLater()
            self.setFocus()

    def _rotate_layout(self, screen_angle: float):
        """Rotate the depiction by *screen_angle* (radians, clockwise on screen)."""
        self._rotate_layout_by(-screen_angle)  # layout y points up, screen y points down

    def _rotate_layout_by(self, a: float):
        """
        Rotate the depiction by *a* radians (counter-clockwise, y up) about
        the projection of the 3D centroid, and tell the 3D view to turn by
        the same amount about its viewing axis.
        """
        ca, sa = math.cos(a), math.sin(a)
        self._positions = {
            i: (x * ca - y * sa, x * sa + y * ca)
            for i, (x, y) in self._positions.items()
        }
        self.view_rot = _rot_z(a) @ self.view_rot
        _annotate_double_bond_insets(self.molecule, self._positions, self._bond_info)
        self.viewChanged.emit(self.view_rot.copy())
        self.request_render()

    # ── align tool ───────────────────────────────────────────────────────────

    def _align_bond(self, bi: int, horizontal: bool):
        """Turn the depiction in the plane so bond *bi* is vertical (or horizontal), by the shorter way."""
        b = self.molecule.bonds[bi]
        if b.i not in self._positions or b.j not in self._positions:
            return
        (x1, y1), (x2, y2) = self._positions[b.i], self._positions[b.j]
        if math.hypot(x2 - x1, y2 - y1) < 1e-9:
            return
        theta = math.atan2(y2 - y1, x2 - x1)
        base = 0.0 if horizontal else math.pi / 2
        delta = min(((base + k * math.pi - theta + math.pi) % (2 * math.pi) - math.pi for k in (0, 1)), key=abs)
        if abs(delta) > 1e-6:
            self._rotate_layout_by(delta)

    def align_plane(self, indices: List[int]) -> bool:
        """
        Draw the molecule in the plane of the 3 atoms *indices*: turn the 3D
        view so that plane faces the screen, and redraw the 2D layout as the
        literal projection of the 3D geometry onto it (atoms off that plane
        show as wedge/hash). Returns False if the atoms are collinear.
        """
        mol = self.molecule
        if mol is None or len(indices) != 3:
            return False
        p = [np.array([mol.atoms[i].x, mol.atoms[i].y, mol.atoms[i].z]) for i in indices]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        n = float(np.linalg.norm(normal))
        if n < 1e-10:
            return False
        rot = face_plane_rotation(normal / n, self.view_rot)
        self.view_rot = rot
        origin = self._atoms_centroid()
        self._plane = (normal / n, p[0])
        self._positions, self._bond_info = generate_2d_layout(
            mol, self.planarity_tolerance, rot, projection=True, plane=self._plane)
        _apply_drawn_stereo(mol, self._bond_info)
        self._frame_origin = origin
        zoom, pan = self._zoom, self._pan.copy()
        self._fit_view()  # resets zoom (and announces it); put the shared zoom back and re-announce it
        self._pan = pan
        self._set_zoom(zoom)
        self.viewChanged.emit(self.view_rot.copy())
        self.request_render()
        return True

    def mouseDoubleClickEvent(self, event):
        # Labels are edited with the Text tool only; a double-click with no tool resets the view.
        if not (self.build_mode or self.text_mode):
            self.reset_view()

    def _atom_label_text(self, idx: int) -> str:
        """What the atom's label reads: its element, plus "H"/"Hn" for hydrogens hidden on it."""
        text = self.molecule.atoms[idx].element
        n_h = len(self._hidden_h_atoms(idx))
        if n_h:
            text += "H" + (str(n_h) if n_h > 1 else "")
        return text

    @staticmethod
    def _parse_atom_label(text: str) -> Optional[Tuple[str, Optional[int]]]:
        """
        Parse an edited label into (element, hydrogen count or None if no H
        part was typed): "N", "cl", "CH", "CH3", or the reversed "H3C".
        """
        text = text.strip()
        norm = lambda t: t[:1].upper() + t[1:].lower()
        if norm(text) in ATOMIC_MASSES:
            return norm(text), None
        for n in (2, 1):
            m = re.fullmatch(r"[Hh](\d*)", text[n:])
            if norm(text[:n]) in ATOMIC_MASSES and m:
                return norm(text[:n]), int(m.group(1) or 1)
            m = re.fullmatch(r"[Hh](\d*)", text[:-n] if len(text) > n else "x")
            if m and norm(text[-n:]) in ATOMIC_MASSES:
                return norm(text[-n:]), int(m.group(1) or 1)
        return None

    def _label_h_target(self, idx: int, symbol: str, n_h: Optional[int]) -> Tuple[int, List[int]]:
        """(hydrogen count the edited label asks for, hidden H atoms atom *idx* has now)."""
        mol = self.molecule
        cur_h = self._hidden_h_atoms(idx)
        # An H count only means something once the molecule carries explicit H
        # atoms; in a bare heavy-atom sketch hydrogens stay implicit.
        has_h_atoms = any(a.element == "H" for a in mol.atoms)
        want_h = n_h if (n_h is not None and has_h_atoms and mol.atoms[idx].element != "H") else len(cur_h)
        return want_h, cur_h

    def _apply_atom_label(self, idx: int, symbol: str, want_h: int, cur_h: List[int]):
        """Set atom *idx*'s element and add/remove hidden H atoms to reach *want_h*."""
        self.molecule.atoms[idx].element = symbol
        if want_h < len(cur_h):
            for h_idx in sorted(cur_h[want_h:], reverse=True):
                self._delete_atom(h_idx)
                if h_idx < idx:
                    idx -= 1
        elif want_h > len(cur_h):
            self._add_hidden_hydrogens(idx, want_h - len(cur_h))

    def _add_hidden_hydrogens(self, idx: int, count: int):
        """Attach *count* new H atoms to atom *idx*, placed opposite its existing bonds in 3D and 2D."""
        mol = self.molecule
        c = mol.atoms[idx]
        cpos = np.array([c.x, c.y, c.z])
        away = np.zeros(3)
        for b in mol.bonds:
            if idx in (b.i, b.j):
                o = mol.atoms[b.j if b.i == idx else b.i]
                v = np.array([o.x, o.y, o.z]) - cpos
                n = np.linalg.norm(v)
                if n > 1e-6:
                    away -= v / n
        if np.linalg.norm(away) < 1e-6:
            away = np.array([0.0, 0.0, 1.0])
        away /= np.linalg.norm(away)
        ref = np.array([1.0, 0.0, 0.0]) if abs(away[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(away, ref); u /= np.linalg.norm(u)
        v2 = np.cross(away, u)
        px, py = self._positions.get(idx, (c.x, c.y))
        for k in range(count):
            ang = 2 * math.pi * k / count
            d = away + (0.8 if count > 1 else 0.0) * (math.cos(ang) * u + math.sin(ang) * v2)
            d /= np.linalg.norm(d)
            pos = cpos + 1.09 * d
            mol.atoms.append(Atom("H", float(pos[0]), float(pos[1]), float(pos[2])))
            h_idx = len(mol.atoms) - 1
            self._positions[h_idx] = (px + float(d[0]) * 0.9, py + float(d[1]) * 0.9)
            self._add_bond(idx, h_idx)

    def paintEvent(self, event):
        super().paintEvent(event)
        draw_green = self.align_mode and self._green_atoms and self.molecule
        draw_bond = (self.build_mode and self._bonding_from is not None
                     and self._mouse_pos is not None and self.molecule
                     and self._bonding_from in self._positions)
        draw_measure = self.measure_mode and self._measure_atoms and self.molecule
        draw_measure_drag = (self.measure_mode and self._measure_from is not None
                             and self._mouse_pos is not None and self.molecule
                             and self._measure_from in self._positions)
        draw_angles = self.measure_mode and (self._angle_atoms or self._angle_leg1 is not None) and self.molecule
        draw_ring_bond = self.build_mode and self._ring_bond_from is not None and self._mouse_pos is not None
        draw_snap = self.build_mode and self.snapping and not self.text_mode and self._snap_hint is not None
        if not (draw_green or draw_bond or draw_measure or draw_measure_drag or draw_angles
                or draw_ring_bond or draw_snap):
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if draw_green:
            r = max(14.0, self.font_size_px * self._eff_zoom * 0.7)
            for idx in self._green_atoms:
                if idx not in self._positions:
                    continue
                ax, ay = self._to_canvas_point(*self._positions[idx])
                p.setPen(QPen(QColor(0, 200, 0), 2))
                p.setBrush(QColor(0, 200, 0, 60))
                p.drawEllipse(QPointF(ax, ay), r, r)
        if draw_bond:
            p.setPen(QColor("#007bff"))
            ax, ay = self._to_canvas_point(*self._positions[self._bonding_from])
            p.drawLine(int(ax), int(ay), self._mouse_pos.x(), self._mouse_pos.y())
        if draw_ring_bond:
            c = self._ring_center_canvas(self._ring_bond_from)
            if c:
                p.setPen(QPen(QColor("#007bff"), 1.5))
                p.drawLine(QPointF(*c), QPointF(self._mouse_pos))
        if draw_snap:
            x, y, kind = self._snap_hint
            p.setPen(QPen(QColor("#ff8c00"), 1.6))
            p.setBrush(QColor(255, 140, 0, 50))
            p.drawEllipse(QPointF(x, y), 6.0, 6.0)
            if kind == "ring":
                p.drawLine(QPointF(x - 4, y), QPointF(x + 4, y))
                p.drawLine(QPointF(x, y - 4), QPointF(x, y + 4))

        if draw_measure or draw_measure_drag or draw_angles:
            self._paint_measurements(p, draw_measure, draw_measure_drag, draw_angles)

    def _paint_measurements(self, p, draw_measure, draw_measure_drag, draw_angles):
        """
        Distance/angle overlay for Measure Mode — mirrors MoleculeCanvas's
        (the 3D view's) own drawing, but positions come from this view's own
        2D screen coordinates; the numbers are always real 3D distances/
        angles (from Atom.x/y/z), the same values the 3D view reports.
        """
        r = max(10.0, self.font_size_px * self._eff_zoom * 0.35)
        font = p.font()
        font.setPointSize(10)
        font.setBold(True)
        p.setFont(font)

        if draw_measure:
            for i1, i2 in self._measure_atoms:
                if i1 not in self._positions or i2 not in self._positions:
                    continue
                x1, y1 = self._to_canvas_point(*self._positions[i1])
                x2, y2 = self._to_canvas_point(*self._positions[i2])
                a1, a2 = self.molecule.atoms[i1], self.molecule.atoms[i2]
                dist = math.dist((a1.x, a1.y, a1.z), (a2.x, a2.y, a2.z))
                p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.SolidLine))
                p.drawLine(int(x1), int(y1), int(x2), int(y2))
                p.setPen(QColor("red"))
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                p.drawText(int(mx) + 6, int(my) - 6, f"{dist:.2f} Å")

        if draw_measure_drag:
            ax, ay = self._to_canvas_point(*self._positions[self._measure_from])
            p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.SolidLine))
            p.setBrush(QColor(255, 0, 0, 60))
            p.drawEllipse(QPointF(ax, ay), r, r)
            p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(int(ax), int(ay), self._mouse_pos.x(), self._mouse_pos.y())

        if draw_angles:
            def _draw_angle_arc(pts_2d, pts_3d, color):
                x1, y1 = pts_2d[0]
                vx, vy = pts_2d[1]
                x3, y3 = pts_2d[2]
                a1, a2, a3 = pts_3d
                v1 = np.array([a1.x - a2.x, a1.y - a2.y, a1.z - a2.z])
                v2 = np.array([a3.x - a2.x, a3.y - a2.y, a3.z - a2.z])
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-9 or n2 < 1e-9:
                    return
                cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angle_deg = math.degrees(math.acos(cos_angle))
                angle1 = math.degrees(math.atan2(-(y1 - vy), x1 - vx))
                angle2 = math.degrees(math.atan2(-(y3 - vy), x3 - vx))
                span = angle2 - angle1
                if span > 180:
                    span -= 360
                elif span < -180:
                    span += 360
                arc_radius = 28
                rect = QRectF(vx - arc_radius, vy - arc_radius, arc_radius * 2, arc_radius * 2)
                fill_path = QPainterPath()
                fill_path.moveTo(vx, vy)
                fill_path.arcTo(rect, angle1, span)
                fill_path.closeSubpath()
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(color.red(), color.green(), color.blue(), 60))
                p.drawPath(fill_path)
                outline_path = QPainterPath()
                start_x = vx + arc_radius * math.cos(math.radians(angle1))
                start_y = vy - arc_radius * math.sin(math.radians(angle1))
                outline_path.moveTo(start_x, start_y)
                outline_path.arcTo(rect, angle1, span)
                p.setPen(QPen(color, 2, Qt.PenStyle.SolidLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(outline_path)
                mid_angle_rad = math.radians(angle1 + span / 2)
                label_r = arc_radius + 14
                lx = vx + label_r * math.cos(mid_angle_rad)
                ly = vy - label_r * math.sin(mid_angle_rad)
                p.setPen(color)
                p.drawText(int(lx), int(ly), f"{angle_deg:.1f}°")

            for i1, i2, i3 in self._angle_atoms:
                if any(i not in self._positions for i in (i1, i2, i3)):
                    continue
                x1, y1 = self._to_canvas_point(*self._positions[i1])
                vx, vy = self._to_canvas_point(*self._positions[i2])
                x3, y3 = self._to_canvas_point(*self._positions[i3])
                p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.SolidLine))
                p.drawLine(int(vx), int(vy), int(x1), int(y1))
                p.drawLine(int(vx), int(vy), int(x3), int(y3))
                pts_3d = [self.molecule.atoms[i] for i in (i1, i2, i3)]
                _draw_angle_arc([(x1, y1), (vx, vy), (x3, y3)], pts_3d, QColor("red"))

            if self._angle_leg1 is not None and all(i in self._positions for i in self._angle_leg1):
                i1, i2 = self._angle_leg1
                x1, y1 = self._to_canvas_point(*self._positions[i1])
                x2, y2 = self._to_canvas_point(*self._positions[i2])
                p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.SolidLine))
                p.drawLine(int(x1), int(y1), int(x2), int(y2))
                for x, y in ((x1, y1), (x2, y2)):
                    p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.SolidLine))
                    p.setBrush(QColor(255, 0, 0, 60))
                    p.drawEllipse(QPointF(x, y), r, r)
                if self._mouse_pos is not None:
                    p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.DashLine))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawLine(int(x2), int(y2), self._mouse_pos.x(), self._mouse_pos.y())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.measure_mode and (self._measure_atoms or self._measure_from is not None
                                      or self._angle_atoms or self._angle_leg1 is not None):
                self._measure_atoms.clear()
                self._measure_from = None
                self._mouse_pos = None
                self._angle_atoms.clear()
                self._angle_leg1 = None
                self.update()
                return
            if self.selected_atoms or self._green_atoms:
                self.selected_atoms.clear()
                self._green_atoms.clear()
                self.update()
                self.request_render()
                return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        if self.molecule is None:
            return
        pixel = event.pixelDelta()
        angle_y = event.angleDelta().y()
        if not pixel.isNull():
            sign = 1 if angle_y >= 0 else -1
            factor = 1.0 + sign * (abs(pixel.y()) / 120.0) * 1.5
        elif angle_y != 0:
            factor = 1.12 if angle_y > 0 else (1 / 1.12)
        else:
            return
        self._set_zoom(self._zoom * factor)
        self.request_render(delay_ms=0)

    def _set_zoom(self, zoom: float):
        """Set the shared zoom (clamped like the 3D view's) and tell the 3D view and slider."""
        self._zoom = max(0.15, min(6.0, zoom))
        self.zoomChanged.emit(self._zoom)

    def event(self, event):
        # Trackpad pinch (macOS native gesture) zooms like the scroll wheel does.
        if event.type() == QEvent.Type.NativeGesture and self.molecule is not None \
                and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self._set_zoom(self._zoom * (1.0 + event.value()))
            self.request_render(delay_ms=0)
            return True
        return super().event(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.request_render(delay_ms=0)
