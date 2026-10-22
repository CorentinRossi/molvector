"""
molvector_gui.py — Interactive 3D Molecule Viewer
==============================================
PyQt6 GUI with full menu bar, appearance controls, and atom colour editor.

Default controls:
  Left-drag            Rotate molecule
  Right-drag           Pan
  Scroll               Zoom
  Shift+drag           Rotate ard. bond

Build mode (B):
  Left-click   Add atom
  Drag         Create bond
  Click bond   Change bond order

Selection mode (S):
  Click        Select atom
  Drag         Rectangle-select

Align mode (A):
  Click bond       Align molecule vertically
  Shift+click bond  Align molecule horizontally

Measure mode (M):
  Click/drag          Measure distance
  Shift+drag          Measure angle

2D view only:
  Build subtools   Snapping toggle / bond type combo (regular, double, triple,
                   forward wedge, backward hash, ring bond) / Text tool:
                   click an atom label to edit its text, color and boldness

Menus:
  File          Open / Save As / Export SVG / Export View / Quick SVG Export / Quit
  Edit          Settings / Style / Shortcuts / Info / Edit Charge / InChI / Selection Mode / Build Mode / Align Mode
  View          Reset View / Preset orientations / Overlay Settings
  Calculations  Generate G16 Input / Calculate Rotational Constants / Calculate Dipole Moment / Open Terminal / Calculation Results
  Build         Clean (also in the mode toolbar) / Undo / Redo / Optimize / Disable Bond Order
  Help          Open Test Molecule / About

  Note: On Windows/Linux use Ctrl for shortcuts; on macOS use Cmd.

Dependencies:
    pip install PyQt6 numpy svgwrite
"""

import sys, os, math, json, tempfile, platform, shlex, shutil, re, random, string, time, datetime
from typing import List, Tuple, Optional, Dict
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QSlider, QStatusBar, QFrame, QSizePolicy,
    QGroupBox, QMessageBox, QDialog, QDialogButtonBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QColorDialog, QPushButton, QGridLayout,
    QScrollArea, QToolBar, QMenu, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QComboBox, QPlainTextEdit, QLineEdit,
    QButtonGroup, QRadioButton, QKeySequenceEdit, QToolButton,
    QGraphicsOpacityEffect, QStackedWidget, QAbstractButton,
    QStyle, QStyleOptionComboBox, QStylePainter, QListWidget, QListWidgetItem,
    QTreeWidget, QTreeWidgetItem,
)
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import (
    Qt, QByteArray, QPoint, QPointF, pyqtSignal, pyqtProperty, QTimer, QSize, QRect, QRectF,
    QUrl, QMimeData, QEvent, QObject, QThread, QProcess, QProcessEnvironment, QPropertyAnimation, QEasingCurve,
)
from PyQt6.QtGui import QAction, QActionGroup, QColor, QPalette, QFont, QCursor, QIcon, QPixmap, QImage, QPainter, QFontDatabase, QPainterPath, QPdfWriter, QPageSize, QKeySequence, QDesktopServices, QTextCursor, QPen, QLinearGradient, QBrush
from PyQt6.QtGui import QDrag

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── renderer / parsers ───────────────────────────────────────────────────────
from molvector.render import (
    parse_xyz, parse_gaussian, parse_gaussian_log, parse_pdb, parse_mol,
    infer_bonds,
    render_molecule, Molecule, CPK_BASE, CPK_DARK, DEFAULT_BASE, VDW_RADII,
    lighten, darken, hex_to_rgb, rgb_to_hex, auto_dark,
    chemical_formula, molecular_mass, VibrationalMode, ExcitedState,
    save_xyz, save_gaussian_input, save_pdb, save_mol, project_molecule,
    Atom, Bond,
    optimize_geometry, ensure_openbabel, openbabel_unavailable_message, _suppress_ob_stderr, calculate_rotational_constants, calculate_dipole_moment, generate_inchi, check_valence_issues,
    center_positions, ATOMIC_MASSES, mulliken_gradient_color,
    spin_multiplicity, gaussian_atom_label,
    MULLIKEN_NEGATIVE_COLOR, MULLIKEN_ZERO_COLOR, MULLIKEN_POSITIVE_COLOR,
)
from molvector.chemvector import (ChemCanvas, HAS_RDKIT, check_aromaticity, alt_click, ALT_CLICK_HINT,
                                  best_fit_plane_normal, face_plane_rotation)
from molvector import native_guard, paths

# Per-user settings (see molvector/paths.py); never inside the installed package.
CONFIG_FILE = paths.config_path("molvector_config.json")


def _update_config(updates: dict) -> None:
    """Merge *updates* into the settings file (raises OSError if it cannot be written)."""
    config = paths.read_json(CONFIG_FILE, {})
    if not isinstance(config, dict):
        config = {}
    config.update(updates)
    paths.write_json(CONFIG_FILE, config, indent=2)

# ── shortcut modifier prefix ─────────────────────────────────────────────────
def _mod(s: str) -> str:
    """Qt already maps 'Ctrl' → ⌘ on macOS; pass through unchanged."""
    return s


def _log_error(summary: str, exc: BaseException) -> str:
    """Append a full traceback to error.log in the per-user folder, return its path.

    Called from error handlers, so it never raises: if that folder is not
    writable it falls back to the temp folder, and returns "" if both fail.
    """
    import traceback
    for log_dir in (paths.user_dir(),
                    os.path.join(tempfile.gettempdir(), "molvector")):
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "error.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{'='*60}\n")
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {summary}\n")
                f.write(f"{'─'*60}\n")
                traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
                f.write("\n")
            return log_path
        except Exception:
            continue
    return ""


class _ErrorReporter(QObject):
    """Catches exceptions that escape Qt slots and event handlers.

    PyQt6 aborts the whole application on an unhandled exception inside a
    slot or a virtual override (paintEvent, mousePressEvent, ...).  This hook
    logs it and shows a notification instead, so one bug cancels only the
    action that hit it.  It may be called from a worker thread, so the
    notification goes through a queued signal to the GUI thread.
    """

    reported = pyqtSignal(str)
    _REPEAT_WINDOW_S = 5.0

    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self._last = ("", 0.0)
        self.reported.connect(self._notify)

    def install(self):
        sys.excepthook = self._hook

    def _hook(self, exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        if exc is None:
            exc = exc_type()
        if exc.__traceback__ is None:
            exc = exc.with_traceback(tb)
        # A failing paintEvent fires on every repaint; log and notify once.
        key = f"{exc_type.__name__}: {exc}"
        now = time.monotonic()
        if key == self._last[0] and now - self._last[1] < self._REPEAT_WINDOW_S:
            return
        self._last = (key, now)
        log_path = _log_error("Unexpected error", exc)
        try:
            if sys.stderr is not None:
                sys.__excepthook__(exc_type, exc, tb)
        except Exception:
            pass
        self.reported.emit(log_path)

    def _notify(self, log_path: str):
        try:
            self._window.show_notification(
                "Something went wrong; that action was cancelled.", "error", 0, log_file=log_path)
        except Exception:
            pass


def _monospace_font(point_size: Optional[float] = None) -> QFont:
    """The platform's fixed-width font (Menlo, Consolas, DejaVu Sans Mono, ...).

    A family name such as "monospace" or "Menlo, Courier New" is not resolved
    the same way on every platform and can fall back to a proportional font.
    """
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    if point_size is not None:
        font.setPointSizeF(point_size)
    return font


def load_colored_icon(svg_path: str, color: str, size: int = 22) -> QIcon:
    """Load an SVG file, replace #000000 fills with color, return QIcon."""
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    svg_content = svg_content.replace('#000000', color)
    renderer = QSvgRenderer(QByteArray(svg_content.encode("utf-8")))
    default_size = renderer.defaultSize()
    if default_size.isValid() and default_size.width() > 0 and default_size.height() > 0:
        aspect = default_size.width() / default_size.height()
        if aspect >= 1.0:
            w, h = size, int(size / aspect)
        else:
            w, h = int(size * aspect), size
    else:
        w = h = size
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    x = (size - w) // 2
    y = (size - h) // 2
    renderer.render(painter, QRectF(x, y, w, h))
    painter.end()
    return QIcon(pix)

BOND_TYPES = [   # (kind, combo label, tooltip)
    ("single", "Regular",  "Regular bond — click a bond to cycle its order"),
    ("double", "Double",   "Double bond"),
    ("triple", "Triple",   "Triple bond"),
    ("wedge",  "Forward",  "Forward bond (wedge, toward the viewer) — click a wedge again to flip it"),
    ("hash",   "Backward", "Backward bond (hash, away from the viewer) — click a hash again to flip it"),
    ("ring",   "Ring bond", "Ring bond — drag from a ring's center out over one of its bonds "
                            "(substituent at an undetermined ortho/meta/para position)"),
]


def bond_type_icon(kind: str, color: str, size: int = 22) -> QIcon:
    """Small drawn icon for a 2D build bond type (see BOND_TYPES)."""
    pix = QPixmap(size * 2, size * 2)   # drawn at 2x for crisp HiDPI
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(2, 2)
    col = QColor(color)
    pen = QPen(col, 1.8)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    a, b = QPointF(4, size - 5), QPointF(size - 4, 5)
    if kind in ("single", "double", "triple"):
        n = {"single": 1, "double": 2, "triple": 3}[kind]
        for k in range(n):
            off = (k - (n - 1) / 2) * 3.6
            d = QPointF(off * 0.7071, off * 0.7071)
            p.drawLine(a + d, b + d)
    elif kind == "wedge":
        p.setBrush(col)
        p.setPen(Qt.PenStyle.NoPen)
        path = QPainterPath(a)
        path.lineTo(b + QPointF(-2.8, -2.8))
        path.lineTo(b + QPointF(2.8, 2.8))
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "hash":
        p.setPen(QPen(col, 1.4))
        for k in range(1, 6):
            t = k / 5
            c = a + (b - a) * t
            w = 3.0 * t
            p.drawLine(c + QPointF(-w, -w), c + QPointF(w, w))
    elif kind == "ring":
        import math as _m
        cx, cy, r = size * 0.42, size * 0.58, size * 0.3
        pts = [QPointF(cx + r * _m.cos(_m.radians(30 + 60 * k)), cy + r * _m.sin(_m.radians(30 + 60 * k)))
               for k in range(6)]
        p.setPen(QPen(col, 1.4))
        for k in range(6):
            p.drawLine(pts[k], pts[(k + 1) % 6])
        p.setPen(pen)
        p.drawLine(QPointF(cx, cy), QPointF(size - 3, 3))
    p.end()
    pix.setDevicePixelRatio(2)
    return QIcon(pix)


class _IconComboBox(QComboBox):
    """Combo box that shows only the current item's icon when closed; the
    item names appear in the opened list."""

    def _closed_option(self) -> QStyleOptionComboBox:
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        opt.currentText = ""
        return opt

    def paintEvent(self, event):
        p = QStylePainter(self)
        opt = self._closed_option()
        p.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        p.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)

    def sizeHint(self) -> QSize:
        return self.style().sizeFromContents(
            QStyle.ContentsType.CT_ComboBox, self._closed_option(), self.iconSize(), self)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def showPopup(self):
        view = self.view()
        view.setMinimumWidth(view.sizeHintForColumn(0) + 24)
        super().showPopup()


def set_icons(app, win):
    system = platform.system()
    if system == "Darwin":
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "molvector_macos.png")
    elif system == "Linux":
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "molvector_linux.png")
    elif system == "Windows":
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "molvector_windows.png")
    else:
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icons", "icon.svg")
    
    if not os.path.isfile(icon_path):
        return
    
    icon = QIcon(icon_path)
    app.setWindowIcon(icon)
    win.setWindowIcon(icon)

class _OptimizeWorker(QThread):
    """Runs the force-field Clean off the GUI thread.

    Loading OpenBabel the first time (crash probe included) and a long
    optimization take seconds; done here the window keeps repainting.  The
    worker optimizes a copy, so the GUI may keep drawing the original;
    result is (coords, steps, error) where error is None, "unavailable"
    or the exception.
    """

    result = pyqtSignal(object, int, object)

    def __init__(self, mol: Molecule, max_steps: int, tol: float, parent=None):
        super().__init__(parent)
        self.elements = [a.element for a in mol.atoms]
        self._copy = Molecule(
            name=mol.name,
            atoms=[Atom(a.element, a.x, a.y, a.z) for a in mol.atoms],
            bonds=[Bond(b.i, b.j, b.order) for b in mol.bonds],
            charge=mol.charge,
        )
        self._max_steps = max_steps
        self._tol = tol

    def run(self):
        try:
            if not ensure_openbabel():
                self.result.emit(None, 0, "unavailable")
                return
            steps = optimize_geometry(self._copy, max_steps=self._max_steps, tol=self._tol)
            coords = [(a.x, a.y, a.z) for a in self._copy.atoms]
            self.result.emit(coords, int(steps), None)
        except Exception as e:
            self.result.emit(None, 0, e)


def _geometry_key(mol) -> str:
    """Identifies a molecule's atoms and positions (for per-molecule dialog state)."""
    return save_xyz(mol) if mol is not None else ""


def _with_filter_extension(path: str, selected_filter: str, known: tuple) -> str:
    """Give *path* an extension from *known*, taken from the save dialog's filter.

    Some platform dialogs (GTK on Linux, "All files" on Windows) return the
    name exactly as typed, so "molecule" or "molecule.v2" carries no usable
    extension.  The first extension of the selected filter is then appended
    (or known[0] for an extension-less name under a filter like "All files").
    """
    current = os.path.splitext(path)[1].lower()
    if current in known:
        return path
    for ext in re.findall(r"\*(\.\w+)", selected_filter or ""):
        if ext.lower() in known:
            return path + ext.lower()
    return path if current else path + known[0]


def get_safe_filename(name: str, fallback: str = "molecule") -> str:
    """A file name valid on every platform: C60+ -> C60p, spaces -> _, other
    special characters (/ \\ : * ? ... forbidden on Windows) removed."""
    s = name.replace("+", "p").replace("-", "m").replace(" ", "_")
    return re.sub(r'[^a-zA-Z0-9_-]', '', s) or fallback


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ELEM_FULL_NAME = {
    "H":"Hydrogen", "D":"Deuterium (2H)", "T":"Tritium (3H)", "He":"Helium",  "Li":"Lithium", "Be":"Beryllium","B":"Boron",
    "C":"Carbon",   "13C":"Carbon-13", "14C":"Carbon-14", "N":"Nitrogen", "15N":"Nitrogen-15", "O":"Oxygen",   "F":"Fluorine",  "Ne":"Neon",
    "Na":"Sodium",  "Mg":"Magnesium","Al":"Aluminium","Si":"Silicon", "P":"Phosphorus",
    "S":"Sulfur",   "Cl":"Chlorine", "Ar":"Argon",   "K":"Potassium", "Ca":"Calcium",
    "Sc":"Scandium","Ti":"Titanium", "V":"Vanadium", "Cr":"Chromium", "Mn":"Manganese",
    "Fe":"Iron",     "Co":"Cobalt",   "Ni":"Nickel",   "Cu":"Copper",   "Zn":"Zinc",
    "Ga":"Gallium",  "Ge":"Germanium","As":"Arsenic",  "Se":"Selenium", "Br":"Bromine",
    "Kr":"Krypton",  "Rb":"Rubidium", "Sr":"Strontium","Y":"Yttrium",   "Zr":"Zirconium",
    "Nb":"Niobium",  "Mo":"Molybdenum","Tc":"Technetium","Ru":"Ruthenium","Rh":"Rhodium",
    "Pd":"Palladium","Ag":"Silver",   "Cd":"Cadmium",  "In":"Indium",   "Sn":"Tin",
    "Sb":"Antimony", "Te":"Tellurium","I":"Iodine",   "Xe":"Xenon",    "Cs":"Caesium",
    "Ba":"Barium",   "La":"Lanthanum", "Ce":"Cerium",   "Pr":"Praseodymium","Nd":"Neodymium",
    "Pm":"Promethium","Sm":"Samarium", "Eu":"Europium", "Gd":"Gadolinium","Tb":"Terbium",
    "Dy":"Dysprosium","Ho":"Holmium",  "Er":"Erbium",   "Tm":"Thulium",  "Yb":"Ytterbium",
    "Lu":"Lutetium", "Hf":"Hafnium",  "Ta":"Tantalum", "W":"Tungsten",  "Re":"Rhenium",
    "Os":"Osmium",   "Ir":"Iridium",  "Pt":"Platinum", "Au":"Gold",     "Hg":"Mercury",
    "Tl":"Thallium", "Pb":"Lead",     "Bi":"Bismuth",  "Po":"Polonium", "At":"Astatine",
    "Rn":"Radon",    "Fr":"Francium", "Ra":"Radium",
    "Ac":"Actinium","Th":"Thorium",  "Pa":"Protactinium","U":"Uranium",
    "Np":"Neptunium","Pu":"Plutonium","Am":"Americium","Cm":"Curium",
    "Bk":"Berkelium","Cf":"Californium","Es":"Einsteinium","Fm":"Fermium",
    "Md":"Mendelevium","No":"Nobelium","Lr":"Lawrencium",
    "Rf":"Rutherfordium","Db":"Dubnium","Sg":"Seaborgium","Bh":"Bohrium",
    "Hs":"Hassium","Mt":"Meitnerium","Ds":"Darmstadtium","Rg":"Roentgenium",
    "Cn":"Copernicium","Nh":"Nihonium","Fl":"Flerovium","Mc":"Moscovium",
    "Lv":"Livermorium","Ts":"Tennessine","Og":"Oganesson",
}

# Theme Colors
THEMES = {
    "dark": {
        "DARK_BG":  "#0f0f1a",
        "PANEL_BG": "#0d0d18",
        "CARD_BG":  "#13131f",
        "BORDER":   "#2a2a44",
        "FG":       "#ccd6f6",
        "FG_DIM":   "#99aacc",
        "ACCENT":   "#4488cc",
        "CANVAS":   "#0a0a12"
    },
    "light": {
        "DARK_BG":  "#f0f2f5",
        "PANEL_BG": "#e4e6eb",
        "CARD_BG":  "#ffffff",
        "BORDER":   "#ced4da",
        "FG":       "#1c1e21",
        "FG_DIM":   "#65676b",
        "ACCENT":   "#007bff",
        "CANVAS":   "#ffffff"
    }
}

def get_stylesheet(theme_name: str) -> str:
    t = THEMES[theme_name]
    return f"""
    QMainWindow, QDialog {{ background:{t['DARK_BG']}; }}
    QWidget {{ color:{t['FG']}; }}
    QMenuBar {{
        background:{t['CARD_BG']}; color:{t['FG']};
        border-bottom:1px solid {t['BORDER']}; padding:2px 4px;
    }}
    QMenuBar::item:selected {{ background:{t['BORDER']}; border-radius:3px; }}
    QMenu {{
        background:{t['CARD_BG']}; color:{t['FG']};
        border:1px solid {t['BORDER']}; border-radius:6px;
        padding:4px;
    }}
    QMenu::item {{ padding:5px 24px 5px 10px; border-radius:4px; }}
    QMenu::item:selected {{ background:{t['ACCENT']}; color:#fff; }}
    QMenu::separator {{ height:1px; background:{t['BORDER']}; margin:3px 6px; }}
    QToolBar {{
        background:{t['CARD_BG']}; border-bottom:1px solid {t['BORDER']};
        spacing:4px; padding:3px 8px;
    }}
    QToolBar QToolButton {{
        background:transparent; color:{t['FG']};
        border:1px solid transparent; border-radius:4px; padding:4px 8px;
    }}
    QToolBar QToolButton:hover {{ background:{t['BORDER']}; }}
    QToolBar QToolButton:checked {{
        background:{t['ACCENT']}; color:white;
        border:1px solid {t['ACCENT']};
    }}
    QPushButton {{
        background:{t['CARD_BG']}; color:{t['FG']};
        border:1px solid {t['BORDER']}; border-radius:5px;
        padding:5px 14px; font-size:12px;
    }}
    QPushButton:hover  {{ background:{t['BORDER']}; border-color:{t['ACCENT']}; }}
    QToolTip {{
        background: {t['CARD_BG']}; color: {t['FG']};
        border: 1px solid {t['ACCENT']}; border-radius: 4px;
        padding: 4px;
    }}
    QPushButton:pressed{{ background:{t['ACCENT']}; }}
    QPushButton#facetButton {{ padding:4px 10px; }}
    QPushButton#facetButton:checked {{
        background:{t['ACCENT']}; color:white; border-color:{t['ACCENT']};
    }}
    QPushButton#facetButton:disabled {{ color:{t['BORDER']}; }}
    QPushButton#facetButton[orMode="true"]:checked {{
        background:#e8890c; color:white; border-color:#e8890c;
    }}
    QPushButton#facetMode {{ padding:2px 4px; font-size:10px; font-weight:bold;
                             color:{t['ACCENT']}; border-color:{t['ACCENT']}; }}
    QPushButton#facetMode[orMode="true"] {{ color:white; background:#e8890c; border-color:#e8890c; }}
    QPushButton#accent {{
        background:#1e4080; border-color:{t['ACCENT']}; color:#fff;
    }}
    QPushButton#accent:hover {{ background:#2255aa; }}
    QPushButton#color_btn {{
        border-radius:4px; min-width:36px; min-height:24px;
        padding:2px; border:2px solid {t['BORDER']};
    }}
    QPushButton#color_btn:hover {{ border-color:{t['ACCENT']}; }}
    QLabel {{ color:{t['FG']}; }}
    QSlider::groove:horizontal {{
        height:4px; background:{t['BORDER']}; border-radius:2px;
    }}
    QSlider::handle:horizontal {{
        background:{t['ACCENT']}; border-radius:6px;
        width:14px; height:14px; margin:-5px 0;
    }}
    QSlider::sub-page:horizontal {{ background:{t['ACCENT']}; border-radius:2px; }}
    QDoubleSpinBox, QSpinBox, QComboBox {{
        background:{t['DARK_BG']}; border:1px solid {t['BORDER']};
        border-radius:4px; padding:3px 6px; color:{t['FG']};
    }}
    QComboBox::drop-down {{
        border:none; width:20px;
    }}
    QComboBox QAbstractItemView {{
        background:{t['CARD_BG']}; color:{t['FG']};
        border:1px solid {t['BORDER']}; selection-background-color:{t['ACCENT']};
        selection-color:#fff; outline:none;
    }}
    QCheckBox {{
        spacing:6px; color:{t['FG']};
    }}
    QCheckBox::indicator {{
        width:14px; height:14px; border:1px solid {t['BORDER']};
        border-radius:3px; background:{t['DARK_BG']};
    }}
    QCheckBox::indicator:checked {{
        background:{t['ACCENT']}; border-color:{t['ACCENT']};
    }}
    QGroupBox {{
        color:{t['ACCENT']}; border:1px solid {t['BORDER']};
        border-radius:6px; margin-top:10px; font-size:11px;
    }}
    QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; }}
    QStatusBar {{ background:{t['PANEL_BG']}; color:{t['FG_DIM']}; font-size:11px; }}
    QScrollArea {{ background:transparent; border:none; }}
    QScrollArea > QWidget > QWidget {{ background:transparent; }}
    QDialogButtonBox QPushButton {{ min-width:80px; }}
    QTabWidget::pane {{ border: 1px solid {t['BORDER']}; background: transparent; }}
    QTabBar::tab {{
        background: {t['DARK_BG']}; border: 1px solid {t['BORDER']};
        padding: 5px 12px; border-top-left-radius: 4px; border-top-right-radius: 4px;
    }}
    QTabBar::tab:selected {{ background: {t['CARD_BG']}; border-bottom-color: {t['CARD_BG']}; }}
    QTableWidget {{ background: {t['DARK_BG']}; gridline-color: {t['BORDER']}; border: 1px solid {t['BORDER']}; }}
    QHeaderView::section {{ background: {t['CARD_BG']}; color: {t['FG']}; border: 1px solid {t['BORDER']}; }}
    
    QLabel#dim {{
        color:{t['FG_DIM']}; font-size:11px;
    }}
    QLabel#zoom_label {{
        color:{t['ACCENT']}; font-size:11px; padding:0 4px;
    }}
    QWidget#sidebar {{
        background:{t['PANEL_BG']}; border-right:1px solid {t['BORDER']};
    }}
    QLabel#mol_name {{
        color:{t['FG']}; font-weight:bold; font-size:13px; letter-spacing:1px;
    }}
    QLabel#hint {{
        color:{t['FG_DIM']}; font-size:10px; line-height:160%;
    }}
    QToolBar {{
        background:{t['PANEL_BG']}; border-bottom:1px solid {t['BORDER']};
        spacing: 4px; padding: 2px;
    }}
    QToolButton {{
        border-radius: 4px; padding: 4px; color: {t['FG']};
        border: 1px solid transparent;
    }}
    QToolButton:hover {{ background: {t['BORDER']}; }}
    QToolButton:checked {{
        background: {t['ACCENT']}; color: white;
        border: 1px solid {t['ACCENT']};
    }}
    QToolButton:checked {{
        background: {t['ACCENT']}; color: white;
        border: 1px solid {t['ACCENT']};
    }}
    """


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR SWATCH BUTTON
# ─────────────────────────────────────────────────────────────────────────────

def color_pixmap(hex_color: str, size: int = 22) -> QPixmap:
    px = QPixmap(size, size)
    px.fill(QColor(hex_color))
    return px

class ViewModeToggle(QAbstractButton):
    """
    A small two-state slider switch for the main toolbar — a pill thumb
    slides between the two halves of a neutral-grey track (neither side
    reads as the "activated"/highlighted one), animated rather than
    snapping instantly. The "3D"/"2D" labels are drawn beside it (see
    _build_toolbar), not on the switch itself.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedSize(44, 18)
        self.setToolTip("Switch between the 3D viewer and the 2D Chemvector whiteboard")
        self._pos = 0.0  # 0.0 = left ("3D"/unchecked) .. 1.0 = right ("2D"/checked)
        self._anim = QPropertyAnimation(self, b"thumbPos", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate_to)

    def _animate_to(self, checked: bool):
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def _get_thumb_pos(self) -> float:
        return self._pos

    def _set_thumb_pos(self, value: float):
        self._pos = value
        self.update()

    thumbPos = pyqtProperty(float, _get_thumb_pos, _set_thumb_pos)

    def sizeHint(self) -> QSize:
        return QSize(44, 18)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect().adjusted(1, 1, -1, -1))
        radius = rect.height() / 2.0
        t = self._pos

        # Track color is fixed regardless of state — 2D isn't a special
        # "on" state relative to 3D, just the other side of the switch.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#787878"))
        p.drawRoundedRect(rect, radius, radius)

        # Thumb: a pill covering the active half, sliding continuously
        # between them rather than jumping.
        half_w = rect.width() / 2.0
        thumb_rect = QRectF(
            rect.left() + 1.0 + (half_w - 1.0) * t, rect.top() + 1.0,
            half_w - 1.0, rect.height() - 2.0,
        )
        p.setBrush(QColor("#d8d8d8"))
        p.drawRoundedRect(thumb_rect, radius - 1.0, radius - 1.0)


class ColorButton(QPushButton):
    """Push-button that shows a solid colour swatch and opens a colour dialog."""
    colorChanged = pyqtSignal(str)   # emits new hex string

    def __init__(self, initial_color: str = "#ffffff", parent=None):
        super().__init__(parent)
        self.setObjectName("color_btn")
        self.setFixedSize(16, 16)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._color = initial_color
        self._update_swatch()
        self.clicked.connect(self._pick)

    def color(self) -> str:
        return self._color

    def set_color(self, hex_color: str, emit: bool = True):
        orig = self._color
        self._color = hex_color
        self._update_swatch()
        if emit and orig != hex_color:
            self.colorChanged.emit(self._color)

    def _update_swatch(self):
        s = 16
        r = s // 2
        self.setStyleSheet(
            f"QPushButton#color_btn {{"
            f"  background:{self._color};"
            f"  border-radius:{r}px;"
            f"  border:1.5px solid palette(mid);"
            f"  min-width:{s}px; min-height:{s}px;"
            f"  max-width:{s}px; max-height:{s}px;"
            f"  padding:0px;"
            f"}}"
            f"QPushButton#color_btn:hover {{"
            f"  border-color:palette(highlight);"
            f"}}"
        )

    def _pick(self):
        col = QColorDialog.getColor(QColor(self._color), self, "Pick colour")
        if col.isValid():
            self._color = col.name()
            self._update_swatch()
            self.colorChanged.emit(self._color)


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS DIALOG  (Tabbed: General, Overlay, Atoms & Bonds)
# ─────────────────────────────────────────────────────────────────────────────

POSITION_LABELS = {
    "top-left":      "Top left",
    "top":           "Top",
    "top-right":     "Top right",
    "left":          "Left",
    "center":        "Center",
    "right":         "Right",
    "bottom-left":   "Bottom left",
    "bottom":        "Bottom",
    "bottom-right":  "Bottom right",
}

class LightPositionDialog(QDialog):
    def __init__(self, current="top-left", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Light Position")
        self.setFixedSize(320, 260)
        self._selected = current

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        grid = QGridLayout()
        grid.setSpacing(6)

        positions = [
            ("top-left", 0, 0), ("top", 0, 1), ("top-right", 0, 2),
            ("left",     1, 0), ("center", 1, 1), ("right",     1, 2),
            ("bottom-left", 2, 0), ("bottom", 2, 1), ("bottom-right", 2, 2),
        ]

        self._btns = {}
        for key, r, c in positions:
            label = POSITION_LABELS[key]
            btn = QPushButton(label)
            btn.setFixedSize(90, 40)
            btn.setCheckable(True)
            btn.setChecked(key == current)
            if key == current:
                btn.setStyleSheet(
                    "QPushButton { background-color: #4488cc; color: white; "
                    "font-weight: bold; border: 2px solid #66aaff; border-radius: 4px; }"
                )
            btn.clicked.connect(lambda checked, k=key: self._select(k))
            grid.addWidget(btn, r, c)
            self._btns[key] = btn

        layout.addLayout(grid)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btn_row.addWidget(btns)
        layout.addLayout(btn_row)

    def _select(self, key: str):
        old = self._selected
        self._selected = key
        if old in self._btns:
            self._btns[old].setChecked(False)
            self._btns[old].setStyleSheet("")
        self._btns[key].setChecked(True)
        self._btns[key].setStyleSheet(
            "QPushButton { background-color: #4488cc; color: white; "
            "font-weight: bold; border: 2px solid #66aaff; border-radius: 4px; }"
        )

    @property
    def position(self) -> str:
        return self._selected


class SettingsDialog(QDialog):
    CONFIG_FILE = CONFIG_FILE

    def __init__(self, theme, bg_color, atom_scale, bond_width, bond_style, color_overrides,
                 atom_border_mode="scaled", atom_border_scale=1.04, atom_border_width=2.0,
                 bond_color="#444444", lighting_intensity=1.0,
                 light_position="top-left", roughness=1.0,
                 show_axes=False, show_principal_axes=False,
                 axes_position="bottom-left", principal_axes_position="bottom-left",
                 restore_molecule=False, start_view="3D", fast_render=False, show_debug_menu=False, live_callback=None,
                 shortcut_actions=None, alt_shortcut_defaults=None, parent=None,
                 show_dipole_arrow=False, quick_svg_widget=False,
                 chem_label_colors="cpk", chem_layout="match3d", chem_callback=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        self._live_callback = live_callback
        self._chem_callback = chem_callback
        self._color_overrides = color_overrides

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_general_tab(theme, restore_molecule, start_view=start_view, fast_render=fast_render,
                                show_debug_menu=show_debug_menu, quick_svg_widget=quick_svg_widget)
        self._build_shortcuts_tab(shortcut_actions or {}, alt_shortcut_defaults or {})
        self._build_overlay_tab(bg_color, show_axes, show_principal_axes, axes_position, principal_axes_position,
                                show_dipole_arrow=show_dipole_arrow)
        self._build_atoms_bonds_tab(atom_scale, bond_width, bond_style,
                                     atom_border_mode, atom_border_scale, atom_border_width,
                                     bond_color, lighting_intensity, light_position, roughness,
                                     fast_render=fast_render)
        self._build_chem2d_tab(chem_label_colors, chem_layout)

        btn_row = QHBoxLayout()
        btn_make_config = QPushButton("Make Default")
        btn_make_config.clicked.connect(self._save_config)
        btn_restore = QPushButton("Restore Defaults")
        btn_restore.clicked.connect(self._restore_defaults)
        btn_row.addWidget(btn_make_config)
        btn_row.addWidget(btn_restore)
        btn_row.addStretch()
        dialog_btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        dialog_btns.accepted.connect(self.accept)
        dialog_btns.rejected.connect(self.reject)
        btn_row.addWidget(dialog_btns)
        layout.addLayout(btn_row)

    def _build_general_tab(self, theme, restore_molecule, start_view="3D", fast_render=False, show_debug_menu=False,
                           quick_svg_widget=False):
        page = QWidget()
        form = QFormLayout(page)
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._restore_cb = QCheckBox("Restore molecule")
        self._restore_cb.setChecked(restore_molecule)
        self._restore_cb.toggled.connect(self._on_change)
        form.addRow("At login:", self._restore_cb)

        self._quick_svg_widget_cb = QCheckBox("Activated")
        self._quick_svg_widget_cb.setChecked(quick_svg_widget)
        self._quick_svg_widget_cb.toggled.connect(self._on_change)
        form.addRow("Quick SVG export widget:", self._quick_svg_widget_cb)

        self._start_view_combo = QComboBox()
        self._start_view_combo.addItems(["3D", "2D"])
        self._start_view_combo.setCurrentText(start_view)
        self._start_view_combo.setMinimumWidth(100)
        self._start_view_combo.setToolTip("Which view (3D or 2D Chemvector) the app opens in — takes effect next launch")
        self._start_view_combo.currentTextChanged.connect(self._on_change)
        form.addRow("Starts in:", self._start_view_combo)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["Dark", "Light"])
        self._theme_combo.setCurrentText(theme.capitalize())
        self._theme_combo.setMinimumWidth(100)
        self._theme_combo.currentTextChanged.connect(self._on_change)
        form.addRow("Theme:", self._theme_combo)

        self._fast_render_cb = QCheckBox("Fast render mode (flat atoms during drag)")
        self._fast_render_cb.setChecked(fast_render)
        self._fast_render_cb.stateChanged.connect(self._on_change)
        form.addRow("Performance:", self._fast_render_cb)

        self._debug_menu_cb = QCheckBox("Show debug menu")
        self._debug_menu_cb.setChecked(show_debug_menu)
        self._debug_menu_cb.stateChanged.connect(self._on_change)
        form.addRow("Debug:", self._debug_menu_cb)

        self.tabs.addTab(page, "&General")

    def _build_chem2d_tab(self, label_colors: str, layout_mode: str = "match3d"):
        page = QWidget()
        form = QFormLayout(page)
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._chem_label_combo = QComboBox()
        self._chem_label_combo.addItems(["CPK colors", "All black"])
        self._chem_label_combo.setCurrentIndex(1 if label_colors == "black" else 0)
        self._chem_label_combo.currentIndexChanged.connect(self._on_change)
        form.addRow("Atom letters:", self._chem_label_combo)

        self._chem_layout_combo = QComboBox()
        self._chem_layout_combo.addItems(["Match 3D view", "Clean depiction"])
        self._chem_layout_combo.setCurrentIndex(1 if layout_mode == "clean" else 0)
        self._chem_layout_combo.setToolTip(
            "Match 3D view: every atom is drawn exactly where the 3D view has it, so switching "
            "views never shifts the molecule.\nClean depiction: idealized bond angles (RDKit); "
            "turned to match the 3D view, but atoms can shift when switching."
        )
        self._chem_layout_combo.currentIndexChanged.connect(self._on_change)
        form.addRow("2D layout:", self._chem_layout_combo)

        self.tabs.addTab(page, "&2D")

    @property
    def chem_label_colors(self) -> str:
        return "black" if self._chem_label_combo.currentIndex() == 1 else "cpk"

    @property
    def chem_layout(self) -> str:
        return "clean" if self._chem_layout_combo.currentIndex() == 1 else "match3d"

    def _build_shortcuts_tab(self, shortcut_actions: dict, alt_defaults: dict):
        page = QWidget()
        vl = QVBoxLayout(page)
        vl.setSpacing(6)

        self._shortcut_actions = dict(shortcut_actions)
        self._sc_alt_defaults  = dict(alt_defaults)

        # Conflict warning
        self._sc_conflict_lbl = QLabel()
        self._sc_conflict_lbl.setVisible(False)
        vl.addWidget(self._sc_conflict_lbl)

        # Table
        self._sc_table = QTableWidget(len(self._shortcut_actions), 3)
        self._sc_table.setHorizontalHeaderLabels(["Action", "Primary", "Alt"])
        self._sc_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._sc_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._sc_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._sc_table.verticalHeader().setVisible(False)
        self._sc_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        self._sc_editors = []
        for row, (aid, action) in enumerate(self._shortcut_actions.items()):
            name_item = QTableWidgetItem(action.text().replace("&", ""))
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._sc_table.setItem(row, 0, name_item)

            ks1 = QKeySequenceEdit(action.shortcut())
            ks1.setClearButtonEnabled(True)
            self._sc_table.setCellWidget(row, 1, ks1)
            ks1.keySequenceChanged.connect(self._sc_check_conflicts)

            alt_seq = self._sc_alt_defaults.get(aid, "")
            ks2 = QKeySequenceEdit(QKeySequence(alt_seq))
            ks2.setClearButtonEnabled(True)
            self._sc_table.setCellWidget(row, 2, ks2)
            ks2.keySequenceChanged.connect(self._sc_check_conflicts)

            self._sc_editors.append((ks1, ks2))

        self._sc_check_conflicts()
        vl.addWidget(self._sc_table)

        # Buttons
        btn_row = QHBoxLayout()
        btn_restore = QPushButton("Restore Defaults")
        btn_restore.clicked.connect(self._sc_restore_defaults)
        btn_row.addWidget(btn_restore)

        btn_make_default = QPushButton("Make Default")
        btn_make_default.clicked.connect(self._sc_make_default)
        btn_row.addWidget(btn_make_default)
        btn_row.addStretch()
        vl.addLayout(btn_row)

        self.tabs.addTab(page, "&Shortcuts")

    def _sc_check_conflicts(self):
        usage = {}
        for row, (aid, action) in enumerate(self._shortcut_actions.items()):
            for col, editor in enumerate([self._sc_editors[row][0], self._sc_editors[row][1]]):
                ks = editor.keySequence()
                if not ks.isEmpty():
                    seq_str = ks.toString().lower()
                    if seq_str not in usage:
                        usage[seq_str] = []
                    usage[seq_str].append((aid, action.text().replace("&", ""), col))

        conflicts = {k: v for k, v in usage.items() if len(v) > 1}

        if conflicts:
            lines = []
            for seq, items in conflicts.items():
                names = [f"{name} ({'Alt' if c == 1 else 'Primary'})" for _, name, c in items]
                lines.append(f"⚠ {seq} used by: {', '.join(names)}")
            self._sc_conflict_lbl.setText(
                '<span style="color:#ff6b6b; font-weight:bold;">' +
                "<br>".join(lines) +
                '</span>'
            )
            self._sc_conflict_lbl.setVisible(True)
        else:
            self._sc_conflict_lbl.setVisible(False)

    def _sc_restore_defaults(self):
        for row, (aid, action) in enumerate(self._shortcut_actions.items()):
            primary = MainWindow.DEFAULT_SHORTCUTS.get(aid, "")
            alt = MainWindow.ALT_SHORTCUT_DEFAULTS.get(aid, "")
            self._sc_editors[row][0].setKeySequence(QKeySequence(primary))
            self._sc_editors[row][1].setKeySequence(QKeySequence(alt))
        self._sc_check_conflicts()

    def _sc_make_default(self):
        primary, alt = self._sc_collect_shortcuts()
        try:
            _update_config({"shortcuts": primary, "alt_shortcuts": alt})
            QMessageBox.information(self, "Saved", f"Shortcuts saved to:\n{self.CONFIG_FILE}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save shortcuts:\n{e}")

    def _sc_collect_shortcuts(self) -> tuple:
        primary = {}
        alt = {}
        for row, (aid, action) in enumerate(self._shortcut_actions.items()):
            ks1 = self._sc_editors[row][0].keySequence()
            ks2 = self._sc_editors[row][1].keySequence()
            primary[aid] = ks1.toString() if not ks1.isEmpty() else ""
            alt[aid]     = ks2.toString() if not ks2.isEmpty() else ""
        return primary, alt

    def get_shortcut_overrides(self) -> tuple:
        return self._sc_collect_shortcuts()

    def _build_overlay_tab(self, bg_color, show_axes, show_principal_axes, axes_position, principal_axes_position,
                           show_dipole_arrow=False):
        page = QWidget()
        form = QFormLayout(page)
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._bg_btn = ColorButton(bg_color)
        self._bg_btn.colorChanged.connect(self._on_change)
        form.addRow("Background:", self._bg_btn)

        self._show_axes_cb = QCheckBox("Show XYZ Axes")
        self._show_axes_cb.setChecked(show_axes)
        self._show_axes_cb.toggled.connect(self._on_change)
        form.addRow("XYZ axes:", self._show_axes_cb)

        self._axes_pos_combo = QComboBox()
        self._axes_pos_combo.addItems(["Bottom left", "Bottom right", "Top left", "Top right"])
        self._axes_pos_combo.setCurrentText(axes_position.replace("-", " ").title())
        self._axes_pos_combo.setFixedWidth(120)
        self._axes_pos_combo.currentIndexChanged.connect(self._on_change)
        form.addRow("Position:", self._axes_pos_combo)

        self._show_principal_axes_cb = QCheckBox("Show a/b/c Axes")
        self._show_principal_axes_cb.setChecked(show_principal_axes)
        self._show_principal_axes_cb.toggled.connect(self._on_change)
        form.addRow("a/b/c axes:", self._show_principal_axes_cb)

        self._principal_axes_pos_combo = QComboBox()
        self._principal_axes_pos_combo.addItems(["Bottom right", "Bottom left", "Top left", "Top right"])
        self._principal_axes_pos_combo.setCurrentText(principal_axes_position.replace("-", " ").title())
        self._principal_axes_pos_combo.setFixedWidth(120)
        self._principal_axes_pos_combo.currentIndexChanged.connect(self._on_change)
        form.addRow("Position:", self._principal_axes_pos_combo)

        self._show_dipole_arrow_cb = QCheckBox("Show dipole moment arrow")
        self._show_dipole_arrow_cb.setChecked(show_dipole_arrow)
        self._show_dipole_arrow_cb.toggled.connect(self._on_change)
        form.addRow("Dipole arrow:", self._show_dipole_arrow_cb)

        self.tabs.addTab(page, "&Overlay")

    def _build_atoms_bonds_tab(self, atom_scale, bond_width, bond_style,
                                atom_border_mode, atom_border_scale, atom_border_width,
                                bond_color, lighting_intensity, light_position, roughness,
                                fast_render=False):
        page = QWidget()
        form = QFormLayout(page)
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._edit_colors_btn = QPushButton("Edit Atom Colours…")
        self._edit_colors_btn.clicked.connect(self._edit_atom_colors)
        form.addRow("Atom color:", self._edit_colors_btn)

        self._ball_slider = QSlider(Qt.Orientation.Horizontal)
        self._ball_slider.setRange(20, 150)
        self._ball_slider.setValue(int(atom_scale * 100))
        self._ball_slider.setFixedWidth(160)
        self._ball_lbl = QLabel(f"{atom_scale:.2f}")
        self._ball_slider.valueChanged.connect(self._on_change)
        ball_row = QHBoxLayout()
        ball_row.addWidget(self._ball_slider)
        ball_row.addWidget(self._ball_lbl)
        form.addRow("Ball size:", ball_row)

        self._bondw_slider = QSlider(Qt.Orientation.Horizontal)
        self._bondw_slider.setRange(2, 30)
        self._bondw_slider.setValue(int(bond_width))
        self._bondw_slider.setFixedWidth(160)
        self._bondw_lbl = QLabel(f"{bond_width:.0f}")
        self._bondw_slider.valueChanged.connect(self._on_change)
        bondw_row = QHBoxLayout()
        bondw_row.addWidget(self._bondw_slider)
        bondw_row.addWidget(self._bondw_lbl)
        form.addRow("Bond size:", bondw_row)

        self._style_combo = QComboBox()
        self._style_combo.addItems(["Splitted", "Unicolor", "Gradient"])
        self._style_combo.setMinimumWidth(90)
        self._bond_color = bond_color
        style_row = QHBoxLayout()
        style_row.addWidget(self._style_combo)
        self._unicolor_btn = QPushButton()
        self._unicolor_btn.setFixedSize(28, 22)
        self._unicolor_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._unicolor_btn.clicked.connect(self._pick_unicolor)
        self._update_unicolor_btn()
        style_row.addWidget(self._unicolor_btn)
        self._style_combo.currentTextChanged.connect(self._on_style_change)
        current = bond_style.capitalize() if bond_style != "grey" else "Unicolor"
        self._style_combo.setCurrentText(current)
        self._unicolor_btn.setVisible(current == "Unicolor")
        style_row.addStretch()
        form.addRow("Bond Style:", style_row)

        self._roughness_slider = QSlider(Qt.Orientation.Horizontal)
        self._roughness_slider.setRange(0, 200)
        self._roughness_slider.setValue(int(roughness * 100))
        self._roughness_slider.setFixedWidth(160)
        self._roughness_lbl = QLabel(f"{roughness:.2f}")
        self._roughness_slider.valueChanged.connect(self._on_change)
        roughness_row = QHBoxLayout()
        roughness_row.addWidget(self._roughness_slider)
        roughness_row.addWidget(self._roughness_lbl)
        form.addRow("Roughness:", roughness_row)

        self._lighting_slider = QSlider(Qt.Orientation.Horizontal)
        self._lighting_slider.setRange(0, 200)
        self._lighting_slider.setValue(int(lighting_intensity * 100))
        self._lighting_slider.setFixedWidth(160)
        self._lighting_lbl = QLabel(f"{lighting_intensity:.2f}")
        self._lighting_slider.valueChanged.connect(self._on_change)
        lighting_row = QHBoxLayout()
        lighting_row.addWidget(self._lighting_slider)
        lighting_row.addWidget(self._lighting_lbl)
        form.addRow("Lighting:", lighting_row)

        self._light_position = light_position
        self._light_pos_btn = QPushButton(POSITION_LABELS.get(light_position, light_position))
        self._light_pos_btn.setFixedWidth(160)
        self._light_pos_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._light_pos_btn.clicked.connect(self._pick_light_position)
        form.addRow("Lighting Position:", self._light_pos_btn)

        self._border_mode = QComboBox()
        self._border_mode.addItems(["None", "Scaled", "Constant"])
        self._border_mode.setMinimumWidth(90)
        self._border_mode.setCurrentText(atom_border_mode.capitalize())
        self._border_mode.currentTextChanged.connect(self._on_mode_change)

        self._border_slider = QSlider(Qt.Orientation.Horizontal)
        self._border_slider.setFixedWidth(160)
        self._border_slider.valueChanged.connect(self._on_change)

        border_row = QHBoxLayout()
        border_row.addWidget(self._border_mode)
        border_row.addWidget(self._border_slider)
        form.addRow("Border:", border_row)

        self._set_border_slider_for_mode(atom_border_mode, atom_border_scale, atom_border_width)
        self._update_border_visibility()

        self.tabs.addTab(page, "Atoms && Bonds")

        self._ball_slider.installEventFilter(self)
        self._bondw_slider.installEventFilter(self)
        self._roughness_slider.installEventFilter(self)
        self._lighting_slider.installEventFilter(self)
        self._border_slider.installEventFilter(self)

    def _set_border_slider_for_mode(self, mode, scale, width):
        if mode == "constant":
            self._border_slider.setRange(1, 20)
            self._border_slider.setValue(int(width))
        else:
            self._border_slider.setRange(100, 150)
            self._border_slider.setValue(int(scale * 100))

    def _update_border_visibility(self):
        visible = self._border_mode.currentText().lower() != "none"
        if hasattr(self, '_border_slider'):
            self._border_slider.setVisible(visible)

    def _on_mode_change(self):
        mode = self._border_mode.currentText().lower()
        if mode == "constant":
            self._border_slider.setRange(1, 20)
            self._border_slider.setValue(2)
        else:
            self._border_slider.setRange(100, 150)
            self._border_slider.setValue(104)
        self._on_change()

    def _on_style_change(self, text: str):
        self._unicolor_btn.setVisible(text == "Unicolor")
        self._on_change()

    def _on_change(self):
        if hasattr(self, '_ball_lbl'):
            self._ball_lbl.setText(f"{self.ball_scale:.2f}")
            self._bondw_lbl.setText(f"{self.bond_width:.0f}")
            self._lighting_lbl.setText(f"{self.lighting_intensity:.2f}")
            self._roughness_lbl.setText(f"{self.roughness:.2f}")
            self._update_border_visibility()
        if self._live_callback:
            self._live_callback(
                self.theme, self.bg_color,
                self.ball_scale, self.bond_width, self.bond_style,
                self._color_overrides, self.border_mode, self.border_scale,
                self.border_width, self.bond_color,
                self.lighting_intensity, self.light_position,
                self.roughness,
                self.show_axes, self.show_principal_axes,
                self.axes_position, self.principal_axes_position,
            )
        if self._chem_callback and hasattr(self, '_chem_layout_combo'):
            self._chem_callback(self.chem_label_colors, self.chem_layout)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonDblClick:
            cfg = self.load_config() or {}
            acfg = cfg.get("atoms_bonds", {})
            if obj is self._ball_slider:
                self._ball_slider.setValue(int(acfg.get("atom_scale", 0.7) * 100))
            elif obj is self._bondw_slider:
                self._bondw_slider.setValue(int(acfg.get("bond_width_px", 10)))
            elif obj is self._roughness_slider:
                self._roughness_slider.setValue(int(acfg.get("roughness", 1.0) * 100))
            elif obj is self._lighting_slider:
                self._lighting_slider.setValue(int(acfg.get("lighting_intensity", 1.0) * 100))
            elif obj is self._border_slider:
                mode = self._border_mode.currentText().lower()
                if mode == "constant":
                    self._border_slider.setValue(int(acfg.get("atom_border_width", 2)))
                else:
                    self._border_slider.setValue(int(acfg.get("atom_border_scale", 1.04) * 100))
            return True
        return super().eventFilter(obj, event)

    def _pick_unicolor(self):
        col = QColorDialog.getColor(QColor(self._bond_color), self, "Choose Bond Color")
        if col.isValid():
            self._bond_color = col.name()
            self._update_unicolor_btn()
            self._on_change()

    def _update_unicolor_btn(self):
        self._unicolor_btn.setStyleSheet(
            f"background-color: {self._bond_color}; border: 1px solid #888; border-radius: 3px;"
        )

    def _pick_light_position(self):
        dlg = LightPositionDialog(self._light_position, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._light_position = dlg.position
            self._light_pos_btn.setText(POSITION_LABELS.get(self._light_position, self._light_position))
            self._on_change()

    def _edit_atom_colors(self):
        dlg = AtomColorDialog(self._color_overrides, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._color_overrides = dlg.get_overrides()
            self._on_change()

    @property
    def theme(self) -> str:
        return self._theme_combo.currentText().lower()

    @property
    def bg_color(self) -> str:
        return self._bg_btn.color()

    @property
    def restore_molecule(self) -> bool:
        return self._restore_cb.isChecked()

    @property
    def quick_svg_export_widget(self) -> bool:
        return self._quick_svg_widget_cb.isChecked()

    @property
    def start_view(self) -> str:
        return self._start_view_combo.currentText()

    @property
    def ball_scale(self) -> float:
        return self._ball_slider.value() / 100.0

    @property
    def bond_width(self) -> float:
        return float(self._bondw_slider.value())

    @property
    def bond_style(self) -> str:
        text = self._style_combo.currentText()
        return "unicolor" if text == "Unicolor" else text.lower()

    @property
    def bond_color(self) -> str:
        return self._bond_color

    @property
    def border_mode(self) -> str:
        return self._border_mode.currentText().lower()

    @property
    def border_scale(self) -> float:
        return self._border_slider.value() / 100.0

    @property
    def border_width(self) -> float:
        return float(self._border_slider.value())

    @property
    def lighting_intensity(self) -> float:
        return self._lighting_slider.value() / 100.0

    @property
    def light_position(self) -> str:
        return self._light_position

    @property
    def roughness(self) -> float:
        return self._roughness_slider.value() / 100.0

    @property
    def fast_render(self) -> bool:
        return self._fast_render_cb.isChecked()

    @property
    def show_debug_menu(self) -> bool:
        return self._debug_menu_cb.isChecked()

    @property
    def show_axes(self) -> bool:
        return self._show_axes_cb.isChecked()

    @property
    def show_principal_axes(self) -> bool:
        return self._show_principal_axes_cb.isChecked()

    @property
    def axes_position(self) -> str:
        return self._axes_pos_combo.currentText().lower().replace(" ", "-")

    @property
    def principal_axes_position(self) -> str:
        return self._principal_axes_pos_combo.currentText().lower().replace(" ", "-")

    @property
    def show_dipole_arrow(self) -> bool:
        return self._show_dipole_arrow_cb.isChecked()

    def _section_key(self) -> str:
        if self.tabs.currentIndex() == 4:
            return "chem2d"
        return {0: "general", 1: "overlay", 2: "atoms_bonds"}.get(
            self.tabs.currentIndex(), "general")

    def _section_data(self, section: str) -> dict:
        if section == "general":
            return {"theme": self.theme, "restore_molecule": self.restore_molecule,
                    "start_view": self.start_view,
                    "fast_render": self.fast_render, "show_debug_menu": self.show_debug_menu,
                    "quick_svg_export_widget": self.quick_svg_export_widget}
        if section == "chem2d":
            return {"label_colors": self.chem_label_colors, "layout": self.chem_layout}
        if section == "overlay":
            return {
                "bg_color": self.bg_color,
                "show_axes": self.show_axes,
                "show_principal_axes": self.show_principal_axes,
                "axes_position": self.axes_position,
                "principal_axes_position": self.principal_axes_position,
                "show_dipole_arrow": self.show_dipole_arrow,
            }
        return {
            "atom_scale": self.ball_scale,
            "bond_width_px": self.bond_width,
            "bond_style": self.bond_style,
            "bond_color": self.bond_color,
            "atom_border_mode": self.border_mode,
            "atom_border_scale": self.border_scale,
            "atom_border_width": self.border_width,
            "lighting_intensity": self.lighting_intensity,
            "light_position": self.light_position,
            "roughness": self.roughness,
            "color_overrides": self._color_overrides,
        }

    def _save_config(self):
        section = self._section_key()
        try:
            _update_config({section: self._section_data(section)})
            QMessageBox.information(self, "Config Saved",
                f"{section.title()} settings saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save config:\n{e}")

    def _restore_section(self, section: str):
        if section == "general":
            self._theme_combo.setCurrentText("Light")
            self._restore_cb.setChecked(False)
            self._start_view_combo.setCurrentText("3D")
            self._fast_render_cb.setChecked(False)
            self._quick_svg_widget_cb.setChecked(False)
        elif section == "chem2d":
            self._chem_label_combo.setCurrentIndex(0)
            self._chem_layout_combo.setCurrentIndex(0)
        elif section == "overlay":
            self._bg_btn.set_color("#ffffff")
            self._show_axes_cb.setChecked(False)
            self._show_principal_axes_cb.setChecked(False)
            self._axes_pos_combo.setCurrentText("Bottom left")
            self._principal_axes_pos_combo.setCurrentText("Bottom right")
        else:
            self._ball_slider.setValue(70)
            self._bondw_slider.setValue(10)
            self._style_combo.setCurrentText("Splitted")
            self._bond_color = "#444444"
            self._update_unicolor_btn()
            self._roughness_slider.setValue(100)
            self._lighting_slider.setValue(100)
            self._light_position = "top-left"
            self._light_pos_btn.setText("Top left")
            self._border_mode.setCurrentText("Scaled")
            self._border_slider.setValue(104)
            self._color_overrides = {}

    def _restore_defaults(self):
        section = self._section_key()
        self._restore_section(section)
        self._on_change()

    @staticmethod
    def load_config():
        path = SettingsDialog.CONFIG_FILE
        data = paths.read_json(path)
        if not isinstance(data, dict):
            return None

        TAB_KEYS = {"general", "overlay", "atoms_bonds", "chem2d"}
        if not data.keys() & TAB_KEYS:
            data = {
                "general": {
                    "theme": data.get("theme", "light"),
                    "restore_molecule": data.get("restore_molecule", False),
                    "start_view": data.get("start_view", "3D"),
                    "fast_render": data.get("fast_render", False),
                    "show_debug_menu": data.get("show_debug_menu", False),
                    "quick_svg_export_widget": data.get("quick_svg_export_widget", False),
                },
                "overlay": {
                    "bg_color": data.get("bg_color", "#ffffff"),
                    "show_axes": data.get("show_axes", False),
                    "show_principal_axes": data.get("show_principal_axes", False),
                    "axes_position": data.get("axes_position", "bottom-left"),
                    "principal_axes_position": data.get("principal_axes_position", "bottom-right"),
                },
                "atoms_bonds": {
                    "atom_scale": data.get("atom_scale", 0.7),
                    "bond_width_px": data.get("bond_width_px", 10.0),
                    "bond_style": data.get("bond_style", "splitted"),
                    "bond_color": data.get("bond_color", "#444444"),
                    "atom_border_mode": data.get("atom_border_mode", "scaled"),
                    "atom_border_scale": data.get("atom_border_scale", 1.04),
                    "atom_border_width": data.get("atom_border_width", 2.0),
                    "lighting_intensity": data.get("lighting_intensity", 1.0),
                    "light_position": data.get("light_position", "top-left"),
                    "roughness": data.get("roughness", 1.0),
                    "color_overrides": data.get("color_overrides", {}),
                },
            }
            try:
                paths.write_json(path, data, indent=2)
            except OSError:
                pass
        return data


# ─────────────────────────────────────────────────────────────────────────────
# ATOM COLOUR EDITOR DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class AtomColorDialog(QDialog):
    """Periodic-table colour editor with per-element override support."""

    CONFIG_FILE = paths.config_path("color_table.json")

    def __init__(self, current_overrides: dict, live_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Atom Colours")
        self.setMinimumSize(1020, 700)
        self._live_callback = live_callback
        self._overrides = dict(current_overrides)

        # Load any saved defaults on top of current overrides
        saved = self._load_defaults()
        for k, v in saved.items():
            if k not in self._overrides:
                self._overrides[k] = v

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        lbl = QLabel("Click a tile to change its CPK colour. "
                      "\"Set Default\" saves your palette to ~/.config/molvector/color_table.json.")
        lbl.setObjectName("dim")
        lbl.setStyleSheet("font-size:11px;")
        layout.addWidget(lbl)

        grid = QGridLayout()
        grid.setSpacing(3)

        btn_size = 52
        dot_size = 13
        dot_margin = 4
        btn_font = QFont("", 10, QFont.Weight.Bold)

        self._all_buttons: dict[str, QPushButton] = {}
        self._dots:        dict[str, QLabel]      = {}

        for sym, Z, r, c in _PERIODIC_TABLE_LAYOUT:
            btn = QPushButton(sym)
            btn.setFixedSize(btn_size, btn_size)
            btn.setFont(btn_font)

            cat = _element_category(sym, Z)
            cat_col = _ELEM_CATEGORY_COLORS.get(cat, "#757575")
            is_dark = sum(int(cat_col[i:i+2], 16) for i in (1, 3, 5)) < 400
            text_color = "#ffffff" if is_dark else "#000000"
            self._apply_tile_style(btn, cat_col, text_color)

            cpk = self._overrides.get(sym, CPK_BASE.get(sym, DEFAULT_BASE))
            dot = QLabel()
            dot.setFixedSize(dot_size, dot_size)
            dot.setStyleSheet(
                f"background:{cpk}; border:1px solid rgba(0,0,0,0.35); "
                f"border-radius:{dot_size // 2}px;"
            )
            dot.setParent(btn)
            dot.move(btn_size - dot_size - dot_margin, btn_size - dot_size - dot_margin)

            tooltip = f"{sym} ({Z}) — {ELEM_FULL_NAME.get(sym, 'Unknown')}\nCPK: {cpk}"
            btn.setToolTip(tooltip)

            def _make_handler(s):
                return lambda checked, sym=s: self._pick_color(sym)

            btn.clicked.connect(_make_handler(sym))
            grid.addWidget(btn, r, c)
            self._all_buttons[sym] = btn
            self._dots[sym]        = dot

        layout.addLayout(grid)

        # Buttons
        btn_row = QHBoxLayout()
        set_default_btn = QPushButton("Set Default")
        set_default_btn.setToolTip("Save current overrides as default palette")
        set_default_btn.clicked.connect(self._save_default)
        btn_row.addWidget(set_default_btn)

        load_default_btn = QPushButton("Load Default")
        load_default_btn.setToolTip("Load the saved default palette (color_table.json)")
        load_default_btn.clicked.connect(self._load_and_apply)
        btn_row.addWidget(load_default_btn)

        reset_cpks_btn = QPushButton("Reset All to CPK")
        reset_cpks_btn.setToolTip("Reset every element to its CPK default colour")
        reset_cpks_btn.clicked.connect(self._reset_all)
        btn_row.addWidget(reset_cpks_btn)

        btn_row.addStretch()

        ok_cancel = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_cancel.accepted.connect(self.accept)
        ok_cancel.rejected.connect(self.reject)
        btn_row.addWidget(ok_cancel)
        layout.addLayout(btn_row)

    # ── internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _apply_tile_style(btn: QPushButton, bg: str, fg: str):
        btn.setStyleSheet(
            f"QPushButton {{ background:{bg}; color:{fg}; "
            f"border:1px solid rgba(0,0,0,0.2); border-radius:4px; }}"
            f"QPushButton:hover {{ border:2px solid white; }}"
        )

    def _update_dot(self, sym: str):
        cpk = self._overrides.get(sym, CPK_BASE.get(sym, DEFAULT_BASE))
        dot = self._dots[sym]
        dot.setStyleSheet(
            f"background:{cpk}; border:1px solid rgba(0,0,0,0.35); "
            f"border-radius:6px;"
        )
        btn = self._all_buttons[sym]
        Z = next((z for s, z, *_ in _PERIODIC_TABLE_LAYOUT if s == sym), 0)
        btn.setToolTip(f"{sym} ({Z}) — {ELEM_FULL_NAME.get(sym, 'Unknown')}\nCPK: {cpk}")

    def _pick_color(self, sym: str):
        current = self._overrides.get(sym, CPK_BASE.get(sym, DEFAULT_BASE))
        color = QColorDialog.getColor(QColor(current), self, f"Pick colour for {sym}")
        if color.isValid():
            self._overrides[sym] = color.name()
            self._update_dot(sym)
            self._trigger_live()

    def _trigger_live(self, *_):
        if self._live_callback:
            self._live_callback(self._overrides)

    def _reset_all(self):
        self._overrides.clear()
        for sym in self._dots:
            self._update_dot(sym)
        self._trigger_live()

    def _save_default(self):
        try:
            paths.write_json(self.CONFIG_FILE, self._overrides, indent=2)
            QMessageBox.information(self, "Saved", f"Default palette saved to:\n{self.CONFIG_FILE}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save palette:\n{e}")

    @staticmethod
    def _load_defaults() -> dict:
        data = paths.read_json(AtomColorDialog.CONFIG_FILE, {})
        return data if isinstance(data, dict) else {}

    def _load_and_apply(self):
        saved = self._load_defaults()
        if not saved:
            QMessageBox.information(self, "No defaults", "No colour_table.json found.")
            return
        self._overrides = saved
        for sym in self._dots:
            self._update_dot(sym)
        self._trigger_live()

    def get_overrides(self) -> dict:
        return dict(self._overrides)


# ─────────────────────────────────────────────────────────────────────────────
# SPECTRUM DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumDialog(QDialog):
    def __init__(self, x_data, y_data, x_label, y_label, title, metadata="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(700, 500)
        self._x = np.array(x_data)
        self._y = np.array(y_data)
        self._meta = metadata
        self._xl, self._yl = x_label, y_label
        self._default_file = "spectrum.txt"

        layout = QVBoxLayout(self)
        
        if not HAS_MPL:
            layout.addWidget(QLabel("Matplotlib not found. Cannot display plot."))
        else:
            # Detect current theme from parent
            theme_name = "dark"
            p = parent
            while p:
                if hasattr(p, "_current_theme"):
                    theme_name = p._current_theme
                    break
                p = p.parent()
            
            t = THEMES[theme_name]
            
            self.fig = Figure(figsize=(6, 4), facecolor=t['CARD_BG'])
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor(t['DARK_BG'])
            
            # Plot stems/peaks
            self.ax.vlines(self._x, 0, self._y, color=t['ACCENT'], linewidth=2)
            self.ax.scatter(self._x, self._y, color=t['ACCENT'], s=20)
            
            self.ax.set_xlabel(x_label, color=t['FG'])
            self.ax.set_ylabel(y_label, color=t['FG'])
            self.ax.tick_params(colors=t['FG_DIM'])
            for spine in self.ax.spines.values():
                spine.set_edgecolor(t['BORDER'])
            
            self.fig.tight_layout()
            layout.addWidget(self.canvas)

            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            layout.addWidget(self.toolbar)

        btns = QHBoxLayout()
        btn_export = QPushButton("Export Data (.txt)")
        btn_export.clicked.connect(self._export)
        btns.addStretch()
        btns.addWidget(btn_export)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

    def set_default_filename(self, name: str):
        self._default_file = name

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Spectrum", self._default_file, "Text files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                if self._meta:
                    for line in self._meta.splitlines():
                        f.write(f"# {line}\n")
                f.write(f"# {self._xl}\t{self._yl}\n")
                for xi, yi in zip(self._x, self._y):
                    f.write(f"{xi:.6f}\t{yi:.6f}\n")
            QMessageBox.information(self, "Export Successful", f"Data saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not export: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MULLIKEN GRADIENT BAR (non-interactive preview)
# ─────────────────────────────────────────────────────────────────────────────

class _MullikenGradientBar(QWidget):
    """A preview of the negative/zero/positive gradient with the three
    colour-pick swatches positioned directly under their actual stop on the
    bar (left end, right end, and wherever the centre currently falls) so
    the picker for a stop's colour lines up with where it sits visually."""
    colorPickRequested = pyqtSignal(str)  # "neg" | "zero" | "pos"

    TICK_H = 6
    BAR_H = 14
    GAP = 4
    SWATCH = 18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.TICK_H + self.BAR_H + self.GAP + self.SWATCH)
        self.setMinimumWidth(140)
        self._neg = MULLIKEN_NEGATIVE_COLOR
        self._zero = MULLIKEN_ZERO_COLOR
        self._pos = MULLIKEN_POSITIVE_COLOR
        self._zero_frac = 0.5

        self._neg_btn = self._make_swatch("neg", "Colour for the most negative charge")
        self._zero_btn = self._make_swatch("zero", "Colour at the centre value")
        self._pos_btn = self._make_swatch("pos", "Colour for the most positive charge")
        self._restyle_swatches()

    def _make_swatch(self, key: str, tooltip: str) -> QPushButton:
        btn = QPushButton(self)
        btn.setFixedSize(self.SWATCH, self.SWATCH)
        btn.setAutoDefault(False)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tooltip)
        btn.clicked.connect(lambda: self.colorPickRequested.emit(key))
        return btn

    def set_colors(self, neg: str, zero: str, pos: str, zero_frac: float):
        self._neg, self._zero, self._pos = neg, zero, pos
        self._zero_frac = max(0.0, min(1.0, zero_frac))
        self._restyle_swatches()
        self._position_swatches()
        self.update()

    def _restyle_swatches(self):
        for btn, color in (
            (self._neg_btn, self._neg), (self._zero_btn, self._zero), (self._pos_btn, self._pos),
        ):
            btn.setStyleSheet(f"background:{color}; border-radius:3px; border:1.5px solid #eeeeee;")

    def _bar_rect(self) -> QRect:
        # Inset by half a swatch on each side so the neg/pos swatches can be
        # centred exactly on the bar's own left/right edge and still fit
        # fully inside the widget — no clamping needed, so no misalignment.
        half = self.SWATCH // 2
        return QRect(half, self.TICK_H, max(1, self.width() - 2 * half), self.BAR_H)

    SPACING = 2  # minimum gap kept between adjacent swatches

    def _position_swatches(self):
        bar = self._bar_rect()
        y = bar.bottom() + self.GAP
        half = self.SWATCH // 2
        max_x = self.width() - self.SWATCH

        def clamp_x(x):
            return max(0, min(max_x, x))

        neg_x = bar.left() - half
        pos_x = bar.right() - half
        zero_x = clamp_x(int(bar.left() + self._zero_frac * bar.width()) - half)

        # The middle swatch tracks a data-driven position and can land right
        # next to (or past) an end swatch when the centre sits near range_min
        # or range_max — push it back inside the gap between the two ends
        # so the three never overlap.
        lo = neg_x + self.SWATCH + self.SPACING
        hi = pos_x - self.SWATCH - self.SPACING
        if lo <= hi:
            zero_x = max(lo, min(hi, zero_x))

        self._neg_btn.move(neg_x, y)
        self._pos_btn.move(pos_x, y)
        self._zero_btn.move(zero_x, y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_swatches()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar = self._bar_rect()

        grad = QLinearGradient(bar.left(), 0, bar.right(), 0)
        grad.setColorAt(0.0, QColor(self._neg))
        grad.setColorAt(self._zero_frac, QColor(self._zero))
        grad.setColorAt(1.0, QColor(self._pos))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#666666"), 1))
        painter.drawRoundedRect(bar, 3, 3)

        for frac in (0.0, self._zero_frac, 1.0):
            x = bar.left() + frac * bar.width()
            x = min(max(x, bar.left() + 1), bar.right() - 1)
            width = 2.5 if 0.0 < frac < 1.0 else 1.5
            painter.setPen(QPen(QColor("#ffffff"), width))
            painter.drawLine(int(x), bar.top() - self.TICK_H, int(x), bar.top())
        painter.end()


# ─────────────────────────────────────────────────────────────────────────────
# CALCULATIONS DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class CalculationsDialog(QDialog):
    modeSelected = pyqtSignal(object)  # VibrationalMode
    stateSelected = pyqtSignal(object) # ExcitedState
    viewSpectrum = pyqtSignal(str)     # "ir" or "uvvis"
    animationToggled = pyqtSignal(bool)
    vectorsToggled = pyqtSignal(bool)
    colorByCharge = pyqtSignal(float, float, float, str, str, str, bool, bool)  # (range_min, range_max, center, neg_color, zero_color, pos_color, show_legend, include_in_export)
    clearAtomColors = pyqtSignal()

    def __init__(self, mol: Molecule, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Calculations — {mol.name}")
        self.setMinimumSize(650, 500)
        self.mol = mol

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ── General Tab (first position) ──
        gen_page = QWidget()
        gpl = QVBoxLayout(gen_page)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def sel_label(text):
            lbl = QLabel(text)
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            return lbl

        if mol.g16_route is not None:
            form.addRow("Route:", sel_label(mol.g16_route))
        if mol.g16_point_group is not None:
            form.addRow("Point Group:", sel_label(mol.g16_point_group))
        if mol.g16_opt_energy is not None:
            form.addRow("Optimization Energy:", sel_label(f"{mol.g16_opt_energy:.10f} Ha"))
        if mol.g16_scf_energy is not None:
            form.addRow("SCF Energy:", sel_label(f"{mol.g16_scf_energy:.10f} Ha"))
        if mol.g16_rotconst is not None:
            a, b, c = mol.g16_rotconst
            rc = sel_label(f"A = {a:.4f} MHz\nB = {b:.4f} MHz\nC = {c:.4f} MHz")
            form.addRow("Rotational Constants:", rc)
        if mol.g16_dipole is not None:
            x, y, z, t = mol.g16_dipole
            dm = sel_label(f"X = {x:.4f} D\nY = {y:.4f} D\nZ = {z:.4f} D\nTotal = {t:.4f} D")
            form.addRow("Dipole Moment:", dm)
        if mol.g16_nproc is not None:
            form.addRow("Processors:", sel_label(str(mol.g16_nproc)))
        if mol.g16_mem is not None:
            form.addRow("Memory:", sel_label(mol.g16_mem))
        if mol.g16_chk is not None:
            form.addRow("Checkpoint:", sel_label(mol.g16_chk))
        if mol.g16_zpve is not None:
            form.addRow("ZPVE:", sel_label(f"{mol.g16_zpve:.10f} Ha"))
        if mol.g16_enthalpy is not None:
            form.addRow("Enthalpy:", sel_label(f"{mol.g16_enthalpy:.10f} Ha"))
        if mol.g16_freeenergy is not None:
            form.addRow("Gibbs Free Energy:", sel_label(f"{mol.g16_freeenergy:.10f} Ha"))
        if mol.g16_entropy is not None:
            form.addRow("Entropy:", sel_label(f"{mol.g16_entropy:.10f} Ha/K"))
        if mol.g16_temperature is not None:
            form.addRow("Temperature:", sel_label(f"{mol.g16_temperature:.1f} K"))
        if mol.vibrational_modes:
            n = len(mol.vibrational_modes)
            form.addRow("Vibrational Modes:", sel_label(str(n)))
        if mol.excited_states:
            n = len(mol.excited_states)
            form.addRow("Excited States:", sel_label(str(n)))
        if mol.mulliken_charges:
            n = len(mol.mulliken_charges)
            form.addRow("Mulliken Charges:", sel_label(f"{n} atom(s)"))

        if form.rowCount() > 0:
            gpl.addLayout(form)
            gpl.addStretch()
            self.tabs.addTab(gen_page, "General")

        # ── Frequencies Tab ──
        if mol.vibrational_modes:
            freq_page = QWidget()
            fpl = QVBoxLayout(freq_page)
            
            self._freq_table = QTableWidget(len(mol.vibrational_modes), 3)
            self._freq_table.setHorizontalHeaderLabels(["Mode", "Freq (cm⁻¹)", "Intensity (km/mol)"])
            self._freq_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self._freq_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self._freq_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self._freq_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            
            for i, m in enumerate(mol.vibrational_modes):
                self._freq_table.setItem(i, 0, QTableWidgetItem(str(m.index)))
                self._freq_table.setItem(i, 1, QTableWidgetItem(f"{m.frequency:.2f}"))
                self._freq_table.setItem(i, 2, QTableWidgetItem(f"{m.intensity:.2f}"))
            
            self._freq_table.itemSelectionChanged.connect(lambda t=self._freq_table: self._on_freq_sel(t))
            fpl.addWidget(self._freq_table)
            
            btn_ir = QPushButton("View IR Spectrum…")
            btn_ir.setAutoDefault(False)
            btn_ir.clicked.connect(lambda: self.viewSpectrum.emit("ir"))
            fpl.addWidget(btn_ir)
            
            # Animation control
            self._anim_check = QCheckBox("Show animation")
            self._anim_check.setObjectName("dim")
            self._anim_check.toggled.connect(self.animationToggled.emit)
            fpl.addWidget(self._anim_check)

            # Vector control
            self._vector_check = QCheckBox("Show displacement vectors")
            self._vector_check.setChecked(False)
            self._vector_check.setObjectName("dim")
            self._vector_check.toggled.connect(self.vectorsToggled.emit)
            fpl.addWidget(self._vector_check)

            btn_export_freq = QPushButton("Export Data (.txt)")
            btn_export_freq.setAutoDefault(False)
            btn_export_freq.clicked.connect(lambda: self._export_table(self._freq_table, "frequencies"))
            fpl.addWidget(btn_export_freq)
            
            self.tabs.addTab(freq_page, "Frequencies")

        # ── TDDFT Tab ──
        if mol.excited_states:
            td_page = QWidget()
            tpl = QVBoxLayout(td_page)
            
            self._state_table = QTableWidget(len(mol.excited_states), 4)
            self._state_table.setHorizontalHeaderLabels(["State", "Energy (eV)", "Wavelength (nm)", "f"])
            self._state_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self._state_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self._state_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self._state_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            
            for i, s in enumerate(mol.excited_states):
                self._state_table.setItem(i, 0, QTableWidgetItem(f"S{s.index} ({s.symmetry})"))
                self._state_table.setItem(i, 1, QTableWidgetItem(f"{s.energy_ev:.4f}"))
                self._state_table.setItem(i, 2, QTableWidgetItem(f"{s.wavelength_nm:.2f}"))
                self._state_table.setItem(i, 3, QTableWidgetItem(f"{s.oscillator_strength:.4f}"))
            
            self._state_table.itemSelectionChanged.connect(lambda t=self._state_table: self._on_state_sel(t))
            tpl.addWidget(self._state_table)
            
            btn_uv = QPushButton("View UV-Vis Spectrum…")
            btn_uv.setAutoDefault(False)
            btn_uv.clicked.connect(lambda: self.viewSpectrum.emit("uvvis"))
            tpl.addWidget(btn_uv)

            btn_export_state = QPushButton("Export Data (.txt)")
            btn_export_state.setAutoDefault(False)
            btn_export_state.clicked.connect(lambda: self._export_table(self._state_table, "excited_states"))
            tpl.addWidget(btn_export_state)
            
            self.tabs.addTab(td_page, "Excited States (TDDFT)")

        # ── Mulliken Charges Tab ──
        if mol.mulliken_charges:
            charges = mol.mulliken_charges
            mull_page = QWidget()
            mpl = QVBoxLayout(mull_page)

            self._mulliken_table = QTableWidget(len(charges), 3)
            self._mulliken_table.setHorizontalHeaderLabels(["Atom #", "Element", "Charge (e)"])
            self._mulliken_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self._mulliken_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self._mulliken_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self._mulliken_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

            for i, q in enumerate(charges):
                elem = mol.atoms[i].element if i < len(mol.atoms) else ""
                self._mulliken_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self._mulliken_table.setItem(i, 1, QTableWidgetItem(elem))
                self._mulliken_table.setItem(i, 2, QTableWidgetItem(f"{q:.6f}"))

            mpl.addWidget(self._mulliken_table)

            btn_export_mull = QPushButton("Export Data (.txt)")
            btn_export_mull.setAutoDefault(False)
            btn_export_mull.clicked.connect(lambda: self._export_table(self._mulliken_table, "mulliken_charges"))
            mpl.addWidget(btn_export_mull)

            color_box = QGroupBox("Color Atoms by Charge")
            cbl = QFormLayout(color_box)

            # Red/green default to the actual min/max partial charges. The
            # neutral ("0") colour sits at the mean charge by default, not
            # literal zero, but it's freely editable to any value by typing.
            init_center = sum(charges) / len(charges)
            data_min = min(charges)
            data_max = max(charges)

            self._mulliken_colors = {
                "neg": MULLIKEN_NEGATIVE_COLOR,
                "zero": MULLIKEN_ZERO_COLOR,
                "pos": MULLIKEN_POSITIVE_COLOR,
            }

            self._mulliken_min = QDoubleSpinBox()
            self._mulliken_min.setDecimals(4)
            self._mulliken_min.setRange(-1000.0, 1000.0)
            self._mulliken_min.setSingleStep(0.01)
            self._mulliken_min.setValue(data_min)
            cbl.addRow("Range Min (red):", self._mulliken_min)

            self._mulliken_max = QDoubleSpinBox()
            self._mulliken_max.setDecimals(4)
            self._mulliken_max.setRange(-1000.0, 1000.0)
            self._mulliken_max.setSingleStep(0.01)
            self._mulliken_max.setValue(data_max)
            cbl.addRow("Range Max (green):", self._mulliken_max)

            self._mulliken_center_spin = QDoubleSpinBox()
            self._mulliken_center_spin.setDecimals(4)
            self._mulliken_center_spin.setRange(-1000.0, 1000.0)
            self._mulliken_center_spin.setSingleStep(0.01)
            self._mulliken_center_spin.setValue(init_center)
            cbl.addRow("Center (black, mean by default):", self._mulliken_center_spin)

            self._mulliken_bar = _MullikenGradientBar()
            self._mulliken_bar.colorPickRequested.connect(self._pick_mulliken_color)
            cbl.addRow("Gradient:", self._mulliken_bar)

            self._mulliken_debounce = QTimer(self)
            self._mulliken_debounce.setSingleShot(True)
            self._mulliken_debounce.timeout.connect(self._emit_color_by_charge)
            self._mulliken_min.valueChanged.connect(self._on_mulliken_range_changed)
            self._mulliken_max.valueChanged.connect(self._on_mulliken_range_changed)
            self._mulliken_center_spin.valueChanged.connect(self._on_mulliken_range_changed)
            self._update_mulliken_bar()

            self._mulliken_legend_check = QCheckBox("Show gradient legend on canvas")
            self._mulliken_legend_check.toggled.connect(lambda _: self._emit_color_by_charge())
            cbl.addRow(self._mulliken_legend_check)

            self._mulliken_export_check = QCheckBox("Include gradient in exports")
            self._mulliken_export_check.setEnabled(False)
            self._mulliken_legend_check.toggled.connect(self._mulliken_export_check.setEnabled)
            self._mulliken_export_check.toggled.connect(lambda _: self._emit_color_by_charge())
            cbl.addRow(self._mulliken_export_check)

            btn_row = QHBoxLayout()
            btn_apply_color = QPushButton("Apply Colors")
            btn_apply_color.setAutoDefault(False)
            btn_apply_color.clicked.connect(self._emit_color_by_charge)
            btn_row.addWidget(btn_apply_color)
            btn_clear_color = QPushButton("Clear Colors")
            btn_clear_color.setAutoDefault(False)
            btn_clear_color.clicked.connect(self._clear_mulliken_colors)
            btn_row.addWidget(btn_clear_color)
            cbl.addRow(btn_row)

            mpl.addWidget(color_box)

            self.tabs.addTab(mull_page, "Mulliken Charges")

        close_btn = QPushButton("Close")
        close_btn.setAutoDefault(False)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _pick_mulliken_color(self, key: str):
        dlg = QColorDialog(QColor(self._mulliken_colors[key]), self)
        dlg.setWindowTitle("Pick Colour")
        if dlg.exec():
            self._mulliken_colors[key] = dlg.selectedColor().name()
            self._update_mulliken_bar()
            self._emit_color_by_charge()

    def _on_mulliken_range_changed(self, _value):
        self._update_mulliken_bar()
        self._mulliken_debounce.start(150)

    def _update_mulliken_bar(self):
        lo, hi = self._mulliken_min.value(), self._mulliken_max.value()
        center = self._mulliken_center_spin.value()
        span = (hi - lo) or 1e-9
        zero_frac = max(0.0, min(1.0, (center - lo) / span))
        self._mulliken_bar.set_colors(
            self._mulliken_colors["neg"], self._mulliken_colors["zero"], self._mulliken_colors["pos"],
            zero_frac,
        )

    def _emit_color_by_charge(self):
        show_legend = self._mulliken_legend_check.isChecked()
        self.colorByCharge.emit(
            self._mulliken_min.value(), self._mulliken_max.value(), self._mulliken_center_spin.value(),
            self._mulliken_colors["neg"], self._mulliken_colors["zero"], self._mulliken_colors["pos"],
            show_legend,
            show_legend and self._mulliken_export_check.isChecked(),
        )

    def _clear_mulliken_colors(self):
        self._mulliken_legend_check.setChecked(False)
        self._mulliken_export_check.setChecked(False)
        self.clearAtomColors.emit()

    def _on_freq_sel(self, table):
        rows = table.selectedItems()
        if not rows: return
        idx = int(rows[0].text()) - 1
        self.modeSelected.emit(self.mol.vibrational_modes[idx])

    def _on_state_sel(self, table):
        rows = table.selectedItems()
        if not rows: return
        # Extract "S1" -> 1
        row_text = rows[0].text()
        idx = int(row_text.split()[0][1:]) - 1
        self.stateSelected.emit(self.mol.excited_states[idx])

    def _export_table(self, table, kind):
        safe = get_safe_filename(self.mol.name)
        default_file = f"{safe}_{kind}.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Export Table", default_file, "Text files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Molecule: {self.mol.name}\n")
                f.write(f"# Calculation: {kind}\n")
                headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
                f.write("# " + "\t".join(headers) + "\n")
                for row in range(table.rowCount()):
                    vals = []
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        vals.append(item.text() if item else "")
                    f.write("\t".join(vals) + "\n")
            QMessageBox.information(self, "Export Successful", f"Data saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not export: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class TerminalDialog(QDialog):
    SCP_SERVERS_FILE = paths.config_path("scp_servers.json")

    def __init__(self, workdir: str = "", current_file: str = "", mol=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Terminal")
        self.setMinimumSize(600, 350)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._workdir = workdir
        self._current_file = current_file
        self._mol = mol
        self._process = None

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        self._toolbar = QHBoxLayout()
        self._macro_btn = QPushButton("SCP…", self)
        self._macro_btn.setToolTip("Copy file to server via SCP")
        self._macro_btn.clicked.connect(self._run_macro)
        self._toolbar.addWidget(self._macro_btn)
        self._toolbar.addStretch()
        layout.addLayout(self._toolbar)

        self._output = QPlainTextEdit(self)
        self._output.setReadOnly(True)
        self._output.setFont(_monospace_font(11))
        self._output.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #d4d4d4; border: none; }"
        )
        self._output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._output, 1)

        input_row = QHBoxLayout()
        self._prompt = QLabel("$ ")
        self._prompt.setFont(_monospace_font(11))
        self._prompt.setStyleSheet("color: #d4d4d4;")
        input_row.addWidget(self._prompt)

        self._input = QLineEdit(self)
        self._input.setFont(_monospace_font(11))
        self._input.setStyleSheet(
            "QLineEdit { background: #1e1e1e; color: #d4d4d4; border: 1px solid #333; padding: 2px 4px; }"
        )
        self._input.returnPressed.connect(self._on_input)
        input_row.addWidget(self._input, 1)

        layout.addLayout(input_row)

        self._macro_btn.setAutoDefault(False)
        self._input.setFocus()

    def showEvent(self, event):
        super().showEvent(event)
        if self._process is None:
            self._start_process()

    _ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def _start_process(self):
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_process_finished)
        self._process.errorOccurred.connect(self._on_process_error)

        is_win = platform.system() == "Windows"

        if is_win:
            # chcp 65001: make cmd.exe print UTF-8, which _on_stdout decodes.
            self._process.start("cmd.exe", ["/Q", "/K", "chcp 65001 >nul"])
            if self._workdir:
                self._process.write(f'cd /d "{self._workdir}"\r\n'.encode("utf-8"))
                self.setWindowTitle(f"Terminal — {self._workdir}")
        else:
            env = QProcessEnvironment.systemEnvironment()
            env.insert("TERM", "xterm-256color")
            env.insert("PS1", "$ ")
            self._process.setProcessEnvironment(env)

            bash = shutil.which("bash") or "/bin/bash"
            # `script` gives bash a pseudo-terminal; its syntax differs between
            # BSD/macOS (command as trailing arguments) and util-linux (-c).
            script_path = shutil.which("script")
            if script_path and sys.platform == "darwin":
                self._process.start(script_path, ["-q", "/dev/null", bash, "-i"])
            elif script_path:
                self._process.start(script_path, ["-q", "-c", f"{shlex.quote(bash)} -i", "/dev/null"])
            else:
                self._process.start(bash, ["-i"])

            if self._workdir:
                self._process.write(f"cd {shlex.quote(self._workdir)}\n".encode("utf-8"))
                self.setWindowTitle(f"Terminal — {self._workdir}")

    def _on_stdout(self):
        data = self._process.readAllStandardOutput()
        text = data.data().decode("utf-8", errors="replace")
        text = self._ANSI_RE.sub("", text)
        self._output.moveCursor(QTextCursor.MoveOperation.End)
        self._output.insertPlainText(text)
        self._output.moveCursor(QTextCursor.MoveOperation.End)

    def _on_input(self):
        cmd = self._input.text()
        self._input.clear()
        if self._process is None or self._process.state() != QProcess.ProcessState.Running:
            self._output.insertPlainText("[Process not running]\n")
            return
        if cmd.strip().lower() in ("exit", "quit"):
            self._process.write(b"exit\n")
            return
        self._process.write((cmd + "\n").encode("utf-8"))

    @staticmethod
    def _load_scp_servers():
        data = paths.read_json(TerminalDialog.SCP_SERVERS_FILE, {})
        servers = data.get("servers", []) if isinstance(data, dict) else []
        cleaned = []
        for s in servers:
            if ":" in s:
                host, path = s.split(":", 1)
                base = os.path.basename(path.rstrip("/"))
                if "." in base:
                    path = os.path.dirname(path)
                    s = f"{host}:{path}"
            if s not in cleaned:
                cleaned.append(s)
        return cleaned

    @staticmethod
    def _save_scp_server(destination: str):
        servers = TerminalDialog._load_scp_servers()
        if destination not in servers:
            servers.append(destination)
        paths.write_json(TerminalDialog.SCP_SERVERS_FILE, {"servers": servers})

    def _run_macro(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("SCP — Copy to Server")
        dlg.setMinimumWidth(750)
        form = QFormLayout(dlg)

        file_row = QHBoxLayout()
        file_edit = QLineEdit(dlg)
        file_edit.setPlaceholderText("/path/to/local/file")
        file_edit.setMinimumWidth(400)
        file_row.addWidget(file_edit, 1)
        form.addRow("File:", file_row)

        use_row = QHBoxLayout()
        use_input_cb = QCheckBox("Use current input", dlg)
        use_input_cb.setEnabled(self._mol is not None)
        use_input_cb.setChecked(True)
        use_row.addWidget(use_input_cb)
        use_row.addSpacing(10)
        input_combo = QComboBox(dlg)
        input_combo.addItem("G16")
        input_combo.setToolTip("Input type to generate")
        input_combo.setMinimumContentsLength(8)
        input_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        use_row.addWidget(input_combo)
        form.addRow("", use_row)

        as_edit = QLineEdit(dlg)
        as_edit.setPlaceholderText("filename on server (optional)")
        as_edit.setMinimumWidth(400)
        form.addRow("As:", as_edit)

        def _on_use_input_toggled(checked: bool):
            if checked and self._mol is not None and not as_edit.text().strip():
                safe = get_safe_filename(self._mol.name, "input")
                as_edit.setText(f"{safe}.com")

        use_input_cb.toggled.connect(_on_use_input_toggled)
        if use_input_cb.isChecked() and self._mol is not None:
            _on_use_input_toggled(True)

        servers = self._load_scp_servers()
        dest_edit = QLineEdit(dlg)
        dest_edit.setPlaceholderText("user@server:/path/on/server")
        dest_edit.setMinimumWidth(400)
        if servers:
            dest_edit.setText(servers[-1])
        form.addRow("Server path:", dest_edit)

        remember_cb = QCheckBox("Remember server", dlg)
        form.addRow("", remember_cb)

        btn_row = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        btn_row.accepted.connect(dlg.accept)
        btn_row.rejected.connect(dlg.reject)
        form.addRow(btn_row)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        src = file_edit.text().strip()
        dst = dest_edit.text().strip()
        if not dst:
            return

        created_tmp = False
        if use_input_cb.isChecked() and self._mol is not None:
            kind = input_combo.currentText()
            if kind == "G16":
                saved = G16InputDialog._saved_state
                if saved.get("preview") and saved.get("geometry") == _geometry_key(self._mol):
                    content = saved["preview"]
                else:
                    content = save_gaussian_input(self._mol)
                suffix = ".com"
            else:
                content = None
            if content is not None:
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n",
                    suffix=suffix, prefix="molvector_input_",
                    delete=False, dir=tempfile.gettempdir(),
                )
                tmp.write(content)
                tmp.close()
                src = tmp.name
                created_tmp = True

        if not src:
            return

        remote_name = as_edit.text().strip()
        if remote_name:
            sep = "" if dst.endswith("/") else "/"
            dst = f"{dst}{sep}{remote_name}"

        if remember_cb.isChecked() and dst:
            save_dst = dest_edit.text().strip()
            if not save_dst.endswith("/"):
                save_dst += "/"
            self._save_scp_server(save_dst)

        if platform.system() == "Windows":
            # cmd.exe: double quotes, del instead of rm (scp ships with Windows 10+).
            cmd = f'scp "{src}" "{dst}"'
            if created_tmp:
                cmd += f' && del /q "{src}"'
            cmd += "\r\n"
        else:
            cmd = f"scp {shlex.quote(src)} {shlex.quote(dst)}"
            if created_tmp:
                cmd += f" && rm -f {shlex.quote(src)}"
            cmd += "\n"
        self._output.moveCursor(QTextCursor.MoveOperation.End)
        self._output.insertPlainText(f"$ {cmd}")
        if self._process and self._process.state() == QProcess.ProcessState.Running:
            self._process.write(cmd.encode("utf-8"))

    def _on_process_finished(self):
        self._output.appendPlainText("\n[Process terminated]")
        self._input.setEnabled(False)

    def _on_process_error(self, error):
        self._output.appendPlainText(f"\n[Process error: {error}]")
        self._input.setEnabled(False)

    def closeEvent(self, event):
        if self._process is not None:
            if platform.system() == "Windows":
                # terminate() posts WM_CLOSE, which console programs ignore:
                # it would only stall the window for the full timeout.
                self._process.kill()
                self._process.waitForFinished(1000)
            else:
                self._process.terminate()
                if not self._process.waitForFinished(3000):
                    self._process.kill()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# G16 INPUT GENERATOR DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class G16InputDialog(QDialog):
    _saved_state: dict = {}

    def __init__(self, mol, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate G16 Input")
        self.setMinimumSize(680, 540)
        self.setStyleSheet("""
            QComboBox:disabled, QSpinBox:disabled {
                background: #e8e8e8;
                color: #aaa;
                border: 1px solid #ccc;
            }
        """)
        self.mol = mol
        self._generating = False
        self._last_generated = ""
        self._loaded_route = None
        self._all_fields = []

        layout = QVBoxLayout(self)

        # ── Route card fields (each with custom override) ──
        form = QFormLayout()
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def combo_row(items):
            row = QHBoxLayout()
            combo = QComboBox()
            combo.addItems(items + ["Custom"])
            combo.setMinimumWidth(120)
            row.addWidget(combo)
            custom = QLineEdit()
            custom.setPlaceholderText("Enter custom value…")
            custom.setFixedWidth(180)
            custom.setVisible(False)
            row.addWidget(custom)
            row.addStretch()
            return combo, custom, row

        self._job_combo, self._job_custom, j_row = combo_row(
            ["opt", "freq", "opt freq", "sp", "scan"])
        form.addRow("Job type:", j_row)

        self._method_combo, self._method_custom, m_row = combo_row(
            ["b3lyp", "wb97xd", "m062x", "mp2", "hf", "ccsd(t)"])
        form.addRow("Method:", m_row)

        self._basis_combo, self._basis_custom, b_row = combo_row(
            ["6-31g(d)", "6-311+g(d,p)", "aug-cc-pvdz", "def2svp", "def2tzvp"])
        form.addRow("Basis set:", b_row)

        # Toggle custom field visibility and reconnect signals
        for combo, custom in [(self._job_combo, self._job_custom),
                              (self._method_combo, self._method_custom),
                              (self._basis_combo, self._basis_custom)]:
            combo.currentTextChanged.connect(
                lambda t, c=custom: c.setVisible(t == "Custom"))
            combo.currentTextChanged.connect(self._on_field_changed)
            custom.textChanged.connect(self._on_field_changed)

        btn_style = """
            QSpinBox::up-button, QSpinBox::down-button { width: 24px; }
            QSpinBox:disabled { background: #e8e8e8; color: #aaa; border: 1px solid #ccc; }
        """

        chg_layout = QHBoxLayout()
        self._charge_spin = QSpinBox()
        self._charge_spin.setRange(-10, 10)
        self._charge_spin.setValue(mol.charge)
        self._charge_spin.setMinimumWidth(90)
        self._charge_spin.setStyleSheet(btn_style)
        self._charge_spin.valueChanged.connect(self._on_field_changed)
        chg_layout.addWidget(self._charge_spin)
        chg_layout.addSpacing(4)
        chg_layout.addWidget(QLabel("Mult:"))
        self._mult_spin = QSpinBox()
        self._mult_spin.setRange(1, 10)
        self._mult_spin.setValue(spin_multiplicity(mol))
        self._mult_spin.setMinimumWidth(90)
        self._mult_spin.setStyleSheet(btn_style)
        self._mult_spin.valueChanged.connect(self._on_field_changed)
        chg_layout.addWidget(self._mult_spin)
        chg_layout.addStretch()
        form.addRow("Charge / Mult:", chg_layout)

        proc_row = QHBoxLayout()
        self._nproc_spin = QSpinBox()
        self._nproc_spin.setRange(1, 64)
        self._nproc_spin.setValue(4)
        self._nproc_spin.setMinimumWidth(90)
        self._nproc_spin.setStyleSheet(btn_style)
        self._nproc_spin.valueChanged.connect(self._on_field_changed)
        proc_row.addWidget(self._nproc_spin)
        proc_row.addStretch()
        form.addRow("Processors:", proc_row)

        mem_row = QHBoxLayout()
        self._mem_combo = QComboBox()
        self._mem_combo.addItems(["2", "4", "8", "16", "32", "64"])
        self._mem_combo.setEditable(True)
        self._mem_combo.setMinimumWidth(80)
        self._mem_combo.setCurrentText("8")
        self._mem_combo.currentTextChanged.connect(self._on_field_changed)
        mem_row.addWidget(self._mem_combo)
        mem_row.addStretch()
        form.addRow("Memory:", mem_row)

        self._custom_pct = QLineEdit()
        self._custom_pct.setPlaceholderText("e.g. %chk=file.chk")
        self._custom_pct.textChanged.connect(self._on_field_changed)
        form.addRow("Custom % line:", self._custom_pct)

        layout.addLayout(form)

        # ── Preview (editable) ──
        layout.addWidget(QLabel("Preview (editable):"))
        self._preview = QPlainTextEdit()
        self._preview.setMinimumHeight(180)
        self._preview.setFont(_monospace_font())
        self._preview.textChanged.connect(self._on_preview_edited)
        layout.addWidget(self._preview)

        # ── Buttons ──
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Save…")
        btn_save.clicked.connect(self._save)
        btn_layout.addWidget(btn_save)
        btn_copy_matrix = QPushButton("Copy matrix")
        btn_copy_matrix.clicked.connect(self._copy_matrix)
        btn_layout.addWidget(btn_copy_matrix)
        self._btn_sync = QPushButton("Sync from fields")
        self._btn_sync.clicked.connect(self._sync_from_fields)
        btn_layout.addWidget(self._btn_sync)
        btn_layout.addStretch()
        btn_ok = QPushButton("Ok")
        btn_ok.clicked.connect(self._on_ok)
        btn_layout.addWidget(btn_ok)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        # Collect all form fields for enable/disable
        self._all_fields = [
            self._job_combo, self._method_combo, self._basis_combo,
            self._charge_spin, self._mult_spin, self._nproc_spin,
            self._mem_combo, self._custom_pct,
        ]

        # Pre-populate fields from loaded molecule header
        if G16InputDialog._saved_state:
            self._restore_state()
        else:
            if mol.g16_nproc is not None:
                self._nproc_spin.blockSignals(True)
                self._nproc_spin.setValue(mol.g16_nproc)
                self._nproc_spin.blockSignals(False)
            if mol.g16_mem is not None:
                self._mem_combo.blockSignals(True)
                self._mem_combo.setCurrentText(mol.g16_mem.rstrip("GB"))
                self._mem_combo.blockSignals(False)
            if mol.g16_chk is not None:
                self._custom_pct.setText(f"%chk={mol.g16_chk}")
            if mol.g16_route is not None:
                self._loaded_route = mol.g16_route
                self._parse_route_into_fields(mol.g16_route)
            self._generating = True
            self._last_generated = self._build_text()
            self._preview.setPlainText(self._last_generated)
            self._generating = False

    def _on_ok(self):
        self._save_state()
        self.accept()

    def _save_state(self):
        G16InputDialog._saved_state = {
            "geometry": _geometry_key(self.mol),
            "preview": self._preview.toPlainText(),
            "job": self._job_combo.currentIndex(),
            "job_custom": self._job_custom.text(),
            "method": self._method_combo.currentIndex(),
            "method_custom": self._method_custom.text(),
            "basis": self._basis_combo.currentIndex(),
            "basis_custom": self._basis_custom.text(),
            "charge": self._charge_spin.value(),
            "mult": self._mult_spin.value(),
            "nproc": self._nproc_spin.value(),
            "mem": self._mem_combo.currentText(),
            "custom_pct": self._custom_pct.text(),
            "loaded_route": self._loaded_route,
        }

    def _restore_state(self):
        s = G16InputDialog._saved_state
        # Method, basis, resources carry over between molecules; the preview,
        # charge and multiplicity only when it is still the same geometry.
        same_mol = s.get("geometry") == _geometry_key(self.mol)
        if same_mol:
            self._generating = True
            self._preview.setPlainText(s.get("preview", ""))
            self._generating = False
            self._last_generated = s.get("preview", "")

        self._job_combo.blockSignals(True)
        self._method_combo.blockSignals(True)
        self._basis_combo.blockSignals(True)
        last = self._job_combo.count() - 1
        self._job_combo.setCurrentIndex(s.get("job", 0))
        self._job_custom.setText(s.get("job_custom", ""))
        self._job_custom.setVisible(s.get("job", 0) == last)
        self._method_combo.setCurrentIndex(s.get("method", 0))
        self._method_custom.setText(s.get("method_custom", ""))
        self._method_custom.setVisible(s.get("method", 0) == last)
        self._basis_combo.setCurrentIndex(s.get("basis", 0))
        self._basis_custom.setText(s.get("basis_custom", ""))
        self._basis_custom.setVisible(s.get("basis", 0) == last)
        self._job_combo.blockSignals(False)
        self._method_combo.blockSignals(False)
        self._basis_combo.blockSignals(False)

        if same_mol:
            self._charge_spin.blockSignals(True)
            self._charge_spin.setValue(s.get("charge", self._charge_spin.value()))
            self._charge_spin.blockSignals(False)
            self._mult_spin.blockSignals(True)
            self._mult_spin.setValue(s.get("mult", self._mult_spin.value()))
            self._mult_spin.blockSignals(False)
        self._nproc_spin.blockSignals(True)
        self._nproc_spin.setValue(s.get("nproc", self._nproc_spin.value()))
        self._nproc_spin.blockSignals(False)
        self._mem_combo.blockSignals(True)
        self._mem_combo.setCurrentText(s.get("mem", self._mem_combo.currentText()))
        self._mem_combo.blockSignals(False)
        self._custom_pct.blockSignals(True)
        self._custom_pct.setText(s.get("custom_pct", ""))
        self._custom_pct.blockSignals(False)
        self._loaded_route = s.get("loaded_route")
        if not same_mol:
            self._generating = True
            self._last_generated = self._build_text()
            self._preview.setPlainText(self._last_generated)
            self._generating = False

    def _set_fields_enabled(self, enabled: bool):
        for w in self._all_fields:
            w.setEnabled(enabled)
        self._btn_sync.setEnabled(not enabled)

    def _on_preview_edited(self):
        if self._generating:
            return
        if self._preview.toPlainText() == self._last_generated:
            return
        self._set_fields_enabled(False)

    def _on_field_changed(self):
        self._loaded_route = None
        self._sync_from_fields()

    def _field_val(self, combo, custom) -> str:
        if combo.currentText() == "Custom":
            txt = custom.text().strip()
            return txt if txt else "CUSTOM"
        return combo.currentText()

    def _set_combo_field(self, combo, custom, value):
        if not value:
            return
        items = [combo.itemText(i).lower() for i in range(combo.count())]
        if value.lower() in items:
            idx = items.index(value.lower())
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)
            custom.setVisible(False)
        else:
            combo.blockSignals(True)
            combo.setCurrentText("Custom")
            combo.blockSignals(False)
            custom.blockSignals(True)
            custom.setText(value)
            custom.setVisible(True)
            custom.blockSignals(False)

    def _parse_route_into_fields(self, route: str):
        text = route.lstrip("#").strip()
        tokens = text.split()
        if not tokens:
            return
        slash_idx = -1
        for i, tok in enumerate(tokens):
            if "/" in tok:
                slash_idx = i
        if slash_idx >= 0:
            method_basis = tokens[slash_idx]
            parts = method_basis.split("/", 1)
            method_val = parts[0]
            basis_val = parts[1] if len(parts) > 1 else ""
            if slash_idx + 1 < len(tokens):
                basis_val = basis_val + " " + " ".join(tokens[slash_idx + 1:])
            job_val = " ".join(tokens[:slash_idx])
        else:
            job_val = text
            method_val = ""
            basis_val = ""
        self._set_combo_field(self._job_combo, self._job_custom, job_val)
        self._set_combo_field(self._method_combo, self._method_custom, method_val)
        self._set_combo_field(self._basis_combo, self._basis_custom, basis_val)

    def _build_route(self) -> str:
        job = self._field_val(self._job_combo, self._job_custom)
        method = self._field_val(self._method_combo, self._method_custom)
        basis = self._field_val(self._basis_combo, self._basis_custom)
        return f"# {job} {method}/{basis}"

    def _build_text(self) -> str:
        nproc = self._nproc_spin.value()
        mem = self._mem_combo.currentText().rstrip("GB") + "GB"
        route = self._loaded_route if self._loaded_route else self._build_route()
        charge = self._charge_spin.value()
        mult = self._mult_spin.value()
        custom_pct = self._custom_pct.text().strip()

        header = [f"%nprocshared={nproc}", f"%mem={mem}"]
        if custom_pct:
            header.append(custom_pct)
        header.append(route)

        lines = [
            *header,
            "",
            self.mol.name or chemical_formula(self.mol) or "Molvector",  # Gaussian needs a title
            "",
            f"{charge} {mult}",
        ]
        for a in self.mol.atoms:
            lines.append(f"{gaussian_atom_label(a.element):3s} {a.x:12.6f} {a.y:12.6f} {a.z:12.6f}")
        lines.extend(["", "", ""])
        return "\n".join(lines)

    def _sync_from_fields(self):
        self._generating = True
        self._last_generated = self._build_text()
        self._preview.setPlainText(self._last_generated)
        self._generating = False
        self._set_fields_enabled(True)
        self._btn_sync.setEnabled(False)

    def _copy_matrix(self):
        lines = []
        for a in self.mol.atoms:
            lines.append(f"{gaussian_atom_label(a.element):3s} {a.x:12.6f} {a.y:12.6f} {a.z:12.6f}")
        QApplication.clipboard().setText("\n".join(lines))

    def _save(self):
        safe = get_safe_filename(self.mol.name, "input")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Gaussian Input", f"{safe}.gjf",
            "Gaussian input (*.gjf *.com);;All files (*)"
        )
        if not path:
            return
        try:
            # LF endings: Gaussian on Linux clusters chokes on Windows CRLF.
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(self._preview.toPlainText())
            QMessageBox.information(self, "Saved", f"Gaussian input saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CPK LEGEND PANEL  (sidebar)
# ─────────────────────────────────────────────────────────────────────────────

class LegendPanel(QGroupBox):
    elementColorChanged = pyqtSignal(str, str)  # (element_symbol, hex_color)
    atomColorChanged = pyqtSignal(int, str)     # (atom_index, hex_color)

    MAX_VISIBLE_ROWS = 10

    def __init__(self, parent=None):
        super().__init__("Elements", parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10,14,10,10)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self._scroll)

        rows_widget = QWidget()
        self._layout = QVBoxLayout(rows_widget)
        self._layout.setSpacing(4)
        self._layout.setContentsMargins(0,0,0,0)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(rows_widget)

        self._rows: list = []
        self._custom_atom_map: Dict[str, int] = {}

    def update_for(self, mol: Molecule, overrides: dict):
        for w in self._rows:
            w.setParent(None)
        self._rows.clear()
        self._custom_atom_map.clear()

        # Standard entries — one per element for atoms without custom colors
        seen_elem: set = set()
        for atom in mol.atoms:
            if atom.color:
                continue
            e = atom.element
            if e in seen_elem:
                continue
            seen_elem.add(e)
            color = overrides.get(e, CPK_BASE.get(e, DEFAULT_BASE))
            name  = ELEM_FULL_NAME.get(e, e)
            self._add_row(e, color, f"{e} - {name}", clickable=True)

        # Custom entries — atoms with per-atom color override
        custom_counts: Dict[str, int] = {}
        for idx, atom in enumerate(mol.atoms):
            if not atom.color:
                continue
            e = atom.element
            custom_counts[e] = custom_counts.get(e, 0) + 1
            n = custom_counts[e]
            label = f"{e} - Custom {n}"
            self._custom_atom_map[label] = idx
            self._add_row(label, atom.color, label, clickable=True, custom=True)

        self._update_scroll_height()

    def _update_scroll_height(self):
        n = len(self._rows)
        if n == 0:
            self._scroll.setMinimumHeight(0)
            self._scroll.setMaximumHeight(0)
        else:
            row_h = self._rows[0].sizeHint().height()
            spacing = self._layout.spacing()
            visible_rows = min(n, self.MAX_VISIBLE_ROWS)
            visible_h = visible_rows * row_h + (visible_rows - 1) * spacing
            self._scroll.setMinimumHeight(visible_h)
            self._scroll.setMaximumHeight(visible_h)
        # Pin the group box itself to its content height immediately —
        # sizeHint() recomputes synchronously, unlike the geometry the
        # parent layout applies (which can lag a cycle behind after
        # several back-to-back updates).
        self.setFixedHeight(self.sizeHint().height())

    def _add_row(self, key: str, color: str, label: str, clickable: bool = True, custom: bool = False):
        row = QWidget()
        rl  = QHBoxLayout(row)
        rl.setContentsMargins(0,0,0,0)
        rl.setSpacing(7)

        swatch = QPushButton()
        swatch.setFixedSize(14, 14)
        swatch.setStyleSheet(
            f"background:{color}; border-radius:7px; border:1px solid #555;"
        )
        if clickable:
            swatch.setCursor(Qt.CursorShape.PointingHandCursor)
            swatch.setToolTip(f"Click to change {label} colour")
            if custom:
                swatch.clicked.connect(lambda _, lbl=label, c=color: self._pick_atom_color(lbl, c))
            else:
                swatch.clicked.connect(lambda _, e=key, c=color: self._pick_color(e, c))

        lbl = QLabel(label)
        lbl.setObjectName("dim")
        rl.addWidget(swatch)
        rl.addWidget(lbl)
        rl.addStretch()

        self._layout.addWidget(row)
        self._rows.append(row)

    def _pick_color(self, elem: str, current_hex: str):
        dlg = QColorDialog(QColor(current_hex), self)
        dlg.setWindowTitle(f"Pick Colour for {ELEM_FULL_NAME.get(elem, elem)}")
        if dlg.exec():
            new_hex = dlg.selectedColor().name()
            self.elementColorChanged.emit(elem, new_hex)

    def _pick_atom_color(self, label: str, current_hex: str):
        idx = self._custom_atom_map.get(label)
        if idx is None:
            return
        dlg = QColorDialog(QColor(current_hex), self)
        dlg.setWindowTitle(f"Pick Colour for {label}")
        if dlg.exec():
            new_hex = dlg.selectedColor().name()
            self.atomColorChanged.emit(idx, new_hex)


# ─────────────────────────────────────────────────────────────────────────────
# PERIODIC TABLE DIALOG
# ─────────────────────────────────────────────────────────────────────────────

_PERIODIC_TABLE_LAYOUT = [
    # (symbol, Z, row, col)
    ("H",1,0,0), ("D",0,0,4), ("T",-1,0,5), ("13C",600,0,6), ("14C",601,0,7), ("15N",700,0,8), ("He",2,0,17),
    ("Li",3,1,0), ("Be",4,1,1), ("B",5,1,12), ("C",6,1,13), ("N",7,1,14), ("O",8,1,15), ("F",9,1,16), ("Ne",10,1,17),
    ("Na",11,2,0), ("Mg",12,2,1), ("Al",13,2,12), ("Si",14,2,13), ("P",15,2,14), ("S",16,2,15), ("Cl",17,2,16), ("Ar",18,2,17),
    ("K",19,3,0), ("Ca",20,3,1), ("Sc",21,3,2), ("Ti",22,3,3), ("V",23,3,4), ("Cr",24,3,5), ("Mn",25,3,6), ("Fe",26,3,7), ("Co",27,3,8), ("Ni",28,3,9), ("Cu",29,3,10), ("Zn",30,3,11), ("Ga",31,3,12), ("Ge",32,3,13), ("As",33,3,14), ("Se",34,3,15), ("Br",35,3,16), ("Kr",36,3,17),
    ("Rb",37,4,0), ("Sr",38,4,1), ("Y",39,4,2), ("Zr",40,4,3), ("Nb",41,4,4), ("Mo",42,4,5), ("Tc",43,4,6), ("Ru",44,4,7), ("Rh",45,4,8), ("Pd",46,4,9), ("Ag",47,4,10), ("Cd",48,4,11), ("In",49,4,12), ("Sn",50,4,13), ("Sb",51,4,14), ("Te",52,4,15), ("I",53,4,16), ("Xe",54,4,17),
    ("Cs",55,5,0), ("Ba",56,5,1), ("La",57,5,2), ("Hf",72,5,3), ("Ta",73,5,4), ("W",74,5,5), ("Re",75,5,6), ("Os",76,5,7), ("Ir",77,5,8), ("Pt",78,5,9), ("Au",79,5,10), ("Hg",80,5,11), ("Tl",81,5,12), ("Pb",82,5,13), ("Bi",83,5,14), ("Po",84,5,15), ("At",85,5,16), ("Rn",86,5,17),
    ("Fr",87,6,0), ("Ra",88,6,1), ("Ac",89,6,2), ("Rf",104,6,3), ("Db",105,6,4), ("Sg",106,6,5), ("Bh",107,6,6), ("Hs",108,6,7), ("Mt",109,6,8), ("Ds",110,6,9), ("Rg",111,6,10), ("Cn",112,6,11), ("Nh",113,6,12), ("Fl",114,6,13), ("Mc",115,6,14), ("Lv",116,6,15), ("Ts",117,6,16), ("Og",118,6,17),
    # F-block: lanthanides
    ("Ce",58,7,2), ("Pr",59,7,3), ("Nd",60,7,4), ("Pm",61,7,5), ("Sm",62,7,6), ("Eu",63,7,7), ("Gd",64,7,8), ("Tb",65,7,9), ("Dy",66,7,10), ("Ho",67,7,11), ("Er",68,7,12), ("Tm",69,7,13), ("Yb",70,7,14), ("Lu",71,7,15),
    # F-block: actinides
    ("Th",90,8,2), ("Pa",91,8,3), ("U",92,8,4), ("Np",93,8,5), ("Pu",94,8,6), ("Am",95,8,7), ("Cm",96,8,8), ("Bk",97,8,9), ("Cf",98,8,10), ("Es",99,8,11), ("Fm",100,8,12), ("Md",101,8,13), ("No",102,8,14), ("Lr",103,8,15),
]

_ELEM_CATEGORY_COLORS = {
    "nonmetal":       "#4CAF50",
    "noble_gas":      "#9C27B0",
    "alkali_metal":   "#FF5722",
    "alkaline_earth": "#FF9800",
    "metalloid":      "#00BCD4",
    "halogen":        "#1E88E5",
    "transition":     "#42A5F5",
    "post_transition":"#abad98",
    "lanthanide":     "#E91E63",
    "actinide":       "#D32F2F",
    "unknown":        "#757575",
}

def _element_category(sym: str, Z: int) -> str:
    if sym in ("D", "T", "13C", "14C", "15N"): return "nonmetal"
    if Z == 1: return "nonmetal"
    if Z == 2: return "noble_gas"
    if 3 <= Z <= 4: return "alkali_metal" if Z == 3 else "alkaline_earth"
    if 5 <= Z <= 10:
        if Z == 5: return "metalloid"
        return "nonmetal" if Z in (6,7,8) else "halogen" if Z in (9,17,35,53,85,117) else "noble_gas"
    if 11 <= Z <= 18:
        if Z == 11: return "alkali_metal"
        if Z == 12: return "alkaline_earth"
        if Z in (13,): return "post_transition"
        if Z in (14,): return "metalloid"
        if Z in (15,16): return "nonmetal"
        if Z == 17: return "halogen"
        if Z == 18: return "noble_gas"
    if 19 <= Z <= 36:
        if Z in (19,37,55,87): return "alkali_metal"
        if Z in (20,38,56,88): return "alkaline_earth"
        if 21 <= Z <= 30: return "transition"
        if 31 <= Z <= 36:
            if Z == 31: return "post_transition"
            if Z in (32,): return "metalloid"
            if Z in (33,34): return "metalloid" if Z == 33 else "nonmetal"
            return "halogen" if Z == 35 else "noble_gas"
    if 37 <= Z <= 54:
        if Z in (37,55,87): return "alkali_metal"
        if Z in (38,56,88): return "alkaline_earth"
        if Z == 39 or (40 <= Z <= 48):
            return "transition"
        if 49 <= Z <= 54:
            if Z == 49: return "post_transition"
            if Z == 50: return "post_transition"
            if Z == 51: return "metalloid"
            if Z == 52: return "metalloid"
            return "halogen" if Z == 53 else "noble_gas"
    if 55 <= Z <= 86:
        if Z in (55,87): return "alkali_metal"
        if Z in (56,88): return "alkaline_earth"
        if 57 <= Z <= 71: return "lanthanide"
        if 72 <= Z <= 80: return "transition"
        if 81 <= Z <= 86:
            if Z == 81: return "post_transition"
            if Z in (82,83): return "post_transition"
            if Z in (84,): return "post_transition"
            return "halogen" if Z == 85 else "noble_gas"
    if 89 <= Z <= 103: return "actinide"
    if 104 <= Z <= 111: return "transition"
    if 112 <= Z <= 118:
        if Z in (113,114,115,116): return "post_transition"
        if Z == 117: return "halogen"
        if Z == 118: return "noble_gas"
    return "unknown"


class PeriodicTableDialog(QDialog):
    elementSelected = pyqtSignal(str)

    def __init__(self, parent=None, current_element: str = "C", color_overrides: dict = None):
        super().__init__(parent)
        self.setWindowTitle("Select Element — Periodic Table")
        self.setModal(True)
        self.setMinimumSize(1020, 640)
        self._overrides = dict(color_overrides or {})

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        grid = QGridLayout()
        grid.setSpacing(3)

        btn_size = 52
        dot_size = 13
        dot_margin = 4
        btn_font = QFont("", 10, QFont.Weight.Bold)

        by_Z = {}
        for sym, Z, r, c in _PERIODIC_TABLE_LAYOUT:
            by_Z[Z] = (sym, r, c)

        self._all_buttons: Dict[str, QPushButton] = {}

        for sym, Z, r, c in _PERIODIC_TABLE_LAYOUT:
            btn = QPushButton(sym)
            btn.setFixedSize(btn_size, btn_size)
            btn.setFont(btn_font)

            cat = _element_category(sym, Z)
            cat_col = _ELEM_CATEGORY_COLORS.get(cat, "#757575")
            is_dark = sum(int(cat_col[i:i+2],16) for i in (1,3,5)) < 400
            text_color = "#ffffff" if is_dark else "#000000"
            btn.setStyleSheet(
                f"QPushButton {{ background:{cat_col}; color:{text_color}; "
                f"border:1px solid rgba(0,0,0,0.2); border-radius:4px; }}"
                f"QPushButton:hover {{ border:2px solid white; }}"
            )

            is_current = (sym == current_element)
            if is_current:
                btn.setStyleSheet(
                    f"QPushButton {{ background:{cat_col}; color:{text_color}; "
                    f"border:2px solid #FFD700; border-radius:4px; }}"
                )

            # Small CPK-coloured dot in the bottom-right corner
            cpk = self._overrides.get(sym, CPK_BASE.get(sym, DEFAULT_BASE))
            dot = QLabel()
            dot.setFixedSize(dot_size, dot_size)
            dot.setStyleSheet(
                f"background:{cpk}; border:1px solid rgba(0,0,0,0.35); "
                f"border-radius:{dot_size // 2}px;"
            )
            dot.setParent(btn)
            dot.move(btn_size - dot_size - dot_margin, btn_size - dot_size - dot_margin)

            if self._overrides.get(sym):
                cpk_txt = f"Custom: {cpk}"
            else:
                cpk_txt = f"CPK: {cpk}"
            btn.setToolTip(f"{sym} ({Z}) — {ELEM_FULL_NAME.get(sym, 'Unknown')}\n{cpk_txt}")

            def _make_handler(s):
                return lambda checked, sym=s: self._select_element(sym)

            btn.clicked.connect(_make_handler(sym))
            grid.addWidget(btn, r, c)
            self._all_buttons[sym] = btn

        # Headers for characteristic rows
        label_font = QFont("", 8)
        lbl_la = QLabel("Lanthanides")
        lbl_la.setFont(label_font)
        lbl_la.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(lbl_la, 7, 0, 1, 2)

        lbl_ac = QLabel("Actinides")
        lbl_ac.setFont(label_font)
        lbl_ac.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(lbl_ac, 8, 0, 1, 2)

        layout.addLayout(grid)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Cancel")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _select_element(self, sym: str):
        self.elementSelected.emit(sym)
        self.accept()





# ── Shortcut Configuration Dialog ───────────────────────────────────────────

class ShortcutDialog(QDialog):
    CONFIG_FILE = CONFIG_FILE

    def __init__(self, shortcut_actions: dict, alt_defaults: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Shortcuts")
        self.setMinimumSize(640, 420)
        self._actions = dict(shortcut_actions)
        self._alt_defaults = alt_defaults or {}
        self._initial_primary = {}
        self._initial_alt = {}

        layout = QVBoxLayout(self)

        # Conflict warning label
        self._conflict_lbl = QLabel()
        self._conflict_lbl.setVisible(False)
        layout.addWidget(self._conflict_lbl)

        # Table
        self._table = QTableWidget(len(self._actions), 3)
        self._table.setHorizontalHeaderLabels(["Action", "Primary", "Alt"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        self._editors = []
        for row, (aid, action) in enumerate(self._actions.items()):
            name_item = QTableWidgetItem(action.text().replace("&", ""))
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            ks1 = QKeySequenceEdit(action.shortcut())
            ks1.setClearButtonEnabled(True)
            self._table.setCellWidget(row, 1, ks1)
            ks1.keySequenceChanged.connect(self._check_conflicts)

            alt_seq = self._alt_defaults.get(aid, "")
            ks2 = QKeySequenceEdit(QKeySequence(alt_seq))
            ks2.setClearButtonEnabled(True)
            self._table.setCellWidget(row, 2, ks2)
            ks2.keySequenceChanged.connect(self._check_conflicts)

            self._editors.append((ks1, ks2))
            self._initial_primary[aid] = ks1.keySequence().toString()
            self._initial_alt[aid] = ks2.keySequence().toString()

        self._check_conflicts()
        layout.addWidget(self._table)

        # Buttons
        btn_row = QHBoxLayout()
        btn_restore = QPushButton("Restore Defaults")
        btn_restore.clicked.connect(self._restore_defaults)
        btn_row.addWidget(btn_restore)

        btn_make_default = QPushButton("Make Default")
        btn_make_default.clicked.connect(self._make_default)
        btn_row.addWidget(btn_make_default)

        btn_row.addStretch()

        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_ok)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(btn_cancel)

        layout.addLayout(btn_row)

    def _check_conflicts(self):
        usage = {}
        for row, (aid, action) in enumerate(self._actions.items()):
            for col, editor in enumerate([self._editors[row][0], self._editors[row][1]]):
                ks = editor.keySequence()
                if not ks.isEmpty():
                    seq_str = ks.toString().lower()
                    if seq_str not in usage:
                        usage[seq_str] = []
                    usage[seq_str].append((aid, action.text().replace("&", ""), col))

        conflicts = {k: v for k, v in usage.items() if len(v) > 1}

        if conflicts:
            lines = []
            for seq, items in conflicts.items():
                names = [f"{name} ({'Alt' if c == 1 else 'Primary'})" for _, name, c in items]
                lines.append(f"⚠ {seq} used by: {', '.join(names)}")
            self._conflict_lbl.setText(
                '<span style="color:#ff6b6b; font-weight:bold;">' +
                "<br>".join(lines) +
                '</span>'
            )
            self._conflict_lbl.setVisible(True)
        else:
            self._conflict_lbl.setVisible(False)

    def _restore_defaults(self):
        for row, (aid, action) in enumerate(self._actions.items()):
            primary = MainWindow.DEFAULT_SHORTCUTS.get(aid, "")
            alt = MainWindow.ALT_SHORTCUT_DEFAULTS.get(aid, "")
            self._editors[row][0].setKeySequence(QKeySequence(primary))
            self._editors[row][1].setKeySequence(QKeySequence(alt))
        self._check_conflicts()

    def _make_default(self):
        primary, alt = self._collect_shortcuts()
        try:
            _update_config({"shortcuts": primary, "alt_shortcuts": alt})
            QMessageBox.information(self, "Saved", f"Shortcuts saved to:\n{self.CONFIG_FILE}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save shortcuts:\n{e}")

    def _has_changes(self) -> bool:
        for row, (aid, action) in enumerate(self._actions.items()):
            ks1 = self._editors[row][0].keySequence().toString()
            ks2 = self._editors[row][1].keySequence().toString()
            if ks1 != self._initial_primary.get(aid, "") or ks2 != self._initial_alt.get(aid, ""):
                return True
        return False

    def _on_cancel(self):
        if self._has_changes():
            ret = QMessageBox.warning(
                self, "Unsaved Changes",
                "You have unsaved shortcut changes. Discard them?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if ret == QMessageBox.StandardButton.Discard:
                self.reject()
        else:
            self.reject()

    def closeEvent(self, event):
        if self._has_changes():
            ret = QMessageBox.warning(
                self, "Unsaved Changes",
                "You have unsaved shortcut changes. Discard them?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if ret == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def get_shortcuts(self) -> tuple:
        return self._collect_shortcuts()

    def _collect_shortcuts(self) -> tuple:
        primary = {}
        alt = {}
        for row, (aid, action) in enumerate(self._actions.items()):
            ks1 = self._editors[row][0].keySequence()
            ks2 = self._editors[row][1].keySequence()
            primary[aid] = ks1.toString() if not ks1.isEmpty() else ""
            alt[aid] = ks2.toString() if not ks2.isEmpty() else ""
        return primary, alt


# ─────────────────────────────────────────────────────────────────────────────
# Quick SVG Export Dialog ────────────────────────────────────────────────────

class QuickExportDialog(QDialog):
    """A dialog with a draggable SVG target for drag-and-drop into external editors."""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self._tmp_path = None
        self.setWindowTitle("Quick SVG Export")
        self.setFixedSize(320, 220)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        lbl = QLabel("Drag the icon into Inkscape or another SVG editor:")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self._drag_widget = _DragWidget(self._get_svg_path, self)
        self._drag_widget.setFixedSize(120, 80)
        layout.addWidget(self._drag_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        note = QLabel("Note: in Inkscape, you may have to drag the molecule to see gradients.")
        note.setObjectName("dim")
        note.setWordWrap(True)
        note.setStyleSheet("font-size:10px;")
        layout.addWidget(note)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignCenter)

    def _get_svg_path(self) -> str:
        # Re-render every drag rather than caching: the canvas' appearance
        # (colors, legend, etc.) can change while this dialog stays open.
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass
        svg_data = self._canvas.get_svg_bytes(export_mode=True)
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            f.write(svg_data)
            self._tmp_path = f.name
        return self._tmp_path

    def closeEvent(self, event):
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass
            self._tmp_path = None
        super().closeEvent(event)


_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _pretty_formula(formula: str) -> str:
    """C24H12 -> C₂₄H₁₂, C60+ -> C₆₀⁺ (subscript counts, superscript charge)."""
    charge = formula[len(formula.rstrip("+-")):]
    body = formula[:len(formula) - len(charge)].translate(_SUBSCRIPTS)
    return body + charge.replace("+", "⁺").replace("-", "⁻")


class CatalogDialog(QDialog):
    """File → Open from Catalog…: search the bundled molecule catalog with a preview.

    The preview matches the main window's active view: a Chemvector 2D
    depiction in the 2D view's *layout* ("match3d": the 3D geometry seen
    face-on, as the 2D view shows it on entry; "clean": RDKit's idealized
    depiction), or with *view3d* (the 3D MoleculeCanvas) a ball-and-stick
    rendering in that canvas' current style.
    """

    # Filters: one row of toggle buttons per catalog facet (catalog.facets()).
    # A row's checked buttons combine with AND (all, the default) or, in its
    # OR mode (Alt+click, or its badge), OR (any); rows combine with AND. Each button
    # shows how many molecules it would give and is disabled at 0.
    PREVIEW_W, PREVIEW_H = 360, 300

    def __init__(self, parent=None, view3d=None, layout: str = "match3d"):
        super().__init__(parent)
        from molvector import catalog
        self._catalog = catalog
        self._view3d = view3d
        self._layout = layout
        self.selected_record = None
        self.setWindowTitle("Open from Catalog")
        self.resize(940, 760)

        layout = QVBoxLayout(self)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search by name, formula (C24H12) or SMILES…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._refresh)
        layout.addWidget(self._search)

        # One row per facet: its name, an OR/AND badge (click to switch) and
        # the option buttons. Rows start in AND (a molecule must have all of
        # the row's choices); Alt+click a filter: OR (any of them);
        # Shift+click: back to AND; plain click: the row's current mode. Rows
        # combine with AND. Options no catalog entry has are not shown.
        totals = catalog.facet_counts()
        filters_box = QVBoxLayout()
        filters_box.setSpacing(4)
        self._facet_buttons = {}         # facet -> {value: (button, label)}
        self._mode_badges = {}           # facet -> badge button
        self._modes = {}                 # facet -> "or" / "and" (missing: "and", the default)
        for facet, facet_label, options in catalog.facets():
            shown = [o for o in options if totals[facet][o[0]] > 0]
            if not shown:
                continue
            line = QHBoxLayout()
            line.setSpacing(4)
            title = QLabel(facet_label)
            title.setObjectName("dim")
            title.setFixedWidth(118)
            line.addWidget(title)
            badge = QPushButton()
            badge.setObjectName("facetMode")
            badge.setFixedWidth(46)
            badge.setToolTip("How this row's choices combine — click to switch\n"
                             "OR: any of them · AND: all of them")
            badge.clicked.connect(lambda _c=False, f=facet: self._set_mode(
                f, "or" if self._modes.get(f, "and") == "and" else "and"))
            self._mode_badges[facet] = badge
            line.addWidget(badge)
            buttons = {}
            for value, label, tip in shown:
                btn = QPushButton(label)
                btn.setObjectName("facetButton")
                btn.setCheckable(True)
                btn.setToolTip(tip + f"\n{ALT_CLICK_HINT}: OR within the row · Shift+click: AND")
                btn.clicked.connect(lambda _c=False, f=facet: self._on_facet_clicked(f))
                buttons[value] = (btn, label)
                line.addWidget(btn)
            line.addStretch(1)
            if facet == "detected":
                line.addWidget(self._detections_source_label(catalog.sources().get("astrochymist")))
            self._facet_buttons[facet] = buttons
            filters_box.addLayout(line)

        ring_lo, ring_hi = catalog.ring_range()
        ring_row = QHBoxLayout()
        ring_title = QLabel("Ring count")
        ring_title.setObjectName("dim")
        ring_title.setFixedWidth(118 + 4 + 46)
        ring_row.addWidget(ring_title)
        self._ring_min = QSpinBox()
        self._ring_max = QSpinBox()
        for box, value in ((self._ring_min, ring_lo), (self._ring_max, ring_hi)):
            box.setRange(ring_lo, ring_hi)
            box.setValue(value)
            box.valueChanged.connect(self._on_ring_range_changed)
        ring_row.addWidget(self._ring_min)
        ring_row.addWidget(QLabel("to"))
        ring_row.addWidget(self._ring_max)
        ring_row.addStretch(1)
        hint = QLabel(f"Within a row: AND · {ALT_CLICK_HINT}: OR · Shift+click: back to AND · "
                      "rows combine with AND")
        hint.setObjectName("dim")
        hint.setStyleSheet("font-size:11px;")
        ring_row.addWidget(hint)
        btn_clear = QPushButton("Clear Filters")
        btn_clear.clicked.connect(self._clear_filters)
        ring_row.addWidget(btn_clear)
        filters_box.addLayout(ring_row)
        layout.addLayout(filters_box)
        for facet in self._facet_buttons:
            self._show_mode(facet)

        body = QHBoxLayout()
        left = QVBoxLayout()
        # Name | Formula | Mass, aligned in columns; clicking the Name or Mass
        # header sorts by it (the Sort box below follows).
        self._list = QTreeWidget()
        self._list.setColumnCount(3)
        self._list.setHeaderLabels(["Name", "Formula", "Mass (u)"])
        self._list.setRootIsDecorated(False)
        self._list.setUniformRowHeights(True)
        self._list.setAllColumnsShowFocus(True)
        header = self._list.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._list.headerItem().setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.setSectionsClickable(True)
        header.sectionClicked.connect(self._on_header_clicked)
        self._list.currentItemChanged.connect(self._on_current_changed)
        self._list.itemDoubleClicked.connect(lambda _item, _col: self.accept())
        left.addWidget(self._list)
        count_row = QHBoxLayout()
        self._count = QLabel()
        self._count.setObjectName("dim")
        count_row.addWidget(self._count)
        count_row.addStretch(1)
        sort_label = QLabel("Sort")
        sort_label.setObjectName("dim")
        count_row.addWidget(sort_label)
        self._sort = QComboBox()
        for key, label in catalog.SORTS:
            self._sort.addItem(label, key)
        self._sort.currentIndexChanged.connect(self._refresh)
        count_row.addWidget(self._sort)
        left.addLayout(count_row)
        body.addLayout(left, 1)

        right = QVBoxLayout()
        # The 2D SVG is rendered without a background (export mode: no name
        # caption either), so the frame provides the card behind it: white,
        # or the 3D canvas' own background.
        preview_bg = view3d.background if view3d is not None else "#ffffff"
        preview_frame = QFrame()
        preview_frame.setObjectName("catalogPreview")
        preview_frame.setStyleSheet(
            f"#catalogPreview {{ background: {preview_bg}; border-radius: 6px; }}")
        frame_layout = QVBoxLayout(preview_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        self._preview = QSvgWidget()
        self._preview.setFixedSize(self.PREVIEW_W, self.PREVIEW_H)
        frame_layout.addWidget(self._preview)
        right.addWidget(preview_frame, alignment=Qt.AlignmentFlag.AlignHCenter)
        self._info = QLabel()
        self._info.setWordWrap(True)
        self._info.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self._info.setOpenExternalLinks(True)   # detection papers
        self._info.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QFrame.Shape.NoFrame)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        info_scroll.setFixedWidth(self.PREVIEW_W)
        info_scroll.setWidget(self._info)
        right.addWidget(info_scroll, 1)
        body.addLayout(right)
        layout.addLayout(body, 1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._refresh()
        self._search.setFocus()

    @staticmethod
    def _detections_source_label(src) -> QLabel:
        """"The Astrochymist, updated 1 Oct 2026", linked to the list."""
        label = QLabel()
        label.setObjectName("dim")
        label.setStyleSheet("font-size:11px;")
        if not src:
            return label
        def nice(iso):
            try:
                d = datetime.date.fromisoformat(iso)
                return f"{d.day} {d.strftime('%b %Y')}"
            except (TypeError, ValueError):
                return iso or "?"
        label.setText(f"<a href='{src['url']}'>{src['name']}</a>, updated {nice(src.get('updated'))}")
        label.setToolTip(f"List of molecules detected in space as of {nice(src.get('updated'))}; "
                         f"bundled copy fetched {nice(src.get('fetched'))}")
        label.setOpenExternalLinks(True)
        return label

    def _on_header_clicked(self, column: int):
        """Name header: sort by name; Mass header: by mass, flipping direction on each click."""
        current = self._sort.currentData()
        key = {0: "name", 2: "mass-desc" if current == "mass" else "mass"}.get(column)
        if key is not None:
            self._sort.setCurrentIndex(self._sort.findData(key))

    def _filters(self):
        return {facet: [v for v, (btn, _) in buttons.items() if btn.isChecked()]
                for facet, buttons in self._facet_buttons.items()}

    def _on_facet_clicked(self, facet: str):
        mods = QApplication.keyboardModifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            self._set_mode(facet, "and")
        elif alt_click(mods):
            self._set_mode(facet, "or")
        else:
            self._refresh()

    def _effective_modes(self):
        """Every row's mode, AND unless switched to OR."""
        return {facet: self._modes.get(facet, "and") for facet in self._facet_buttons}

    def _set_mode(self, facet: str, mode: str):
        self._modes[facet] = mode
        self._show_mode(facet)
        self._refresh()

    def _show_mode(self, facet: str):
        """Badge text, and the amber highlight of an OR row (AND is the default look)."""
        is_and = self._modes.get(facet, "and") == "and"
        badge = self._mode_badges[facet]
        badge.setText("AND" if is_and else "OR")
        badge.setProperty("orMode", not is_and)
        for btn, _ in self._facet_buttons[facet].values():
            btn.setProperty("orMode", not is_and)
        for w in [badge] + [b for b, _ in self._facet_buttons[facet].values()]:
            w.style().unpolish(w)
            w.style().polish(w)

    def _rings(self):
        return (self._ring_min.value(), self._ring_max.value())

    def _on_ring_range_changed(self, _value):
        # Keep min <= max: moving one past the other drags the other along.
        if self._ring_min.value() > self._ring_max.value():
            other = self._ring_max if self.sender() is self._ring_min else self._ring_min
            other.blockSignals(True)
            other.setValue(self.sender().value())
            other.blockSignals(False)
        self._refresh()

    def _clear_filters(self):
        self._modes = {}
        for facet in self._facet_buttons:
            self._show_mode(facet)
        for buttons in self._facet_buttons.values():
            for btn, _ in buttons.values():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
        lo, hi = self._catalog.ring_range()
        for box, value in ((self._ring_min, lo), (self._ring_max, hi)):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self._refresh()

    def _refresh(self, *_):
        query, filters, rings = self._search.text(), self._filters(), self._rings()
        modes = self._effective_modes()
        counts = self._catalog.facet_counts(query, filters, rings, modes)
        for facet, buttons in self._facet_buttons.items():
            # An option that would empty an AND row stays clickable when the
            # row as OR would give it results, so Alt+click can switch to OR.
            or_counts = None
            for value, (btn, label) in buttons.items():
                n = counts[facet][value]
                btn.setText(f"{label}  {n}")
                if n == 0 and not btn.isChecked() and modes[facet] == "and":
                    if or_counts is None:
                        or_counts = self._catalog.facet_counts(
                            query, filters, rings, {**modes, facet: "or"})[facet]
                    n = or_counts[value]
                # A checked button stays enabled so it can always be unchecked.
                btn.setEnabled(n > 0 or btn.isChecked())

        hits = self._catalog.search(query, filters, rings, modes, self._sort.currentData())
        current = self.selected_record["id"] if self.selected_record else None
        self._list.blockSignals(True)
        self._list.clear()
        right_aligned = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        keep_row = 0
        for row, rec in enumerate(hits):
            item = QTreeWidgetItem([rec["name"], _pretty_formula(rec["formula"]),
                                    f"{self._catalog.mass(rec):.2f}"])
            item.setTextAlignment(2, right_aligned)
            item.setData(0, Qt.ItemDataRole.UserRole, rec["id"])
            self._list.addTopLevelItem(item)
            if rec["id"] == current and query == getattr(self, "_last_query", None):
                keep_row = row
        self._list.blockSignals(False)
        self._last_query = query
        total = len(self._catalog.records())
        self._count.setText(f"{len(hits)} of {total} molecules")
        if hits:
            self._list.setCurrentItem(self._list.topLevelItem(keep_row))   # currentItemChanged shows it
        else:
            self._show_record(None)

    def _on_current_changed(self, current, _previous):
        rec = self._catalog.get(current.data(0, Qt.ItemDataRole.UserRole)) if current else None
        self._show_record(rec)

    def _show_record(self, rec):
        self.selected_record = rec
        self._buttons.button(QDialogButtonBox.StandardButton.Open).setEnabled(rec is not None)
        if rec is None:
            self._preview.load(b"")
            self._info.setText("")
            return
        try:
            self._preview.load(self._preview_svg(rec))
        except Exception as e:
            _log_error(f"Catalog preview failed for {rec['id']}", e)
            self._preview.load(b"")
        from html import escape
        n_heavy = sum(1 for a in rec["atoms"] if a[0] != "H")
        lines = [f"<b>{escape(rec['name'])}</b> — {rec['formula']}"]
        if rec.get("synonyms"):
            lines.append("<i>" + escape(", ".join(rec["synonyms"])) + "</i>")
        lines.append(f"{len(rec['atoms'])} atoms ({n_heavy} heavy), {rec['rings']} rings, "
                     f"charge {rec['charge']}, multiplicity {rec['multiplicity']}")
        order = {facet: k for k, (facet, _, _) in enumerate(self._catalog.facets())}
        tags = sorted((t for t in rec["tags"] if not t.startswith("atom:")),   # the formula shows those
                      key=lambda t: order.get(t.partition(":")[0], len(order)))
        known = {f"{facet}:{v}" for facet, _, options in self._catalog.facets() for v, _, _ in options}
        labels = []
        for t in tags:
            label = self._catalog.tag_label(t)
            if t in known and label not in labels:   # only tags a filter row offers, once each
                labels.append(label)
        lines.append(" · ".join(labels))
        src = rec["geometry_source"]
        if src.startswith("pubchem-3d:"):
            src = f"PubChem 3D conformer (CID {src.split(':', 1)[1]})"
        else:
            src = {"rdkit-etkdg-mmff94": "RDKit ETKDG + MMFF94",
                   "rdkit-etkdg-uff": "RDKit ETKDG + UFF",
                   "rdkit-etkdg": "RDKit ETKDG (no force field)",
                   "rdkit-spectral-uff": "Spectral embedding + UFF",
                   "fixed": "Fixed experimental geometry"}.get(src, src)
        lines.append(f"Geometry: {src}")
        smiles = rec["smiles"] if len(rec["smiles"]) <= 48 else rec["smiles"][:47] + "…"
        lines.append(f"<span style='font-size:10px'>{smiles}</span>")
        # Detection in space: one line per reported detection, linked to its paper.
        for det in rec.get("detections", []):
            status = "" if det["status"] == "confirmed" else f" ({det['status']})"
            title = escape(det.get("paper_title") or "detection paper")
            link = f"<a href='{escape(det['paper_url'])}'>{title}</a>" if det.get("paper_url") else title
            ref = escape(det.get("paper_ref") or "")
            lines.append(f"<span style='font-size:11px'>Detected {det['year']}{status}: {link}"
                         f"{'<br>' + ref if ref else ''}</span>")
        for note in rec.get("notes", []):
            lines.append(f"<i style='font-size:11px'>{escape(note)}</i>")
        self._info.setText("<br>".join(lines))

    def _preview_svg(self, rec) -> bytes:
        if self._view3d is not None:
            return self._preview_svg_3d(rec)
        return self._preview_svg_2d(rec)

    def _preview_svg_3d(self, rec) -> bytes:
        c = self._view3d
        mol = self._catalog.to_molecule(rec)
        centered = np.array([[a.x, a.y, a.z] for a in mol.atoms])
        centered -= centered.mean(axis=0)
        # Same fit and default (face-on) view as MoleculeCanvas.load_molecule.
        max_extent = float(np.max(np.linalg.norm(centered, axis=1))) + 1.5
        scale = 0.40 * min(self.PREVIEW_W, self.PREVIEW_H) / max_extent
        m_normal = np.linalg.svd(centered)[2][-1] if len(centered) >= 3 else None
        svg = render_molecule(
            mol, rot_matrix_override=np.eye(3),
            canvas_w=self.PREVIEW_W, canvas_h=self.PREVIEW_H, scale=scale,
            atom_scale=c.atom_scale,
            # Bond width is in screen pixels: shrink it with the preview.
            bond_width_px=c.bond_width_px * min(1.0, scale / max(c.base_scale, 1e-6)),
            bond_style=c.bond_style, bond_color=c.bond_color,
            background=c.background, color_overrides=c.color_overrides or None,
            atom_border_mode=c.atom_border_mode, atom_border_scale=c.atom_border_scale,
            atom_border_width=c.atom_border_width, lighting_intensity=c.lighting_intensity,
            light_position=c.light_position, roughness=c.roughness,
            m_normal_override=m_normal, output_path=None, export_mode=True,
        )
        return svg.encode("utf-8") if isinstance(svg, str) else svg

    def _preview_svg_2d(self, rec) -> bytes:
        from molvector.chemvector import generate_2d_layout, render_chemvector, DOUBLE_BOND_GAP_PX
        mol = self._catalog.to_molecule(rec)
        # A flat depiction of a cage is unreadable: always show the 3D projection.
        projection = self._layout == "match3d" or "topology:cage" in rec["tags"]
        rot = np.eye(3)
        normal = best_fit_plane_normal(mol)
        if projection and normal is not None:
            rot = face_plane_rotation(normal, rot)   # as MainWindow._face_molecule_plane
        positions, bond_info = generate_2d_layout(mol, view_rot=rot, projection=projection)
        heavy = [positions[i] for i, a in enumerate(mol.atoms) if a.element != "H" and i in positions]
        pts = heavy or list(positions.values())
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        span_x = max(xs) - min(xs) + 2.4
        span_y = max(ys) - min(ys) + 2.4
        # Line widths and letters are sized for a 50 px typical bond, as in
        # ChemCanvas: its median bond (heavy-atom bonds first), so H2's
        # 0.74 A bond gets letters to match, not ones sized for C-C.
        ref_len = 1.5
        for heavy_only in (True, False):
            lens = [math.dist(positions[b.i], positions[b.j]) for b in mol.bonds
                    if b.i in positions and b.j in positions
                    and not (heavy_only and "H" in (mol.atoms[b.i].element, mol.atoms[b.j].element))]
            lens = [d for d in lens if d > 1e-6]
            if lens:
                ref_len = float(np.median(lens))
                break
        # Fit the canvas, but never draw a typical bond longer than ~63 px.
        scale = min(self.PREVIEW_W / span_x, self.PREVIEW_H / span_y, 63.0 / ref_len)
        f = scale * ref_len / 50.0
        svg = render_chemvector(
            mol, positions, bond_info,
            canvas_w=self.PREVIEW_W, canvas_h=self.PREVIEW_H, scale=scale,
            center=((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0),
            bond_width_px=max(1.0, 3.4 * f), font_size_px=max(8.0, 26.0 * f),
            double_bond_gap_px=DOUBLE_BOND_GAP_PX * f, export_mode=True,
        )
        return svg.encode("utf-8")


class _DragWidget(QPushButton):
    """A draggable pushbutton that initiates a file drag on mouse move."""

    def __init__(self, get_svg_path_cb, parent=None):
        super().__init__("Drag SVG", parent)
        self._get_svg_path = get_svg_path_cb
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setToolTip("Drag this into an SVG editor")
        self.setStyleSheet("""
            QPushButton {
                border: 2px dashed #888;
                border-radius: 8px;
                font-size: 14px;
                background: #f0f0f0;
                padding: 8px;
            }
            QPushButton:hover {
                border-color: #4a9eff;
                background: #e0edff;
            }
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if (event.position().toPoint() - self._drag_start).manhattanLength() < QApplication.startDragDistance():
            return

        svg_path = self._get_svg_path()

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(svg_path)])

        with open(svg_path, "rb") as f:
            mime.setData("image/svg+xml", f.read())

        drag = QDrag(self)
        drag.setMimeData(mime)

        pixmap = self.grab().scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        drag.setPixmap(pixmap)
        drag.setHotSpot(self.rect().center())

        drag.exec(Qt.DropAction.CopyAction)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE CANVAS
# ─────────────────────────────────────────────────────────────────────────────

def _ideal_directions(n: int, angle_rad: float):
    """Return n unit vectors in ideal VSEPR-like geometry (tetrahedral, trigonal, bent)."""
    if n == 4:
        a = 1.0 / math.sqrt(3.0)
        return [np.array([a,a,a]), np.array([a,-a,-a]),
                np.array([-a,a,-a]), np.array([-a,-a,a])]
    elif n == 3:
        s3 = math.sqrt(3.0)/2.0
        return [np.array([1.0,0.0,0.0]), np.array([-0.5,s3,0.0]), np.array([-0.5,-s3,0.0])]
    elif n == 2:
        half = angle_rad / 2.0
        return [np.array([math.sin(half),0.0,math.cos(half)]),
                np.array([-math.sin(half),0.0,math.cos(half)])]
    dirs = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        phi = math.acos(2.0 * i / n - 1.0)
        dirs.append(np.array([math.sin(phi)*math.cos(theta),
                              math.sin(phi)*math.sin(theta), math.cos(phi)]))
    return dirs


class MoleculeCanvas(QSvgWidget):
    rotationChanged = pyqtSignal()
    fileDropped = pyqtSignal(str)
    moleculeChanged = pyqtSignal() # Emitted when atoms/bonds change
    requestHistorySave = pyqtSignal() # Emitted BEFORE a change

    def __init__(self, parent=None):
        super().__init__(parent)
        self.molecule: Molecule | None = None
        self._rot  = np.eye(3)
        self._zoom = 1.0
        self._pan  = np.array([0.0, 0.0])
        self._axes_ref = np.eye(3)
        self._default_view = np.eye(3)
        self._drag_start: QPoint | None = None
        self._drag_mode = "none"
        self._prev_pinch_value: float | None = None
        self._last_tap_time: float = 0.0

        self.setMinimumSize(500, 450)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._show_vectors = False
        self.build_mode = False
        self.selection_mode = False
        self.align_mode = False
        self.paint_mode = False
        self.measure_mode = False
        self._measure_atoms: list[list[int]] = []
        self._measure_from: int | None = None
        self._angle_atoms: list[list[int]] = []
        self._angle_leg1: tuple[int, int] | None = None

        self._paint_color = "#e63946"
        self.build_element = "C"
        self._cached_m_normal: np.ndarray | None = None
        self._fast_render = False
        # Fast Render Mode also applies while zoom is actively changing (wheel,
        # trackpad pinch, or the toolbar slider), not just while dragging —
        # flagged for a short settle window after the last zoom step, after
        # which one final full-quality render is triggered.
        self._zooming = False
        self._zoom_settle_timer = QTimer(self)
        self._zoom_settle_timer.setSingleShot(True)
        self._zoom_settle_timer.timeout.connect(self._on_zoom_settled)
        self._cached_svg_prefix = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=4)
        )
        self._cached_grad_stops: dict = {}
        self._bonding_from: int | None = None
        self._mouse_pos: QPoint | None = None

        # Selection state
        self.selected_atoms: set = set()
        self._green_atoms: set = set()
        self._sel_drag_start: QPoint | None = None
        self._sel_rect: QRectF | None = None

        # Atom dragging (Alt/Ctrl+click in build/select mode)
        self._drag_atom_idx: int | None = None
        self._align_axis: np.ndarray | None = None  # bond axis for align rotate-around-bond drag

        # Render parameters — all public, set by MainWindow
        self.base_scale     = 110.0
        self.atom_scale     = 0.7
        self.bond_width_px  = 10.0
        self.background     = "#ffffff"
        self.bond_style     = "splitted"
        self.bond_color     = "#444444"
        self.atom_border_mode  = "scaled"
        self.atom_border_scale = 1.04
        self.atom_border_width = 2.0
        self.color_overrides: dict = {}
        self.lighting_intensity: float = 1.0
        self.light_position: str = "Top left"
        self.roughness: float = 1.0
        self.show_axes: bool = False
        self.show_principal_axes: bool = False
        self.axes_position: str = "bottom-left"
        self.principal_axes_position: str = "bottom-right"
        self.animation_phase: float = 0.0
        self.animation_amplitude: float = 0.0

        self._render_timer = QTimer()
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._do_render)

        self.arrows: Optional[List[Tuple[int, float, float, float, str]]] = None
        self.active_vectors: Optional[np.ndarray] = None
        self.animation_phase: float = 0.0
        self.animation_amplitude: float = 0.0
        self.mulliken_legend: Optional[Tuple[float, float, str, str, str]] = None
        self.mulliken_legend_in_export: bool = False

        self.dipole_arrow_3d: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
        self.show_dipole_arrow: bool = False
        
        # Build options
        self.build_mode: bool = False
        self.build_element: str = "C"
        self.auto_adjust_h: bool = False
        self.bond_order_disabled: bool = False
        self._saved_bond_orders: dict[tuple[int, int], int] | None = None

    # ── drag and drop ─────────────────────────────────────────────────────────

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
            path = urls[0].toLocalFile()
            self.fileDropped.emit(path)
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    # ── public ───────────────────────────────────────────────────────────────

    def load_molecule(self, mol: Molecule):
        self.molecule = mol
        self._rot  = np.eye(3)
        self._zoom = 1.0
        self._pan  = np.array([0.0, 0.0])
        self._axes_ref = np.eye(3)
        self._default_view = np.eye(3)
        self.selected_atoms.clear()
        self._cached_grad_stops.clear()
        self.dipole_arrow_3d = None
        # Auto-scale: fit the molecule to 80% of the smaller canvas dimension
        positions = np.array([[a.x, a.y, a.z] for a in mol.atoms])
        if len(positions) > 0:
            centroid = positions.mean(axis=0)
            centered = positions - centroid
            max_extent = np.max(np.linalg.norm(centered, axis=1)) + 1.5  # +1.5 Å for atom radius margin
            if max_extent > 0:
                w = max(self.width(),  500)
                h = max(self.height(), 450)
                self.base_scale = 0.40 * min(w, h) / max_extent
            # Cache SVD normal for bond offset computation (doesn't change with rotation)
            if len(centered) >= 3:
                _, _, Vt = np.linalg.svd(centered)
                self._cached_m_normal = Vt[-1]
            else:
                self._cached_m_normal = None
        else:
            self._cached_m_normal = None
        self.request_render()

    def reset_view(self):
        self._rot  = self._default_view.copy()
        self._zoom = 1.0
        self._pan  = np.array([0.0, 0.0])
        self.request_render()

    def reset_xyz_axes(self):
        self._axes_ref = self._rot.copy()
        self._default_view = self._rot.copy()
        self.request_render()

    def set_principal_axis_preset(self, axis_index):
        import mol_strudel as strudel
        from molvector.render import ATOMIC_MASSES
        mol = self.molecule
        if not mol or len(mol.atoms) < 2:
            return
        coords = np.array([[a.x, a.y, a.z] for a in mol.atoms])
        masses = np.array([ATOMIC_MASSES.get(a.element, 0.0) for a in mol.atoms])
        _, eigvecs = strudel.diagonalize_I_tensor(coords, masses)
        Vt = eigvecs.T
        if axis_index == 0:
            cx, sx = math.cos(-math.pi/2), math.sin(-math.pi/2)
            Ry = np.array([[cx,0,sx],[0,1,0],[-sx,0,cx]])
            self._rot = Ry @ Vt
        elif axis_index == 1:
            cx, sx = math.cos(math.pi/2), math.sin(math.pi/2)
            Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
            self._rot = Rx @ Vt
        else:
            self._rot = Vt
        self._zoom = 1.0
        self._pan  = np.array([0.0, 0.0])
        self.request_render()

    def set_coordinate_plane_preset(self, plane_index):
        from molvector.render import rotation_matrix
        if plane_index == 0:    # XY plane (view from +Z)
            self._rot = rotation_matrix(math.radians(90), 0, 0)
        elif plane_index == 1:  # XZ plane (view from +Y)
            self._rot = rotation_matrix(0, math.radians(90), 0)
        else:                   # YZ plane (view from +X)
            self._rot = rotation_matrix(0, 0, math.radians(90))
        self._zoom = 1.0
        self._pan  = np.array([0.0, 0.0])
        self.request_render()

    def edit_background_color(self):
        col = QColorDialog.getColor(QColor(self.background), self, "Background Colour")
        if col.isValid():
            self.background = col.name()
            self.request_render()

    def request_render(self, delay_ms: int = 0):
        self._render_timer.start(delay_ms)

    def note_zoom_activity(self):
        """Flag zoom as actively changing (for Fast Render Mode) and (re)start the settle timer."""
        self._zooming = True
        self._zoom_settle_timer.start(220)

    def _on_zoom_settled(self):
        self._zooming = False
        self.request_render()

    def get_svg_bytes(self, export_mode: bool = False) -> bytes:
        return self._render_to_bytes(export_mode=export_mode)

    # ── internal ─────────────────────────────────────────────────────────────

    def _render_to_bytes(self, w=None, h=None, export_mode: bool = False) -> bytes:
        if self.molecule is None:
            return b""
        w = w or max(self.width(),  500)
        h = h or max(self.height(), 450)

        result = render_molecule(
            self.molecule,
            rot_matrix_override=self._rot,
            pan_x=self._pan[0], pan_y=self._pan[1],
            canvas_w=w, canvas_h=h,
            scale=self.base_scale * self._zoom,
            atom_scale=self.atom_scale,
            bond_width_px=self.bond_width_px,
            bond_style=self.bond_style,
            bond_color=self.bond_color,
            background=self.background,
            color_overrides=self.color_overrides or None,
            atom_border_mode=self.atom_border_mode,
            atom_border_scale=self.atom_border_scale,
            atom_border_width=self.atom_border_width,
            lighting_intensity=self.lighting_intensity,
            light_position=self.light_position,
            roughness=self.roughness,
            show_axes=self.show_axes,
            show_principal_axes=self.show_principal_axes,
            axes_position=self.axes_position,
            principal_axes_position=self.principal_axes_position,
            axes_rot_override=self._rot @ np.linalg.inv(self._axes_ref),
            m_normal_override=self._cached_m_normal,
            svg_prefix=self._cached_svg_prefix,
            cached_grad_stops=self._cached_grad_stops,
            active_vectors=self.active_vectors,
            animation_phase=self.animation_phase,
            animation_amplitude=self.animation_amplitude,
            output_path=None,
            export_mode=export_mode,
            fast_render=self._fast_render and (self._drag_start is not None or self._zooming) and not export_mode,
            selected_indices=self.selected_atoms if not export_mode else None,
            vectors=self.arrows if self._show_vectors else None,
            mulliken_legend=self.mulliken_legend,
            mulliken_legend_in_export=self.mulliken_legend_in_export,
        )
        return result.encode("utf-8") if isinstance(result, str) else result

    def _do_render(self):
        data = self._render_to_bytes()
        if data:
            self.load(QByteArray(data))
        # Overlay for bonding line
        if self.build_mode and self._bonding_from is not None and self._mouse_pos is not None:
            self.update() # Force paintEvent for overlay

    def paintEvent(self, event):
        super().paintEvent(event)
        draw_sel = self.selection_mode and self._sel_rect is not None and self._sel_rect.isValid()
        draw_bond = self.build_mode and self._bonding_from is not None and self._mouse_pos is not None and self.molecule
        draw_measure = self.measure_mode and self._measure_atoms and self.molecule
        draw_measure_drag = self.measure_mode and self._measure_from is not None and self._mouse_pos is not None and self.molecule
        draw_angles = self.measure_mode and (self._angle_atoms or self._angle_leg1 is not None) and self.molecule
        draw_green = self.align_mode and self._green_atoms and self.molecule
        draw_dipole = self.show_dipole_arrow and self.dipole_arrow_3d is not None and self.molecule
        if not draw_sel and not draw_bond and not draw_measure and not draw_measure_drag and not draw_angles and not draw_green and not draw_dipole:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        if draw_sel:
            p.fillRect(self._sel_rect, QColor(0, 120, 255, 30))
            p.setPen(QColor(0, 140, 255, 200))
            p.drawRect(self._sel_rect.toRect())

        if draw_bond:
            p.setPen(QColor("#007bff"))
            atoms, _ = project_molecule(
                self.molecule, self._rot, self._pan[0], self._pan[1],
                self.width(), self.height(), self.base_scale * self._zoom, self.atom_scale
            )
            ax, ay, az, ar = atoms[self._bonding_from]
            p.drawLine(int(ax), int(ay), self._mouse_pos.x(), self._mouse_pos.y())

        if draw_measure:
            atoms, _ = project_molecule(
                self.molecule, self._rot, self._pan[0], self._pan[1],
                self.width(), self.height(), self.base_scale * self._zoom, self.atom_scale
            )
            font = p.font()
            font.setPointSize(10)
            font.setBold(True)
            p.setFont(font)
            for pair in self._measure_atoms:
                i1, i2 = pair[0], pair[1]
                x1, y1 = atoms[i1][0], atoms[i1][1]
                x2, y2 = atoms[i2][0], atoms[i2][1]
                a1 = self.molecule.atoms[i1]
                a2 = self.molecule.atoms[i2]
                dist = math.sqrt((a2.x - a1.x)**2 + (a2.y - a1.y)**2 + (a2.z - a1.z)**2)
                p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.SolidLine))
                p.drawLine(int(x1), int(y1), int(x2), int(y2))
                p.setPen(QColor("red"))
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                p.drawText(int(mx) + 6, int(my) - 6, f"{dist:.2f} Å")

        if draw_measure_drag:
            atoms, _ = project_molecule(
                self.molecule, self._rot, self._pan[0], self._pan[1],
                self.width(), self.height(), self.base_scale * self._zoom, self.atom_scale
            )
            ax, ay, az, ar = atoms[self._measure_from]
            p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.SolidLine))
            p.setBrush(QColor(255, 0, 0, 60))
            p.drawEllipse(QPointF(ax, ay), ar + 4, ar + 4)
            p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(int(ax), int(ay), self._mouse_pos.x(), self._mouse_pos.y())

        if draw_angles:
            atoms, _ = project_molecule(
                self.molecule, self._rot, self._pan[0], self._pan[1],
                self.width(), self.height(), self.base_scale * self._zoom, self.atom_scale
            )
            font = p.font()
            font.setPointSize(10)
            font.setBold(True)
            p.setFont(font)

            def _draw_angle_arc(painter, pts_2d, pts_3d, color):
                x1, y1 = pts_2d[0]
                vx, vy = pts_2d[1]
                x3, y3 = pts_2d[2]
                a1, a2, a3 = pts_3d[0], pts_3d[1], pts_3d[2]
                v1 = np.array([a1.x - a2.x, a1.y - a2.y, a1.z - a2.z])
                v2 = np.array([a3.x - a2.x, a3.y - a2.y, a3.z - a2.z])
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 < 1e-9 or norm2 < 1e-9:
                    return
                cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
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
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 60))
                painter.drawPath(fill_path)
                outline_path = QPainterPath()
                start_x = vx + arc_radius * math.cos(math.radians(angle1))
                start_y = vy - arc_radius * math.sin(math.radians(angle1))
                outline_path.moveTo(start_x, start_y)
                outline_path.arcTo(rect, angle1, span)
                painter.setPen(QPen(color, 2, Qt.PenStyle.SolidLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(outline_path)
                mid_angle_rad = math.radians(angle1 + span / 2)
                label_r = arc_radius + 14
                lx = vx + label_r * math.cos(mid_angle_rad)
                ly = vy - label_r * math.sin(mid_angle_rad)
                painter.setPen(color)
                painter.drawText(int(lx), int(ly), f"{angle_deg:.1f}\u00b0")

            for triple in self._angle_atoms:
                i1, i2, i3 = triple
                x1, y1 = atoms[i1][0], atoms[i1][1]
                vx, vy = atoms[i2][0], atoms[i2][1]
                x3, y3 = atoms[i3][0], atoms[i3][1]
                pts_2d = [(x1, y1), (vx, vy), (x3, y3)]
                pts_3d = [self.molecule.atoms[i] for i in (i1, i2, i3)]
                p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.SolidLine))
                p.drawLine(int(vx), int(vy), int(x1), int(y1))
                p.drawLine(int(vx), int(vy), int(x3), int(y3))
                _draw_angle_arc(p, pts_2d, pts_3d, QColor("red"))

            if self._angle_leg1 is not None:
                i1, i2 = self._angle_leg1
                x1, y1 = atoms[i1][0], atoms[i1][1]
                x2, y2 = atoms[i2][0], atoms[i2][1]
                p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.SolidLine))
                p.drawLine(int(x1), int(y1), int(x2), int(y2))
                for idx in (i1, i2):
                    ax, ay, az, ar = atoms[idx]
                    p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.SolidLine))
                    p.setBrush(QColor(255, 0, 0, 60))
                    p.drawEllipse(QPointF(ax, ay), ar + 4, ar + 4)
                if self._mouse_pos is not None:
                    p.setPen(QPen(QColor("red"), 2, Qt.PenStyle.DashLine))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawLine(int(x2), int(y2), self._mouse_pos.x(), self._mouse_pos.y())

        if draw_green:
            atoms, _ = project_molecule(
                self.molecule, self._rot, self._pan[0], self._pan[1],
                self.width(), self.height(), self.base_scale * self._zoom, self.atom_scale
            )
            for idx in self._green_atoms:
                if idx >= len(atoms):
                    continue
                ax, ay, az, ar = atoms[idx]
                p.setPen(QPen(QColor(0, 200, 0), 2, Qt.PenStyle.SolidLine))
                p.setBrush(QColor(0, 200, 0, 60))
                p.drawEllipse(QPointF(ax, ay), ar + 4, ar + 4)

        if draw_dipole:
            start_3d, end_3d, mu_mag = self.dipole_arrow_3d
            # Project start and end points to 2D (same logic as render_molecule)
            centered = center_positions(self.molecule.atoms)
            cx = self.width() / 2 + self._pan[0]
            cy = self.height() / 2 + self._pan[1]
            CAMERA_Z = 60.0
            s = self.base_scale * self._zoom

            def _proj(pt3d):
                rp = self._rot @ pt3d
                zf = CAMERA_Z / (CAMERA_Z - rp[2]) if (CAMERA_Z - rp[2]) != 0 else 1.0
                return cx + rp[0] * s * zf, cy - rp[1] * s * zf

            sx, sy = _proj(start_3d)
            ex, ey = _proj(end_3d)

            # Draw thick purple line (stops at arrowhead base)
            dx, dy = ex - sx, ey - sy
            L = math.hypot(dx, dy)
            if L > 1e-3:
                ux, uy = dx / L, dy / L
            else:
                ux, uy = 0.0, 0.0
            ah_len = 14.0
            bx = ex - ux * ah_len
            by = ey - uy * ah_len
            dipole_pen = QPen(QColor("#9933ff"), 4, Qt.PenStyle.SolidLine)
            dipole_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(dipole_pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(int(sx), int(sy), int(bx), int(by))

            # Draw large arrowhead (constant size, tip at very end)
            if L > 1e-3:
                ah_hw = 8.0
                px, py = -uy, ux
                p1 = (ex, ey)
                p2 = (ex - ux * ah_len + px * ah_hw, ey - uy * ah_len + py * ah_hw)
                p3 = (ex - ux * ah_len - px * ah_hw, ey - uy * ah_len - py * ah_hw)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor("#9933ff"))
                path = QPainterPath()
                path.moveTo(*p1)
                path.lineTo(*p2)
                path.lineTo(*p3)
                path.closeSubpath()
                p.drawPath(path)

            # Label with magnitude
            label = f"\u03bc = {mu_mag:.3f} D"
            font = p.font()
            font.setPointSize(12)
            font.setBold(True)
            p.setFont(font)
            p.setPen(QColor("#9933ff"))
            lx = (sx + ex) / 2 + 10
            ly = (sy + ey) / 2 - 10
            p.drawText(int(lx), int(ly), label)

        p.end()

    # ── mouse ─────────────────────────────────────────────────────────────────

    def _get_hit_atom(self, pos: QPoint) -> int | None:
        if not self.molecule: return None
        atoms, _ = project_molecule(
            self.molecule, self._rot, self._pan[0], self._pan[1],
            self.width(), self.height(), self.base_scale * self._zoom, self.atom_scale
        )
        for i, (ax, ay, az, ar) in enumerate(atoms):
            dist = math.hypot(pos.x() - ax, pos.y() - ay)
            if dist < max(ar, 10):
                return i
        return None

    def _get_hit_bond(self, pos: QPoint) -> int | None:
        if not self.molecule: return None
        _, bonds = project_molecule(
            self.molecule, self._rot, self._pan[0], self._pan[1],
            self.width(), self.height(), self.base_scale * self._zoom, self.atom_scale
        )
        threshold = 8.0
        for ax, ay, bx, by, z_avg, idx in bonds:
            # Distance from point to segment
            L2 = (bx-ax)**2 + (by-ay)**2
            if L2 < 1e-6: continue
            t = max(0, min(1, ((pos.x()-ax)*(bx-ax) + (pos.y()-ay)*(by-ay)) / L2))
            proj_x = ax + t * (bx-ax)
            proj_y = ay + t * (by-ay)
            dist = math.hypot(pos.x() - proj_x, pos.y() - proj_y)
            if dist < threshold:
                return idx
        return None

    def _unproject(self, pos: QPoint) -> np.ndarray:
        # returns 3D position in molecule local frame (relative to centroid)
        cx = self.width()/2 + self._pan[0]
        cy = self.height()/2 + self._pan[1]
        scale = self.base_scale * self._zoom
        
        # Assume z_local = 0
        rp_x = (pos.x() - cx) / scale
        rp_y = (cy - pos.y()) / scale
        rp = np.array([rp_x, rp_y, 0.0])
        # local = rot^-1 @ rp
        local = np.linalg.inv(self._rot) @ rp
        return local

    def mousePressEvent(self, event):
        self._update_cursor(event.position().toPoint(), event.modifiers())
        # Alt+click (see alt_click) in align mode toggles green highlight
        mod = event.modifiers()
        if self.align_mode and event.button() == Qt.MouseButton.LeftButton and alt_click(mod):
            idx = self._get_hit_atom(event.position().toPoint())
            if idx is not None and self.molecule:
                if idx in self._green_atoms:
                    self._green_atoms.discard(idx)
                else:
                    self._green_atoms.add(idx)
                self.update()
                return

        # Alt+click (see alt_click) drags an atom in build mode
        if self.build_mode and event.button() == Qt.MouseButton.LeftButton and alt_click(mod):
            idx = self._get_hit_atom(event.position().toPoint())
            if idx is not None and self.molecule:
                self.requestHistorySave.emit()
                self._drag_atom_idx = idx
                self._drag_start = event.position().toPoint()
                self._drag_mode = "atom"
                self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
                return

        if self.paint_mode and event.button() == Qt.MouseButton.LeftButton:
            idx = self._get_hit_atom(event.position().toPoint())
            if idx is not None and self.molecule:
                self.requestHistorySave.emit()
                mod = event.modifiers()
                if mod & Qt.KeyboardModifier.ShiftModifier:
                    self.molecule.atoms[idx].color = None
                else:
                    self.molecule.atoms[idx].color = self._paint_color
                self.moleculeChanged.emit()
                self.request_render()
                return

        if self.selection_mode and event.button() == Qt.MouseButton.LeftButton:
            mod = event.modifiers()
            idx = self._get_hit_atom(event.position().toPoint())
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
            self.selected_atoms.clear()
            self._sel_drag_start = event.position().toPoint()
            self._sel_rect = QRectF()
            self.request_render()
            return

        if self.measure_mode and event.button() == Qt.MouseButton.LeftButton:
            idx = self._get_hit_atom(event.position().toPoint())
            if idx is not None and self.molecule:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    if self._angle_leg1 is None and self._measure_from is None:
                        self._measure_from = idx
                        self._mouse_pos = event.position().toPoint()
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
                self._mouse_pos = event.position().toPoint()
                self.update()
                return
            self._measure_from = None
            self._mouse_pos = None
            self.update()
            return

        if self.build_mode and event.button() == Qt.MouseButton.LeftButton:
            idx = self._get_hit_atom(event.position().toPoint())
            if idx is not None:
                self._bonding_from = idx
                self._mouse_pos = event.position().toPoint()
                return
            
            b_idx = self._get_hit_bond(event.position().toPoint())
            if b_idx is not None:
                self.requestHistorySave.emit()
                if self.bond_order_disabled:
                    self.molecule.bonds[b_idx].order = 1
                else:
                    self.molecule.bonds[b_idx].order = (self.molecule.bonds[b_idx].order % 3) + 1
                
                if self.auto_adjust_h:
                    at1, at2 = self.molecule.atoms[self.molecule.bonds[b_idx].i], self.molecule.atoms[self.molecule.bonds[b_idx].j]
                    self._apply_auto_h(at1)
                    self._apply_auto_h(at2)
                    self._relax_h()
                
                self.moleculeChanged.emit()
                self.request_render()
                return

            # Add new atom if nothing hit
            if not self.molecule:
                self.molecule = Molecule("New Molecule", atoms=[])
            
            self.requestHistorySave.emit()
            loc = self._unproject(event.position().toPoint())
            # Convert local to absolute (if we have a centroid)
            if self.molecule.atoms:
                pos_arr = np.array([[a.x, a.y, a.z] for a in self.molecule.atoms])
                centroid = pos_arr.mean(axis=0)
                abs_pos = loc + centroid
            else:
                abs_pos = loc
            
            new_atom = Atom(self.build_element, abs_pos[0], abs_pos[1], abs_pos[2])
            self.molecule.atoms.append(new_atom)
            new_idx = len(self.molecule.atoms) - 1
            
            if self.auto_adjust_h:
                self._apply_auto_h(self.molecule.atoms[new_idx])
                self._relax_h()
                
            self.moleculeChanged.emit()
            self.request_render()
            return

        if self.build_mode and event.button() == Qt.MouseButton.RightButton:
            idx = self._get_hit_atom(event.position().toPoint())
            if idx is not None:
                self.requestHistorySave.emit()
                # 1. Identify connected H atoms to remove
                h_to_remove = []
                neighbors_to_adjust = []
                
                for b in self.molecule.bonds:
                    if b.i == idx:
                        nb_idx = b.j
                    elif b.j == idx:
                        nb_idx = b.i
                    else:
                        continue
                    
                    nb_atom = self.molecule.atoms[nb_idx]
                    if nb_atom.element == "H":
                        # Only remove if it has no other bonds
                        other_bonds = 0
                        for b2 in self.molecule.bonds:
                            if b2.i == nb_idx or b2.j == nb_idx: other_bonds += 1
                        if other_bonds == 1:
                            h_to_remove.append(nb_idx)
                    else:
                        neighbors_to_adjust.append(nb_atom)

                # 2. Delete the atoms (highest index first to avoid shifts)
                all_to_del = sorted(list(set(h_to_remove + [idx])), reverse=True)
                for d_idx in all_to_del:
                    self.molecule.atoms.pop(d_idx)
                    new_bonds = []
                    for b in self.molecule.bonds:
                        if b.i == d_idx or b.j == d_idx: continue
                        ni = b.i - 1 if b.i > d_idx else b.i
                        nj = b.j - 1 if b.j > d_idx else b.j
                        new_bonds.append(Bond(ni, nj, b.order))
                    self.molecule.bonds = new_bonds

                # 3. Adjust neighbors if auto_adjust_h is on
                if self.auto_adjust_h:
                    for nb_atom in neighbors_to_adjust:
                        self._apply_auto_h(nb_atom)
                    self._relax_h()

                self.moleculeChanged.emit()
                self.request_render()
                return
            
            b_idx = self._get_hit_bond(event.position().toPoint())
            if b_idx is not None:
                self.requestHistorySave.emit()
                # Remove bond
                self.molecule.bonds.pop(b_idx)
                self.moleculeChanged.emit()
                self.request_render()
                return

        if self.align_mode and event.button() == Qt.MouseButton.LeftButton:
            b_idx = self._get_hit_bond(event.position().toPoint())
            if b_idx is not None and self.molecule:
                bond = self.molecule.bonds[b_idx]
                a1 = self.molecule.atoms[bond.i]
                a2 = self.molecule.atoms[bond.j]
                v_local = np.array([a2.x - a1.x, a2.y - a1.y, a2.z - a1.z])
                norm = np.linalg.norm(v_local)
                if norm > 1e-10:
                    v_local_hat = v_local / norm
                    v_view = self._rot @ v_local_hat
                    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        target = np.array([1.0, 0.0, 0.0])
                    else:
                        target = np.array([0.0, 1.0, 0.0])
                    dot = np.dot(v_view, target)
                    # Pick the closer direction to avoid large flips
                    if dot < 0:
                        target = -target
                        dot = -dot
                    if dot < 0.9999:
                        sin_theta = np.linalg.norm(np.cross(v_view, target))
                        if sin_theta > 1e-10:
                            cos_theta = dot
                            axis = np.cross(v_view, target) / sin_theta
                            K = np.array([[0, -axis[2], axis[1]],
                                          [axis[2], 0, -axis[0]],
                                          [-axis[1], axis[0], 0]])
                            R_corr = np.eye(3) + sin_theta * K + (1.0 - cos_theta) * (K @ K)
                            self._rot = R_corr @ self._rot
                            self.rotationChanged.emit()
                            self.request_render()
            return

        # Shift+click on bond in neutral mode: rotate around that bond
        mod = event.modifiers()
        if (not self._any_mode_active()) and event.button() == Qt.MouseButton.LeftButton and (mod & Qt.KeyboardModifier.ShiftModifier):
            b_idx = self._get_hit_bond(event.position().toPoint())
            if b_idx is not None and self.molecule:
                bond = self.molecule.bonds[b_idx]
                a1 = self.molecule.atoms[bond.i]
                a2 = self.molecule.atoms[bond.j]
                v_local = np.array([a2.x - a1.x, a2.y - a1.y, a2.z - a1.z])
                n = np.linalg.norm(v_local)
                if n > 1e-10:
                    self._align_axis = v_local / n
                    self._drag_start = event.position().toPoint()
                    self._drag_mode = "rotate_around_bond"
                    self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                    return

        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            self._drag_mode  = "rotate"
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        elif event.button() == Qt.MouseButton.RightButton:
            self._drag_start = event.position().toPoint()
            self._drag_mode  = "pan"
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))

    def _update_cursor(self, pos: QPoint, mods):
        has_atom = self._get_hit_atom(pos) is not None
        alt_held = alt_click(mods)

        if self.build_mode:
            if alt_held and has_atom:
                self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        elif self.selection_mode:
            if alt_held and has_atom:
                self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            elif has_atom:
                self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        elif self.align_mode:
            if self._get_hit_bond(pos) is not None:
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        elif self.paint_mode:
            if has_atom:
                self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        elif self.measure_mode:
            if has_atom:
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def mouseMoveEvent(self, event):
        self._mouse_pos = event.position().toPoint()
        self._update_cursor(event.position().toPoint(), event.modifiers())
        if self.build_mode and self._bonding_from is not None:
            self.update()
            return

        if self.measure_mode and self._measure_from is not None:
            self._mouse_pos = event.position().toPoint()
            self.update()
            return

        if self.measure_mode and self._angle_leg1 is not None:
            self._mouse_pos = event.position().toPoint()
            self.update()
            return

        if self.selection_mode and self._sel_drag_start is not None:
            cur = event.position().toPoint()
            self._sel_rect = QRectF(QPointF(self._sel_drag_start), QPointF(cur)).normalized()
            self.update()
            return

        if self._drag_mode == "atom" and self._drag_atom_idx is not None and self.molecule:
            cur = event.position().toPoint()
            dx = cur.x() - self._drag_start.x()
            dy = cur.y() - self._drag_start.y()
            self._drag_start = cur
            scale = self.base_scale * self._zoom
            inv_rot = np.linalg.inv(self._rot)
            delta_3d = inv_rot @ np.array([dx / scale, -dy / scale, 0.0])
            # Move clicked atom
            atom = self.molecule.atoms[self._drag_atom_idx]
            atom.x += delta_3d[0]
            atom.y += delta_3d[1]
            atom.z += delta_3d[2]
            # Move all other selected atoms together
            if self._drag_atom_idx in self.selected_atoms:
                for i in self.selected_atoms:
                    if i == self._drag_atom_idx:
                        continue
                    a = self.molecule.atoms[i]
                    a.x += delta_3d[0]
                    a.y += delta_3d[1]
                    a.z += delta_3d[2]
            self.request_render(delay_ms=0)
            return

        if self._drag_start is None or self.molecule is None:
            return
        cur = event.position().toPoint()
        dx  = cur.x() - self._drag_start.x()
        dy  = cur.y() - self._drag_start.y()
        self._drag_start = cur

        if self._drag_mode == "rotate":
            sens = 0.008
            ax, ay = dy*sens, dx*sens
            cx,sx  = math.cos(ax), math.sin(ax)
            cy,sy  = math.cos(ay), math.sin(ay)
            Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
            Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
            self._rot = Rx @ Ry @ self._rot
        elif self._drag_mode == "pan":
            self._pan[0] += dx
            self._pan[1] += dy
        elif self._drag_mode == "rotate_around_bond" and self._align_axis is not None:
            sens = 0.008
            ax, ay, _ = self._rot @ self._align_axis
            bond_2d = np.array([ax, -ay])
            bond_len = np.linalg.norm(bond_2d)
            if bond_len < 1e-6:
                angle = dx * sens
            else:
                bond_2d /= bond_len
                perp = np.array([-bond_2d[1], bond_2d[0]])
                angle = (dx * perp[0] + dy * perp[1]) * sens
            if abs(angle) > 1e-10:
                axis = self._rot @ self._align_axis
                ca = math.cos(angle); sa = math.sin(angle)
                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
                R = np.eye(3) + sa * K + (1.0 - ca) * (K @ K)
                self._rot = R @ self._rot

        self.rotationChanged.emit()
        self.request_render(delay_ms=0)

    def _any_mode_active(self):
        return self.build_mode or self.selection_mode or self.align_mode or self.paint_mode or self.measure_mode

    def mouseDoubleClickEvent(self, event):
        if self.molecule is None or self._any_mode_active():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()
            self.rotationChanged.emit()
            self.request_render(delay_ms=0)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._update_cursor(event.position().toPoint(), event.modifiers())
        if self.selection_mode and self._sel_drag_start is not None:
            final_pt = event.position().toPoint()
            rect = QRectF(QPointF(self._sel_drag_start), QPointF(final_pt)).normalized()
            if rect.width() > 5 and rect.height() > 5 and self.molecule:
                atoms, _ = project_molecule(
                    self.molecule, self._rot, self._pan[0], self._pan[1],
                    self.width(), self.height(), self.base_scale * self._zoom, self.atom_scale
                )
                for i, (ax, ay, az, ar) in enumerate(atoms):
                    if rect.contains(ax, ay):
                        self.selected_atoms.add(i)
                self.request_render()
            self._sel_drag_start = None
            self._sel_rect = None
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.update()
            return

        if self.measure_mode and self._measure_from is not None:
            end_idx = self._get_hit_atom(event.position().toPoint())
            if end_idx is not None and end_idx != self._measure_from and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self._measure_atoms.append([self._measure_from, end_idx])
                self._measure_from = None
                self._mouse_pos = None
            self.update()
            return

        if self.build_mode and self._bonding_from is not None:
            end_idx = self._get_hit_atom(event.position().toPoint())
            if end_idx is not None and end_idx == self._bonding_from:
                self.requestHistorySave.emit()
                self.molecule.atoms[self._bonding_from].element = self.build_element
                if self.auto_adjust_h:
                    self._apply_auto_h(self.molecule.atoms[self._bonding_from])
                    self._relax_h()
                self.moleculeChanged.emit()
            elif end_idx is not None and end_idx != self._bonding_from:
                self.requestHistorySave.emit()
                # Bond existing
                self.molecule.bonds.append(Bond(self._bonding_from, end_idx))
                
                if self.auto_adjust_h:
                    at1, at2 = self.molecule.atoms[self._bonding_from], self.molecule.atoms[end_idx]
                    self._apply_auto_h(at1)
                    self._apply_auto_h(at2)
                    self._relax_h()
                
                self.moleculeChanged.emit()
            elif end_idx is None:
                self.requestHistorySave.emit()
                # Add new atom and bond
                loc = self._unproject(event.position().toPoint())
                pos_arr = np.array([[a.x, a.y, a.z] for a in self.molecule.atoms])
                centroid = pos_arr.mean(axis=0)
                abs_pos = loc + centroid
                new_idx = len(self.molecule.atoms)
                self.molecule.atoms.append(Atom(self.build_element, abs_pos[0], abs_pos[1], abs_pos[2]))
                self.molecule.bonds.append(Bond(self._bonding_from, new_idx))
                
                if self.auto_adjust_h:
                    at1, at2 = self.molecule.atoms[self._bonding_from], self.molecule.atoms[new_idx]
                    self._apply_auto_h(at1)
                    self._apply_auto_h(at2)
                    self._relax_h()
                    
                self.moleculeChanged.emit()
            
            self._bonding_from = None
            self._mouse_pos = None
            self.request_render()
            return

        was_atom_drag = self._drag_mode == "atom"
        self._drag_atom_idx = None
        self._drag_start = None
        self._drag_mode  = "none"
        if was_atom_drag:
            if self.build_mode or self.selection_mode:
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        elif self.align_mode:
            self._update_cursor(event.position().toPoint(), Qt.KeyboardModifier.NoModifier)
        else:
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.request_render(delay_ms=0)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.SelectAll):
            if self.molecule:
                self.selected_atoms = set(range(len(self.molecule.atoms)))
                self.request_render()
            return
        if event.key() == Qt.Key.Key_Escape:
            if self.measure_mode and (self._measure_atoms or self._measure_from is not None or self._angle_atoms or self._angle_leg1 is not None):
                self._measure_atoms.clear()
                self._measure_from = None
                self._mouse_pos = None
                self._angle_atoms.clear()
                self._angle_leg1 = None
                self.update()
            elif self.selected_atoms or self._green_atoms:
                self.selected_atoms.clear()
                self._green_atoms.clear()
                self.update()
                self.request_render()
            return
        mod = event.modifiers()
        if event.key() == Qt.Key.Key_A and (mod & Qt.KeyboardModifier.ShiftModifier) and (mod & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)):
            if self.selected_atoms or self._green_atoms:
                self.selected_atoms.clear()
                self._green_atoms.clear()
                self.update()
                self.request_render()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self._mouse_pos and self._get_hit_atom(self._mouse_pos) is not None:
            self._update_cursor(self._mouse_pos, event.modifiers())
        super().keyReleaseEvent(event)

    def _apply_auto_h(self, atom: Atom):
        valencies = {
            "H": 1, "C": 4, "N": 3, "O": 2, "F": 1,
            "P": 3, "S": 2, "Cl": 1, "Br": 1, "I": 1,
            "Si": 4, "B": 3
        }
        # Ideal X-H bond lengths per atom
        bond_lengths = {"H":0.74,"C":1.09,"N":1.01,"O":0.96,"F":0.92,
                        "P":1.42,"S":1.34,"Cl":1.27,"Br":1.41,"I":1.61,
                        "Si":1.48,"B":1.19}
        try:
            atom_idx = -1
            for i, a in enumerate(self.molecule.atoms):
                if a is atom:
                    atom_idx = i
                    break
            if atom_idx == -1: return
        except Exception: return

        v = valencies.get(atom.element, 0)
        if v <= 0: return

        # Collect existing neighbor indices and bond orders
        neighbors = []
        neighbor_orders = {}
        for b in self.molecule.bonds:
            if b.i == atom_idx:
                neighbors.append(b.j)
                neighbor_orders[b.j] = b.order
            elif b.j == atom_idx:
                neighbors.append(b.i)
                neighbor_orders[b.i] = b.order

        # Separate current H (only bonded to this atom) from heavy neighbors
        h_indices = []
        heavy_indices = []
        heavy_order_sum = 0
        for nb_idx in set(neighbors):
            if self.molecule.atoms[nb_idx].element == "H":
                other_bonds = sum(1 for b in self.molecule.bonds if b.i == nb_idx or b.j == nb_idx)
                if other_bonds == 1:
                    h_indices.append(nb_idx)
                else:
                    heavy_indices.append(nb_idx)
                    heavy_order_sum += neighbor_orders.get(nb_idx, 1)
            else:
                heavy_indices.append(nb_idx)
                heavy_order_sum += neighbor_orders.get(nb_idx, 1)

        needed = max(0, v - heavy_order_sum)

        # Remove excess H atoms
        if len(h_indices) > needed:
            to_del = sorted(h_indices[needed:], reverse=True)
            for idx in to_del:
                self.molecule.atoms.pop(idx)
                new_bonds = []
                for b in self.molecule.bonds:
                    if b.i == idx or b.j == idx: continue
                    ni = b.i - 1 if b.i > idx else b.i
                    nj = b.j - 1 if b.j > idx else b.j
                    new_bonds.append(Bond(ni, nj, b.order))
                self.molecule.bonds = new_bonds
                if atom_idx > idx: atom_idx -= 1
            return

        if len(h_indices) >= needed:
            return

        # Need to add (needed - len(h_indices)) H atoms
        n_add = needed - len(h_indices)

        # Build existing bond direction vectors (from atom to heavy neighbors)
        atom_pos = np.array([atom.x, atom.y, atom.z])
        existing_vecs = []
        for nb_idx in heavy_indices:
            nb = self.molecule.atoms[nb_idx]
            d = np.array([nb.x - atom.x, nb.y - atom.y, nb.z - atom.z])
            nd = np.linalg.norm(d)
            if nd > 1e-4:
                existing_vecs.append(d / nd)
            else:
                existing_vecs.append(np.array([1.0, 0.0, 0.0]))

        bl = bond_lengths.get(atom.element, 1.09)
        steric = len(heavy_indices) + needed  # total bonded atoms (including H to add)
        if atom.element == "C":
            angle_by_s = {4: 109.47, 3: 120.0, 2: 180.0}
        elif atom.element == "O":
            angle_by_s = {4: 109.47, 3: 120.0, 2: 104.5, 1: 180.0}
        elif atom.element == "N":
            angle_by_s = {4: 109.47, 3: 120.0, 2: 104.5, 1: 180.0}
        else:
            angle_by_s = {4: 109.47, 3: 120.0, 2: 104.5, 1: 180.0}
        angle_deg = angle_by_s.get(steric, 109.47)
        angle_rad = math.radians(angle_deg)

        # Generate ideal geometry directions for the full steric number
        ideal_dirs = _ideal_directions(steric, angle_rad)

        # Match existing bond directions to closest ideal directions
        used = set()
        h_dirs = []
        for vec in existing_vecs:
            best = -1
            best_dot = -2
            for j, d in enumerate(ideal_dirs):
                if j in used: continue
                dot = np.dot(vec, d)
                if dot > best_dot:
                    best_dot = dot
                    best = j
            used.add(best)

        for j in range(len(ideal_dirs)):
            if j not in used:
                h_dirs.append(ideal_dirs[j])

        # Add H atoms in the available ideal directions
        for i in range(min(n_add, len(h_dirs))):
            h_idx = len(self.molecule.atoms)
            off = h_dirs[i] * bl
            self.molecule.atoms.append(Atom("H", atom.x + off[0], atom.y + off[1], atom.z + off[2]))
            self.molecule.bonds.append(Bond(atom_idx, h_idx))
        # If more H needed than available ideal slots (edge case), place at uniform offsets
        for i in range(n_add - min(n_add, len(h_dirs))):
            h_idx = len(self.molecule.atoms)
            off = np.array([1.0, 0.0, 0.0])
            self.molecule.atoms.append(Atom("H", atom.x + off[0], atom.y + off[1], atom.z + off[2]))
            self.molecule.bonds.append(Bond(atom_idx, h_idx))

    def _relax_h(self):
        """Geometrically re-place all H atoms (no FF solver needed)."""
        if not self.molecule:
            return
        parents_to_relax = []
        for atom_idx, atom in enumerate(self.molecule.atoms):
            if atom.element == "H":
                continue
            has_h = False
            for b in self.molecule.bonds:
                other = b.j if b.i == atom_idx else (b.i if b.j == atom_idx else -1)
                if 0 <= other < len(self.molecule.atoms) and self.molecule.atoms[other].element == "H":
                    has_h = True
                    break
            if has_h:
                parents_to_relax.append(atom)
        for atom in parents_to_relax:
            if atom in self.molecule.atoms:
                self._apply_auto_h(atom)

    def _get_non_h_indices(self) -> List[int]:
        if not self.molecule: return []
        return [i for i, a in enumerate(self.molecule.atoms) if a.element != "H"]

    def gestureEvent(self, event):
        if event.type() == QEvent.Type.NativeGesture and self.molecule is not None:
            if event.gestureType() == Qt.GestureType.PinchGesture:
                val = event.value()
                if self._prev_pinch_value is not None and self._prev_pinch_value != 0.0:
                    factor = val / self._prev_pinch_value
                    self._zoom = max(0.15, min(6.0, self._zoom * factor))
                    self.rotationChanged.emit()
                    self.note_zoom_activity()
                    self.request_render(delay_ms=0)
                self._prev_pinch_value = val
                return True
            if event.gestureType() == Qt.GestureType.TapGesture and not self._any_mode_active():
                now = time.monotonic()
                if now - self._last_tap_time < 0.4:
                    self._last_tap_time = 0.0
                    self.reset_view()
                    self.rotationChanged.emit()
                    self.request_render(delay_ms=0)
                    return True
                self._last_tap_time = now
        self._prev_pinch_value = None
        return super().gestureEvent(event)

    def wheelEvent(self, event):
        if self.molecule is None:
            return
        pixel = event.pixelDelta()
        angle_y = event.angleDelta().y()
        if not pixel.isNull():
            # pixelDelta can be inverted by macOS natural scrolling;
            # use angleDelta for direction, pixelDelta for smooth magnitude
            sign = 1 if angle_y >= 0 else -1
            factor = 1.0 + sign * (abs(pixel.y()) / 120.0) * 1.5
        elif angle_y != 0:
            factor = 1.12 if angle_y > 0 else (1 / 1.12)
        else:
            return
        self._zoom = max(0.15, min(6.0, self._zoom * factor))
        self.rotationChanged.emit()
        self.note_zoom_activity()
        self.request_render(delay_ms=0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.request_render(delay_ms=0)


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATION WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class NotificationWidget(QWidget):
    """A single toast notification that auto-dismisses and can be closed."""

    closed = pyqtSignal(object)

    def __init__(self, message: str, notification_type: str = "info",
                 duration: int = 4000, parent=None,
                 button_text: str = None, button_callback=None):
        super().__init__(parent)
        self.notification_type = notification_type
        self.duration = duration
        self.setFixedWidth(320)
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        t = THEMES[parent._main_window._current_theme] if parent and hasattr(parent, '_main_window') else THEMES["dark"]

        accent_colors = {
            "info": t['ACCENT'], "success": "#44aa66",
            "warning": "#ccaa44", "error": "#cc4444"
        }
        self._accent = accent_colors.get(notification_type, t['ACCENT'])
        self._bg = "#2a2a3a" if parent._main_window._current_theme == "dark" else "#e8eaed"
        self._border = t['BORDER']
        self._fg = t['FG']
        self._fg_dim = t['FG_DIM']

        self.setStyleSheet(
            f"NotificationWidget {{ background:{self._bg}; "
            f"border:1px solid {self._border}; border-radius:8px; }}"
        )

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 12, 12, 12)
        main_layout.setSpacing(0)

        # Left accent bar
        accent_bar = QFrame()
        accent_bar.setFixedWidth(4)
        accent_bar.setStyleSheet(f"background:{self._accent}; border:none; border-radius:2px;")
        main_layout.addWidget(accent_bar)

        # Inner content (vertical: message on top, optional button row below)
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 0, 0, 0)
        content_layout.setSpacing(6)

        # Message
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet(f"color:{self._fg}; background:transparent; border:none; font-size:12px;")
        content_layout.addWidget(msg_label)

        # Optional button row
        if button_text and button_callback:
            btn_row = QHBoxLayout()
            btn_row.setContentsMargins(0, 0, 0, 0)
            btn_row.addStretch()
            open_btn = QPushButton(button_text)
            open_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            open_btn.setStyleSheet(
                f"QPushButton {{ background:{self._accent}; color:#ffffff; border:none; "
                f"border-radius:4px; font-size:11px; padding:3px 10px; }}"
                f"QPushButton:hover {{ background:#8b0000; }}"
            )
            open_btn.clicked.connect(button_callback)
            btn_row.addWidget(open_btn)
            content_layout.addLayout(btn_row)

        # Close button (top-right corner)
        close_btn = QPushButton("×")
        close_btn.setFixedSize(20, 20)
        close_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        close_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{self._fg_dim}; border:none; "
            f"border-radius:10px; font-size:14px; font-weight:bold; padding:0; }}"
            f"QPushButton:hover {{ background:{self._border}; color:{self._fg}; }}"
        )
        close_btn.clicked.connect(self.close_notification)

        # Wrap content + close button in a horizontal layout
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(0)
        top_row.addLayout(content_layout, 1)
        top_row.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignTop)

        main_layout.addLayout(top_row)

        # Fade
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)

        # Auto-dismiss timer
        self._fade_timer = QTimer(self)
        self._fade_timer.setSingleShot(True)
        self._fade_timer.timeout.connect(self._start_fade_out)
        self._suppress_timer_restart = False

    def sizeHint(self):
        hint = super().sizeHint()
        if hint.height() < self.minimumHeight():
            hint.setHeight(self.minimumHeight())
        return hint

    def showEvent(self, event):
        super().showEvent(event)
        if self.duration > 0 and not self._suppress_timer_restart:
            self._fade_timer.start(self.duration)

    def extend_timer(self, extra_ms=1000):
        remaining = self._fade_timer.remainingTime()
        if remaining > 0:
            self._fade_timer.start(remaining + extra_ms)

    def _start_fade_out(self):
        from PyQt6.QtCore import (
            QPropertyAnimation, QEasingCurve,
        )
        anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        anim.setDuration(250)
        anim.setStartValue(self._opacity_effect.opacity())
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(self._on_fade_done)
        self._fade_anim = anim
        anim.start()

    def _on_fade_done(self):
        self.closed.emit(self)

    def close_notification(self):
        self._fade_timer.stop()
        self._start_fade_out()


class NotificationContainer(QWidget):
    """Overlay widget that holds and manages stacked notifications in the top-right corner."""

    def __init__(self, main_window: 'MainWindow'):
        super().__init__(main_window)
        self._main_window = main_window
        self._notifications = []
        self._removal_queue = []
        self._slide_anim = None
        self._spacing = 10
        self._margin = 10

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background:transparent;")

    def _update_size(self):
        h = self._margin * 2
        max_w = 0
        for i, note in enumerate(self._notifications):
            h += note.sizeHint().height()
            max_w = max(max_w, note.minimumWidth())
            if i > 0:
                h += self._spacing
        self.setFixedWidth(max_w + self._margin * 2)
        self.setFixedHeight(h)

    def _reposition(self):
        mb_height = self._main_window.menuBar().height() if self._main_window.menuBar().isVisible() else 30
        tb_height = 0
        for toolbar in self._main_window.findChildren(QToolBar):
            if toolbar.isVisible():
                tb_height = max(tb_height, toolbar.y() + toolbar.height() - mb_height)
        top = mb_height + max(tb_height, 4) + 8
        mw = self._main_window.width()
        self.move(mw - self.width() - 5, top)
        self.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition()

    def add_notification(self, message: str, notification_type: str = "info", duration: int = 4000,
                          button_text: str = None, button_callback=None):
        note = NotificationWidget(message, notification_type, duration, parent=self,
                                  button_text=button_text, button_callback=button_callback)
        note.closed.connect(self._remove_notification)
        self._notifications.append(note)
        positions = self._compute_positions()
        note.move(positions[id(note)])
        self._update_size()
        self._reposition()
        self.show()
        self.raise_()
        note.show()

    def _remove_notification(self, note):
        if note not in self._notifications:
            return
        # If a slide-up animation is already running, queue this removal
        if self._slide_anim is not None and self._slide_anim.state() == self._slide_anim.State.Running:
            if note not in self._removal_queue:
                self._removal_queue.append(note)
            return

        old_positions = {id(n): n.pos() for n in self._notifications if n is not note}
        self._notifications.remove(note)

        # Hide remaining widgets BEFORE removal so the visual
        # reflow doesn't cause a visible flash at new positions
        for n in self._notifications:
            n.hide()

        note.setParent(None)
        note.blockSignals(True)
        note.hide()
        note.deleteLater()

        if not self._notifications:
            self._update_size()
            self.hide()
        else:
            for n in self._notifications:
                old = old_positions.get(id(n))
                if old:
                    n._suppress_timer_restart = True
                    n.move(old)
                    n.show()
                    n._suppress_timer_restart = False
                    n.extend_timer(1000)
            new_positions = self._compute_positions(old_positions)
            self._animate_slide_up(old_positions, new_positions)

    def _process_removal_queue(self):
        while self._removal_queue:
            note = self._removal_queue.pop(0)
            if note in self._notifications:
                self._remove_notification(note)
                break  # _remove_notification may re-queue if another animation starts

    def _compute_positions(self, old_positions=None):
        positions = {}
        y = self._margin
        for n in self._notifications:
            x = old_positions[id(n)].x() if old_positions and id(n) in old_positions else n.pos().x()
            positions[id(n)] = QPoint(x, y)
            y += n.sizeHint().height() + self._spacing
        return positions

    def _animate_slide_up(self, old_positions, new_positions):
        from PyQt6.QtCore import (
            QPropertyAnimation, QParallelAnimationGroup, QEasingCurve,
        )
        group = QParallelAnimationGroup(self)
        for note in self._notifications:
            old_pos = old_positions.get(id(note))
            new_pos = new_positions.get(id(note))
            if old_pos and new_pos and old_pos != new_pos:
                anim = QPropertyAnimation(note, b"pos")
                anim.setDuration(200)
                anim.setStartValue(old_pos)
                anim.setEndValue(new_pos)
                anim.setEasingCurve(QEasingCurve.Type.OutCubic)
                group.addAnimation(anim)
        if group.animationCount() > 0:
            group.finished.connect(self._finish_slide)
            self._slide_anim = group
            group.start()
        else:
            self._finish_slide()

    def _finish_slide(self):
        self._slide_anim = None
        self._process_removal_queue()
        self._update_size()
        self._reposition()
        if not self._notifications:
            self.hide()

    def clear_all(self):
        for note in self._notifications[:]:
            note.close_notification()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    DEFAULT_SHORTCUTS = {
        "open": _mod("Ctrl+O"),
        "open_catalog": _mod("Ctrl+Shift+O"),
        "save_as": _mod("Ctrl+S"),
        "export_svg": _mod("Ctrl+Shift+S"),
        "export_view": _mod("Ctrl+E"),
        "quick_svg_export": _mod("Ctrl+Shift+X"),
        "quit": _mod("Ctrl+Q"),
        "settings": _mod("Ctrl+;"),
        "style": _mod("Ctrl+="),
        "shortcuts": "",
        "info": _mod("Ctrl+I"),
        "charge": "",
        "inchi": "",
        "selection_mode": "S",
        "build_mode": "B",
        "align_mode": "A",
        "measure_mode": "M",
        "reset_view": "R",
        "reset_xyz_axes": "Shift+R",
        "preset_xy_plane": "",
        "preset_xz_plane": "",
        "preset_yz_plane": "",
        "preset_bc_plane": "1",
        "preset_ac_plane": "2",
        "preset_ab_plane": "3",
        "overlay_settings": "",
        "g16_input": "",
        "rotational_constants": "",
        "dipole_moment": "",
        "terminal": "",
        "calc_results": _mod("Ctrl+M"),
        "clean_molecule": _mod("Ctrl+L"),
        "undo": _mod("Ctrl+Z"),
        "redo": _mod("Ctrl+Shift+Z"),
        "optimize_settings": "",
        "bond_order": "",
        "open_test": _mod("Ctrl+Shift+T"),
        "open_test_folder": "",
        "about": "",
    }

    ALT_SHORTCUT_DEFAULTS = {
        "redo": _mod("Ctrl+Y"),
    }

    LAST_MOLECULE_FILE = paths.config_path("last_molecule.json")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Molvector — Molecule Viewer")
        self.resize(1080, 720)
        self._current_theme = "light"
        self._apply_theme("light")
        self._current_source = "None"
        self._current_path   = ""
        self._color_overrides: dict = {}   # elem -> hex
        
        # Animation timer
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(40)  # ~25 FPS
        self._anim_timer.timeout.connect(self._on_anim_step)
        self._anim_phase = 0.0
        
        # FF Parameters
        self._ff_max_steps = 500
        self._optimize_worker = None
        self._ff_tol = 0.01

        # Undo/Redo history
        self._history = [] # list of deepcopied Molecule
        self._redo_stack = []
        self._max_history = 50

        # Toolbar icon size for mode buttons
        self._mode_icon_size = QSize(22, 22)

        self._shortcut_actions: dict = {}
        self._terminal_dialog: Optional['TerminalDialog'] = None
        self._show_debug_menu = False
        self._debug_menu = None

        self._build_central()
        self._notification_container = NotificationContainer(self)
        self._notification_container.hide()
        self._build_menubar()
        self._build_toolbar()
        # Tool-specific toolbars (Build/Select/Align/Measure and their mode
        # options) always start their own row below the main toolbar, rather
        # than packing onto its end when the window is wide enough.
        self.addToolBarBreak()
        self._setup_builder_toolbar()
        self._build_statusbar()
        self._load_appearance_config()
        self._update_debug_menu()
        self._apply_shortcut_config()
        try:
            self._maybe_restore_molecule()
        except Exception as e:
            # Never let a restored session keep the app from starting.
            log_path = _log_error("Session restore failed", e)
            self._canvas.molecule = None
            QTimer.singleShot(500, lambda: self.show_notification(
                "The last session could not be restored.", "warning", log_file=log_path))
        if getattr(self, '_start_view_pref', '3D') == '2D':
            self._toggle_chem_mode(True)
        self._show_placeholder()
        self.setAcceptDrops(True)

    # ── menu bar ─────────────────────────────────────────────────────────────

    def _build_menubar(self):
        mb = self.menuBar()

        # ── File ──
        file_menu = mb.addMenu("&File")

        act_open = QAction("&Open…", self)
        act_open.setShortcut(_mod("Ctrl+O"))
        act_open.triggered.connect(self._open_file)
        file_menu.addAction(act_open)
        self._shortcut_actions["open"] = act_open

        act_open_catalog = QAction("Open from &Catalog…", self)
        act_open_catalog.setShortcut(_mod("Ctrl+Shift+O"))
        act_open_catalog.triggered.connect(self._open_from_catalog)
        file_menu.addAction(act_open_catalog)
        self._shortcut_actions["open_catalog"] = act_open_catalog

        act_save_as = QAction("Save &As…", self)
        act_save_as.setShortcut(_mod("Ctrl+S"))
        act_save_as.triggered.connect(self._save_as)
        file_menu.addAction(act_save_as)
        self._shortcut_actions["save_as"] = act_save_as

        file_menu.addSeparator()

        act_save_svg = QAction("Export as &SVG…", self)
        act_save_svg.setShortcut(_mod("Ctrl+Shift+S"))
        act_save_svg.triggered.connect(self._save_svg)
        file_menu.addAction(act_save_svg)
        self._shortcut_actions["export_svg"] = act_save_svg

        act_export = QAction("&Export View…", self)
        act_export.setShortcut(_mod("Ctrl+E"))
        act_export.triggered.connect(self._export_view)
        file_menu.addAction(act_export)
        self._shortcut_actions["export_view"] = act_export

        act_quick_svg = QAction("Quick SVG E&xport…", self)
        act_quick_svg.setShortcut(_mod("Ctrl+Shift+X"))
        act_quick_svg.triggered.connect(self._quick_svg_export)
        file_menu.addAction(act_quick_svg)
        self._shortcut_actions["quick_svg_export"] = act_quick_svg

        file_menu.addSeparator()

        act_restart = QAction("&Restart", self)
        act_restart.setShortcut(_mod("Ctrl+Alt+Q"))
        act_restart.triggered.connect(self._restart)
        file_menu.addAction(act_restart)

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(_mod("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)
        self._shortcut_actions["quit"] = act_quit

        # ── Edit ──
        edit_menu = mb.addMenu("&Edit")

        act_settings = QAction("&Settings…", self)
        act_settings.setShortcut(_mod("Ctrl+;"))
        act_settings.triggered.connect(self._edit_settings)
        edit_menu.addAction(act_settings)
        self._shortcut_actions["settings"] = act_settings

        act_style = QAction("&Style…", self)
        act_style.setShortcut(_mod("Ctrl+="))
        act_style.triggered.connect(self._edit_style)
        edit_menu.addAction(act_style)
        self._shortcut_actions["style"] = act_style

        act_shortcuts = QAction("&Shortcuts…", self)
        act_shortcuts.triggered.connect(lambda: self._edit_settings(tab_index=1))
        edit_menu.addAction(act_shortcuts)
        self._shortcut_actions["shortcuts"] = act_shortcuts

        edit_menu.addSeparator()

        act_info = QAction("&Info…", self)
        act_info.setShortcut(_mod("Ctrl+I"))
        act_info.triggered.connect(self._show_molecule_info)
        edit_menu.addAction(act_info)
        self._shortcut_actions["info"] = act_info

        act_charge = QAction("Edit &Charge…", self)
        act_charge.triggered.connect(self._edit_charge)
        edit_menu.addAction(act_charge)
        self._shortcut_actions["charge"] = act_charge

        act_inchi = QAction("&InChI…", self)
        act_inchi.triggered.connect(self._show_inchi)
        edit_menu.addAction(act_inchi)
        self._shortcut_actions["inchi"] = act_inchi

        edit_menu.addSeparator()

        act_select_toggle = QAction("Selection Mode", self)
        act_select_toggle.setCheckable(True)
        act_select_toggle.setShortcut("S")
        act_select_toggle.triggered.connect(self._toggle_selection_mode)
        edit_menu.addAction(act_select_toggle)
        self._act_select_toggle = act_select_toggle
        self._shortcut_actions["selection_mode"] = act_select_toggle

        act_toggle = QAction("Build Mode", self)
        act_toggle.setCheckable(True)
        act_toggle.setShortcut("B")
        act_toggle.triggered.connect(self._toggle_build_mode)
        edit_menu.addAction(act_toggle)
        self._act_build_toggle = act_toggle
        self._shortcut_actions["build_mode"] = act_toggle

        act_align_toggle = QAction("Align Mode", self)
        act_align_toggle.setCheckable(True)
        act_align_toggle.setShortcut("A")
        act_align_toggle.triggered.connect(self._toggle_align_mode)
        edit_menu.addAction(act_align_toggle)
        self._act_align_toggle = act_align_toggle
        self._shortcut_actions["align_mode"] = act_align_toggle

        act_measure_toggle = QAction("Measure Mode", self)
        act_measure_toggle.setCheckable(True)
        act_measure_toggle.setShortcut("M")
        act_measure_toggle.triggered.connect(self._toggle_measure_mode)
        edit_menu.addAction(act_measure_toggle)
        self._act_measure_toggle = act_measure_toggle
        self._shortcut_actions["measure_mode"] = act_measure_toggle

        # ── View ──
        view_menu = mb.addMenu("&View")

        act_reset_view = QAction("&Reset View", self)
        act_reset_view.setShortcut("R")
        act_reset_view.triggered.connect(self._reset_view)
        view_menu.addAction(act_reset_view)
        self._shortcut_actions["reset_view"] = act_reset_view

        act_reset_xyz = QAction("Set XYZ &Axes", self)
        act_reset_xyz.setShortcut("Shift+R")
        act_reset_xyz.triggered.connect(lambda: self._canvas.reset_xyz_axes())
        view_menu.addAction(act_reset_xyz)
        self._shortcut_actions["reset_xyz_axes"] = act_reset_xyz

        view_menu.addSeparator()

        presets_menu = view_menu.addMenu("&Preset Orientation")

        presets_menu.addSeparator()
        for idx, label in [(0, "XY plane"), (1, "XZ plane"), (2, "YZ plane")]:
            a = QAction(label, self)
            a.triggered.connect(lambda _, i=idx: self._canvas.set_coordinate_plane_preset(i))
            presets_menu.addAction(a)
            key = f"preset_{label.lower().replace(' ', '_')}"
            self._shortcut_actions[key] = a

        presets_menu.addSeparator()
        for idx, label in [(0, "BC plane"), (1, "AC plane"), (2, "AB plane")]:
            a = QAction(label, self)
            a.setShortcut(str(idx + 1))
            a.triggered.connect(lambda _, i=idx: self._canvas.set_principal_axis_preset(i))
            presets_menu.addAction(a)
            key = f"preset_{label.lower().replace(' ', '_')}"
            self._shortcut_actions[key] = a

        view_menu.addSeparator()
        act_overlay = QAction("Overlay Settings", self)
        act_overlay.triggered.connect(self._edit_overlay)
        view_menu.addAction(act_overlay)
        self._shortcut_actions["overlay_settings"] = act_overlay

        # ── Calculations ──
        self._menu_calc = mb.addMenu("&Calculations")
        act_g16 = QAction("Generate G16 Input…", self)
        act_g16.triggered.connect(self._generate_g16_input)
        self._menu_calc.addAction(act_g16)
        self._shortcut_actions["g16_input"] = act_g16
        self._menu_calc.setEnabled(True)

        self._menu_calc.addSeparator()
        act_rotconstant = QAction("Calculate Rotational Constants", self)
        act_rotconstant.triggered.connect(self._calculate_rotational_constants)
        self._menu_calc.addAction(act_rotconstant)
        self._shortcut_actions["rotational_constants"] = act_rotconstant

        act_dipole = QAction("Calculate Dipole Moment", self)
        act_dipole.triggered.connect(self._calculate_dipole_moment)
        self._menu_calc.addAction(act_dipole)
        self._shortcut_actions["dipole_moment"] = act_dipole

        self._menu_calc.addSeparator()
        act_terminal = QAction("Open Terminal", self)
        act_terminal.triggered.connect(self._open_terminal)
        self._menu_calc.addAction(act_terminal)
        self._shortcut_actions["terminal"] = act_terminal

        # ── Build ──
        self._menu_build = mb.addMenu("&Build")
        
        act_clean_m = QAction("Clean Molecule", self)
        act_clean_m.setShortcut(_mod("Ctrl+L"))
        act_clean_m.triggered.connect(self._clean_molecule)
        self._menu_build.addAction(act_clean_m)
        self._shortcut_actions["clean_molecule"] = act_clean_m

        act_undo = QAction("Undo", self)
        act_undo.setShortcut(_mod("Ctrl+Z"))
        act_undo.triggered.connect(self._undo)
        self._menu_build.addAction(act_undo)
        self._shortcut_actions["undo"] = act_undo

        act_redo = QAction("Redo", self)
        act_redo.setShortcuts([QKeySequence(_mod("Ctrl+Shift+Z")), QKeySequence(_mod("Ctrl+Y"))])
        act_redo.triggered.connect(self._redo)
        self._menu_build.addAction(act_redo)
        self._shortcut_actions["redo"] = act_redo

        act_ff = QAction("Optimize Settings…", self)
        act_ff.triggered.connect(self._edit_ff_settings)
        self._menu_build.addAction(act_ff)
        self._shortcut_actions["optimize_settings"] = act_ff

        self._menu_build.addSeparator()

        self._act_bond_order = QAction("Disable Bond Order", self)
        self._act_bond_order.setCheckable(True)
        self._act_bond_order.triggered.connect(self._toggle_bond_order)
        self._menu_build.addAction(self._act_bond_order)
        self._shortcut_actions["bond_order"] = self._act_bond_order

        act_paint = QAction("Paint Selected Atoms…", self)
        act_paint.triggered.connect(self._paint_selected_atoms)
        self._menu_build.addAction(act_paint)
        self._shortcut_actions["paint_atoms"] = act_paint

        act_clear_paint = QAction("Clear Paint", self)
        act_clear_paint.triggered.connect(self._clear_paint)
        self._menu_build.addAction(act_clear_paint)
        self._shortcut_actions["clear_paint"] = act_clear_paint

        self._menu_build.addSeparator()

        # Same dialog as File → Open from Catalog… (which holds the shortcut).
        act_build_catalog = QAction("Catalog…", self)
        act_build_catalog.triggered.connect(self._open_from_catalog)
        self._menu_build.addAction(act_build_catalog)

        # ── 2D Viewer (Chemvector) ──
        chem_menu = mb.addMenu("2D &Viewer")

        act_chem_mode = QAction("&Chem View (2D Whiteboard)", self)
        act_chem_mode.setCheckable(True)
        act_chem_mode.setShortcut(_mod("Ctrl+2"))
        act_chem_mode.setToolTip("Switch between the 3D viewer and the 2D Chemvector whiteboard")
        act_chem_mode.triggered.connect(self._toggle_chem_mode)
        chem_menu.addAction(act_chem_mode)
        self._act_chem_mode = act_chem_mode
        self._shortcut_actions["chem_mode"] = act_chem_mode

        # Build/Selection Mode are shared with the 3D view (same QActions as
        # in Edit menu / toolbar) — each applies to whichever canvas is
        # currently on screen.
        self._act_build_toggle.setToolTip(
            "3D: click atom add/bond, drag to bond, click bond change order. "
            "2D: click empty space to add an atom, drag from an atom to bond it, "
            "click a bond to cycle its order, right-click to delete."
        )
        chem_menu.addAction(self._act_build_toggle)

        self._act_select_toggle.setToolTip(
            "3D: click or drag-select atoms. "
            "2D: click an atom to select it, click a bond or anywhere inside "
            "a ring to select the whole ring (Shift adds to the selection)."
        )
        chem_menu.addAction(self._act_select_toggle)

        chem_menu.addSeparator()

        act_chem_reset = QAction("Reset 2D View", self)
        act_chem_reset.triggered.connect(self._reset_chem_view)
        chem_menu.addAction(act_chem_reset)
        self._shortcut_actions["chem_reset_view"] = act_chem_reset

        chem_menu.addSeparator()

        act_chem_carbons = QAction("Show Carbon Labels", self)
        act_chem_carbons.setCheckable(True)
        act_chem_carbons.setChecked(self._chem_canvas.show_carbon_labels)
        act_chem_carbons.setToolTip("Show 'C' on every carbon instead of implicit (unlabeled) vertices")
        act_chem_carbons.triggered.connect(self._toggle_chem_carbon_labels)
        chem_menu.addAction(act_chem_carbons)
        self._act_chem_carbons = act_chem_carbons
        self._shortcut_actions["chem_show_carbons"] = act_chem_carbons

        act_chem_hydrogens = QAction("Show Hydrogens", self)
        act_chem_hydrogens.setCheckable(True)
        act_chem_hydrogens.setChecked(self._chem_canvas.show_hydrogens)
        act_chem_hydrogens.setToolTip("Draw explicit hydrogen atoms and bonds instead of leaving them implicit")
        act_chem_hydrogens.triggered.connect(self._toggle_chem_hydrogens)
        chem_menu.addAction(act_chem_hydrogens)
        self._act_chem_hydrogens = act_chem_hydrogens
        self._shortcut_actions["chem_show_hydrogens"] = act_chem_hydrogens

        chem_menu.addSeparator()

        act_chem_settings = QAction("2D Settings…", self)
        act_chem_settings.setToolTip("Open Settings on the 2D tab (e.g. atom letter colors)")
        act_chem_settings.triggered.connect(lambda: self._edit_settings(tab_index=4))
        chem_menu.addAction(act_chem_settings)

        act_chem_aromaticity = QAction("Check Aromaticity", self)
        act_chem_aromaticity.setToolTip(
            "Perceive aromaticity from the current (explicit Kekulé) bond orders "
            "and show aromatic rings as circles — 2D Viewer only, does not change "
            "bond orders or the 3D view."
        )
        act_chem_aromaticity.triggered.connect(self._check_chem_aromaticity)
        chem_menu.addAction(act_chem_aromaticity)
        self._shortcut_actions["chem_check_aromaticity"] = act_chem_aromaticity

        # ── Help ──
        help_menu = mb.addMenu("&Help")
        act_open_test = QAction("Open Test Molecule", self)
        act_open_test.setShortcut(_mod("Ctrl+Shift+T"))
        act_open_test.triggered.connect(self._open_test_molecule)
        help_menu.addAction(act_open_test)
        self._shortcut_actions["open_test"] = act_open_test

        # "Open Test Files Folder" lives in the Debug menu (see
        # _update_debug_menu) rather than here — created now so it always
        # exists for the shortcut editor, whether or not Debug is shown yet.
        act_open_test_folder = QAction("Open Test Files Folder", self)
        act_open_test_folder.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(
                os.path.join(os.path.dirname(__file__), "test_files"))))
        self._shortcut_actions["open_test_folder"] = act_open_test_folder
        self._act_open_test_folder = act_open_test_folder

        help_menu.addSeparator()

        act_about = QAction("&About Molvector", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)
        self._shortcut_actions["about"] = act_about

    def _setup_builder_toolbar(self):
        assets_dir = os.path.join(os.path.dirname(__file__), "assets", "icons")
        self._build_toolbar_obj = QToolBar("Builder")
        self._build_toolbar_obj.setObjectName("builderToolbar")
        self._build_toolbar_obj.setIconSize(QSize(22, 22))
        self._build_toolbar_obj.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(self._build_toolbar_obj)

        icon_color = '#ccd6f6' if self._current_theme == 'dark' else '#000000'

        # Selection tool
        select_path = os.path.join(assets_dir, "icon_select.svg")
        self._act_select_btn = QAction(load_colored_icon(select_path, icon_color), "Selection", self)
        self._act_select_btn.setCheckable(True)
        self._act_select_btn.setToolTip("Selection tool — click or drag to select atoms")
        self._act_select_btn.triggered.connect(self._toggle_selection_mode)
        self._build_toolbar_obj.addAction(self._act_select_btn)

        # Build tool
        draw_path = os.path.join(assets_dir, "icon_draw.svg")
        self._act_build_btn = QAction(load_colored_icon(draw_path, icon_color), "Build Mode", self)
        self._act_build_btn.setCheckable(True)
        self._act_build_btn.setToolTip("Build mode — add / bond atoms")
        self._act_build_btn.triggered.connect(self._toggle_build_mode)
        self._build_toolbar_obj.addAction(self._act_build_btn)

        # Align tool
        align_path = os.path.join(assets_dir, "icon_align.svg")
        self._act_align_btn = QAction(load_colored_icon(align_path, icon_color), "Align", self)
        self._act_align_btn.setCheckable(True)
        self._act_align_btn.setToolTip("Align tool — click a bond to align it vertically")
        self._act_align_btn.triggered.connect(self._toggle_align_mode)
        self._build_toolbar_obj.addAction(self._act_align_btn)

        # Measure tool
        measure_path = os.path.join(assets_dir, "icon_measure.svg")
        self._act_measure_btn = QAction(load_colored_icon(measure_path, icon_color), "Measure", self)
        self._act_measure_btn.setCheckable(True)
        self._act_measure_btn.setToolTip(
    "Measure tool — click/drag two atoms for distance, Shift+drag three for angle. "
    "Works in the 3D and 2D views alike."
)
        self._act_measure_btn.triggered.connect(self._toggle_measure_mode)
        self._build_toolbar_obj.addAction(self._act_measure_btn)

        clean_path = os.path.join(assets_dir, "icon_clean.svg")
        self._act_clean = QAction(load_colored_icon(clean_path, icon_color), "Clean", self)
        self._act_clean.setToolTip("Clean — rapidly optimize geometry (force field)")
        self._act_clean.triggered.connect(self._clean_molecule)
        self._build_toolbar_obj.addAction(self._act_clean)

        # ── Mode-specific toolbars ────────────────────────────────────────────
        # Each mode has its own toolbar; only the active mode's toolbar is visible.

        # Measure toolbar
        self._measure_toolbar = QToolBar("Measure Tools")
        self._measure_toolbar.setObjectName("measureToolbar")
        self._measure_toolbar.setIconSize(QSize(22, 22))
        self._act_clear_measure = QAction("Clear", self)
        self._act_clear_measure.setToolTip("Clear all measurements")
        self._act_clear_measure.triggered.connect(self._clear_measurements)
        self._measure_toolbar.addAction(self._act_clear_measure)
        self.addToolBar(self._measure_toolbar)
        self._measure_toolbar.setVisible(False)

        # Selection toolbar
        self._selection_toolbar = QToolBar("Selection Tools")
        self._selection_toolbar.setObjectName("selectionToolbar")
        self._selection_toolbar.setIconSize(QSize(22, 22))
        self._act_paint_btn = QAction("Paint atom", self)
        self._act_paint_btn.setCheckable(True)
        self._act_paint_btn.setToolTip("Toggle paint mode — click an atom to paint it (Shift+click to remove paint)")
        self._act_paint_btn.triggered.connect(self._toggle_paint_mode)
        self._selection_toolbar.addAction(self._act_paint_btn)
        self._paint_color_btn = ColorButton("#e63946")
        self._paint_color_btn.setToolTip("Pick paint colour")
        self._paint_color_btn.colorChanged.connect(self._on_paint_color_changed)
        self._selection_toolbar.addWidget(self._paint_color_btn)
        self._act_clear_paint = QAction("Clear paint", self)
        self._act_clear_paint.setToolTip("Remove paint from all atoms")
        self._act_clear_paint.triggered.connect(self._clear_paint)
        self._selection_toolbar.addAction(self._act_clear_paint)
        self.addToolBar(self._selection_toolbar)
        self._selection_toolbar.setVisible(False)

        # Build toolbar
        self._build_options_toolbar = QToolBar("Build Tools")
        self._build_options_toolbar.setObjectName("buildOptionsToolbar")
        self._build_options_toolbar.setIconSize(QSize(22, 22))
        self._build_options_toolbar.addWidget(QLabel(" Element: "))
        self._elem_combo = QComboBox()
        self._elem_combo.setMinimumContentsLength(3)
        self._elem_combo.setMaximumWidth(60)
        self._elem_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._common_elements = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
        self._elem_combo.addItems(self._common_elements + ["..."])
        self._elem_combo.setCurrentText("C")
        self._elem_combo.currentTextChanged.connect(self._on_build_elem_change)
        self._build_options_toolbar.addWidget(self._elem_combo)
        self._build_options_toolbar.addSeparator()
        self._auto_h_check = QCheckBox("Auto adjust H")
        self._auto_h_check.setChecked(False)
        self._auto_h_check.toggled.connect(self._on_auto_h_toggle)
        self._build_options_toolbar.addWidget(self._auto_h_check)

        # 2D-only build subtools: snapping, the bond type selector and the
        # text tool. Their toolbar entries are shown only in the 2D view.
        self._build_2d_only_actions: list[QAction] = [self._build_options_toolbar.addSeparator()]
        magnet_path = os.path.join(assets_dir, "icon_magnet.svg")
        self._act_snapping = QAction(load_colored_icon(magnet_path, icon_color), "Snapping", self)
        self._act_snapping.setCheckable(True)
        self._act_snapping.setToolTip(
            "Snapping — snap the cursor to atoms, ring centers, bond midpoints and ideal next-atom positions")
        self._act_snapping.toggled.connect(lambda on: self._chem_canvas.set_snapping(on))
        self._build_options_toolbar.addAction(self._act_snapping)
        self._build_2d_only_actions.append(self._act_snapping)
        self._bond_type_combo = _IconComboBox()
        self._bond_type_combo.setIconSize(QSize(20, 20))
        self._bond_type_combo.setStyleSheet("QComboBox { padding: 3px 4px; } QComboBox::drop-down { width: 6px; }")
        self._bond_type_combo.setToolTip("Bond type for new bonds, and for bonds you click")
        for k, (kind, label, tip) in enumerate(BOND_TYPES):
            self._bond_type_combo.addItem(bond_type_icon(kind, icon_color), label, kind)
            self._bond_type_combo.setItemData(k, tip, Qt.ItemDataRole.ToolTipRole)
        self._bond_type_combo.currentIndexChanged.connect(
            lambda _i: self._on_bond_type_change(self._bond_type_combo.currentData()))
        self._build_2d_only_actions.append(self._build_options_toolbar.addWidget(self._bond_type_combo))

        # Text tool: while it is on, clicking an atom drawn as letters edits its label
        text_path = os.path.join(assets_dir, "icon_text.svg")
        self._act_text_btn = QAction(load_colored_icon(text_path, icon_color), "Text", self)
        self._act_text_btn.setCheckable(True)
        self._act_text_btn.setToolTip(
            "Text tool — click an atom shown as letters to edit its label "
            "(text, color, bold)"
        )
        self._act_text_btn.toggled.connect(self._toggle_text_mode)
        self._build_options_toolbar.addAction(self._act_text_btn)
        self._build_2d_only_actions.append(self._act_text_btn)
        self.addToolBar(self._build_options_toolbar)
        self._build_options_toolbar.setVisible(False)

        # Align toolbar
        self._align_toolbar = QToolBar("Align Tools")
        self._align_toolbar.setObjectName("alignToolbar")
        self._align_toolbar.setIconSize(QSize(22, 22))
        self._act_align_plane = QAction("Align plane", self)
        self._act_align_plane.setToolTip("Align the plane of 3 selected atoms parallel to the screen")
        self._act_align_plane.triggered.connect(self._align_plane_to_screen)
        self._align_toolbar.addAction(self._act_align_plane)
        self.addToolBar(self._align_toolbar)
        self._align_toolbar.setVisible(False)

        # Map mode name → toolbar for centralized switching
        self._mode_toolbars: dict[str, QToolBar] = {
            "measure":   self._measure_toolbar,
            "selection": self._selection_toolbar,
            "build":     self._build_options_toolbar,
            "align":     self._align_toolbar,
        }

    def _switch_mode_toolbar(self, active_mode: str | None):
        """Show only the toolbar for *active_mode*, hide all others.

        *active_mode* is one of "measure", "selection", "build", "align"
        or ``None`` when no mode is active.
        """
        for name, tb in self._mode_toolbars.items():
            tb.setVisible(name == active_mode)

    # ── toolbar  (only the most frequent actions) ─────────────────────────────

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16,16))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        def tb_action(label, slot, shortcut=None, tooltip=None):
            a = QAction(label, self)
            if shortcut: a.setShortcut(shortcut)
            if tooltip:  a.setToolTip(tooltip)
            a.triggered.connect(slot)
            tb.addAction(a)
            return a

        tb_action("Open",      self._open_file, _mod("Ctrl+O"), "Open molecule file")
        tb_action("Save",        self._save_as,      _mod("Ctrl+S"), "Save molecule file")
        tb_action("Export view", self._export_view,  _mod("Ctrl+E"), "Export view as PDF/PNG/SVG")
        tb.addSeparator()
        tb_action("Reset view",     self._reset_view)
        tb_action("Clear All",      self._clear_molecule)
        tb.addSeparator()

        # 3D / 2D view switch — a two-state slider rather than a plain
        # checkbox, mirroring (and mirrored by) the "Chem View" menu action.
        # "3D"/"2D" are written beside it rather than on the switch itself.
        self._view_mode_lbl_3d = QLabel("3D")
        self._view_mode_lbl_2d = QLabel("2D")
        for lbl in (self._view_mode_lbl_3d, self._view_mode_lbl_2d):
            lbl.setObjectName("view_mode_label")
        tb.addWidget(self._view_mode_lbl_3d)
        self._view_mode_toggle = ViewModeToggle()
        self._view_mode_toggle.toggled.connect(self._toggle_chem_mode)
        tb.addWidget(self._view_mode_toggle)
        tb.addWidget(self._view_mode_lbl_2d)
        self._update_view_mode_labels(False)
        tb.addSeparator()

        # Zoom readout
        self._zoom_lbl = QLabel("100%")
        self._zoom_lbl.setFixedWidth(46)
        self._zoom_lbl.setObjectName("zoom_label")
        # Style this in get_stylesheet instead of here
        tb.addWidget(self._zoom_lbl)

        # Zoom slider
        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(15, 600)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setFixedWidth(120)
        self._zoom_slider.setToolTip("Zoom level (%)")
        self._zoom_slider.valueChanged.connect(self._on_zoom_change)
        self._zoom_slider.installEventFilter(self)
        tb.addWidget(self._zoom_slider)

    # ── central layout ────────────────────────────────────────────────────────

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(172)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(10,14,10,14)
        sl.setSpacing(10)

        # Molecule info
        info = QGroupBox("Molecule")
        il   = QVBoxLayout(info)
        self._lbl_name  = QLabel("—")
        self._lbl_name.setWordWrap(True)
        self._lbl_name.setObjectName("mol_name")
        self._lbl_charge = QLabel("Charge: —")
        self._lbl_atoms = QLabel("Atoms: —")
        self._lbl_bonds = QLabel("Bonds: —")
        self._lbl_mass  = QLabel("Mass: —")
        self._lbl_src   = QLabel("Source: —")
        self._lbl_src.setObjectName("dim")
        self._lbl_src.setStyleSheet("font-size:10px;")
        for w in (self._lbl_name, self._lbl_atoms, self._lbl_bonds, self._lbl_mass, self._lbl_charge, self._lbl_src):
            il.addWidget(w)
        sl.addWidget(info)

        # Calculations info
        self._calc_group = QGroupBox("Calculations")
        cl = QVBoxLayout(self._calc_group)
        self._lbl_energy = QLabel("Energy: —")
        self._lbl_vib = QLabel("Vibrations: —")
        self._lbl_td  = QLabel("States: —")
        for w in (self._lbl_energy, self._lbl_vib, self._lbl_td):
            cl.addWidget(w)
        self._btn_view_calc = QPushButton("View Results…")
        self._btn_view_calc.clicked.connect(self._show_calculations_dialog)
        cl.addWidget(self._btn_view_calc)
        sl.addWidget(self._calc_group)
        self._calc_group.hide()

        # Legend
        self._legend = LegendPanel()
        self._legend.elementColorChanged.connect(self._on_legend_color_changed)
        self._legend.atomColorChanged.connect(self._on_atom_color_changed)
        sl.addWidget(self._legend)
        sl.addStretch()

        # Quick SVG export drag widget (toggle in Settings > General)
        self._quick_svg_tmp_path: Optional[str] = None
        self._quick_svg_widget_enabled = False
        self._quick_svg_drag = _DragWidget(self._get_quick_svg_path, sidebar)
        self._quick_svg_drag.setFixedHeight(52)
        self._quick_svg_drag.setToolTip("Drag the current view into an SVG editor")
        self._quick_svg_drag.hide()
        sl.addWidget(self._quick_svg_drag)

        # Hint
        self._hint = QLabel("Drag  rotate\nRight-drag  pan\nShift+drag  rotate ard. bond\nScroll  zoom")
        self._hint.setObjectName("hint")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(self._hint)

        root.addWidget(sidebar)

        # Canvas stack: 3D molecule view + 2D "Chemvector" whiteboard view
        self._view_stack = QStackedWidget()

        self._canvas = MoleculeCanvas()
        self._canvas.rotationChanged.connect(self._on_rotation_changed)
        self._canvas.fileDropped.connect(self._load_and_display)
        self._canvas.moleculeChanged.connect(self._on_structure_changed)
        self._canvas.requestHistorySave.connect(self._save_history)
        self._view_stack.addWidget(self._canvas)

        self._chem_canvas = ChemCanvas()
        self._chem_canvas.fileDropped.connect(self._load_and_display)
        self._chem_canvas.moleculeChanged.connect(self._on_chem_structure_changed)
        self._chem_canvas.requestHistorySave.connect(self._save_history)
        self._chem_canvas.viewChanged.connect(self._on_chem_view_changed)
        self._chem_canvas.zoomChanged.connect(self._on_chem_zoom_changed)
        self._view_stack.addWidget(self._chem_canvas)

        self._chem_mode = False
        self._chem_view_dirty = True

        root.addWidget(self._view_stack, 1)

    # ── status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            "No file loaded — use File → Open to load an .xyz, .gjf, .com or .log file."
        )

    # ── placeholder ───────────────────────────────────────────────────────────

    def _show_placeholder(self):
        bg = THEMES[self._current_theme]["CANVAS"]
        fg = THEMES[self._current_theme]["FG_DIM"]
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="450" viewBox="0 0 600 450">' 
            f'<rect width="600" height="450" fill="{bg}"/>' 
            f'<text x="300" y="210" text-anchor="middle" font-family="monospace" font-size="15" fill="{fg}">' 
            'Open a molecule file to begin' 
            '</text>' 
            '</svg>'
        ).encode("ascii")
        self._canvas.load(QByteArray(svg))
        self._chem_canvas.load(QByteArray(svg))

    # ── drag and drop ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            self._load_and_display(path)
            event.acceptProposedAction()
        else:
            event.ignore()

    # ── file open / save ──────────────────────────────────────────────────────

    def _load_mol_from_path(self, path: str):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        base = os.path.basename(path)
        name = os.path.splitext(base)[0]

        mol = None
        src = ""
        errors = []

        # The parser the extension names goes first; the others follow, in
        # case the extension is wrong or generic (.txt).  Trying the Gaussian
        # log parser first on everything loaded cclib for every plain XYZ.
        parsers = [
            ("Gaussian log", lambda t: parse_gaussian_log(t)),
            ("XYZ", lambda t: parse_xyz(t, name=name)),
            ("PDB", lambda t: parse_pdb(t, name=name)),
            ("MDL Molfile", lambda t: parse_mol(t, name=name)),
            ("Gaussian input", lambda t: parse_gaussian(t))
        ]
        by_ext = {".log": "Gaussian log", ".out": "Gaussian log", ".xyz": "XYZ",
                  ".pdb": "PDB", ".ent": "PDB", ".mol": "MDL Molfile", ".sdf": "MDL Molfile",
                  ".gjf": "Gaussian input", ".com": "Gaussian input", ".gau": "Gaussian input"}
        first = by_ext.get(os.path.splitext(base)[1].lower())
        parsers.sort(key=lambda p: p[0] != first)

        with _suppress_ob_stderr():
            for source_name, parser in parsers:
                try:
                    mol = parser(text)
                    if mol and mol.atoms:
                        src = source_name
                        break
                except Exception as e:
                    errors.append(f"{source_name}: {e}")

        if mol is None or not mol.atoms:
            err_msg = "\n".join(errors)
            raise ValueError(f"Could not parse file format.\nAttempts:\n{err_msg}")

        infer_bonds(mol)
        return mol, src

    def _load_and_display(self, path: str):
        """Parse a file and update all UI elements. Used by drag-and-drop and Open."""
        try:
            mol, src = self._load_mol_from_path(path)
            self._display_loaded_molecule(mol, src, path)
            if not getattr(self, '_restoring_session', False):
                self.show_notification(f"Loaded {os.path.basename(path)}", "success")
        except Exception as e:
            if not getattr(self, '_restoring_session', False):
                log_path = _log_error(f"Failed to load {path}", e)
                self.show_notification("Failed to load file.", "error", 0, log_file=log_path)

    def _display_loaded_molecule(self, mol: Molecule, src: str, path: str):
        """Show a freshly opened molecule in the 3D and 2D views and the sidebar."""
        self._color_overrides = {}
        self._canvas.color_overrides = {}
        self._canvas.load_molecule(mol)
        self._legend.update_for(mol, {})
        self._current_source = src
        self._current_path = path
        self._update_info_panel(mol)
        self._update_calculations_menu(mol)
        self._mark_chem_dirty(reset_view=True)

    def _open_from_catalog(self):
        try:
            dlg = CatalogDialog(self, view3d=None if self._chem_mode else self._canvas,
                                layout=self._chem_canvas.layout_mode)
        except Exception as e:
            log_path = _log_error("Failed to open the molecule catalog", e)
            self.show_notification("Could not read the molecule catalog.", "error", 0, log_file=log_path)
            return
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.selected_record is None:
            return
        from molvector import catalog
        rec = dlg.selected_record
        try:
            self._display_loaded_molecule(catalog.to_molecule(rec), "Catalog", "")
            self.show_notification(f"Loaded {rec['name']} from the catalog", "success")
        except Exception as e:
            log_path = _log_error(f"Failed to load catalog entry {rec['id']}", e)
            self.show_notification("Failed to load catalog entry.", "error", 0, log_file=log_path)

    def _update_calculations_menu(self, mol: Molecule):
        # Keep G16 Input, sep, Rotational Constants, Dipole Moment, sep, Open Terminal
        while len(self._menu_calc.actions()) > 6:
            self._menu_calc.removeAction(self._menu_calc.actions()[-1])

        if mol and (mol.vibrational_modes or mol.excited_states or mol.mulliken_charges):
            self._menu_calc.addSeparator()
            act_results = QAction("Calculation Results…", self)
            overrides, alt_overrides = self._load_shortcut_overrides()
            ks = overrides.get("calc_results", _mod("Ctrl+M"))
            act_results.setShortcut(ks)
            alt_ks = alt_overrides.get("calc_results", "")
            if alt_ks and ks:
                act_results.setShortcuts([QKeySequence(ks), QKeySequence(alt_ks)])
            act_results.triggered.connect(self._show_calculations_dialog)
            self._menu_calc.addAction(act_results)
            self._shortcut_actions["calc_results"] = act_results

    def _generate_g16_input(self):
        mol = self._canvas.molecule
        if not mol or not mol.atoms:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return
        dlg = G16InputDialog(mol, parent=self)
        dlg.exec()

    def _show_calculations_dialog(self):
        mol = self._canvas.molecule
        if not mol: return
        dlg = CalculationsDialog(mol, parent=self)
        dlg.modeSelected.connect(self._show_vibration)
        dlg.stateSelected.connect(self._show_excited_state)
        dlg.viewSpectrum.connect(self._on_view_spectrum)
        dlg.animationToggled.connect(self._on_anim_toggle)
        dlg.vectorsToggled.connect(self._on_vectors_toggle)
        dlg.colorByCharge.connect(self._color_atoms_by_mulliken)
        dlg.clearAtomColors.connect(self._clear_paint)
        dlg.finished.connect(self._on_calculations_closed)
        dlg.show()

    def _color_atoms_by_mulliken(self, range_min: float, range_max: float, center: float,
                                  neg_color: str, zero_color: str, pos_color: str,
                                  show_legend: bool, include_in_export: bool):
        mol = self._canvas.molecule
        if not mol or not mol.mulliken_charges:
            return
        for atom, charge in zip(mol.atoms, mol.mulliken_charges):
            atom.color = mulliken_gradient_color(charge, range_min, range_max, neg_color, zero_color, pos_color, center)
        self._canvas.color_overrides = {}
        self._color_overrides = {}
        self._canvas.mulliken_legend = (range_min, range_max, center, neg_color, zero_color, pos_color) if show_legend else None
        self._canvas.mulliken_legend_in_export = include_in_export
        # Re-applying the gradient reassigns the SAME atom indices new colours;
        # without invalidating the per-atom gradient-stop cache, the renderer
        # keeps drawing the stale stops from the previous application.
        self._canvas._cached_grad_stops.clear()
        # NOT self._legend.update_for(): it builds one sidebar row per
        # custom-coloured atom, and Mulliken coloring gives every atom its
        # own colour — for a few hundred atoms that rebuild alone takes over
        # a second and freezes the UI on every drag/edit. The gradient bar
        # (in the dialog and, if enabled, on the canvas) is the legend for
        # this mode; the per-element sidebar legend isn't meaningful here.
        self._canvas.request_render()
        self._status.showMessage(f"Coloured {len(mol.atoms)} atom(s) by Mulliken charge.")

    def _calculate_rotational_constants(self):
        mol = self._canvas.molecule
        if not mol:
            QMessageBox.information(self, 'No molecule', "Load or build a molecule first.")
            return

        rot_consts = calculate_rotational_constants(mol)
    
        QMessageBox.information(self, 'Rotational constants', f"Rotational constants:\nA: {rot_consts[0]:10.4f} MHz\nB: {rot_consts[1]:10.4f} MHz\nC: {rot_consts[2]:10.4f} MHz")
        return(rot_consts)

    def _require_openbabel(self) -> bool:
        """Load OpenBabel on first use; show a hint and return False if unusable."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok = ensure_openbabel()
        finally:
            QApplication.restoreOverrideCursor()
        if not ok:
            self.show_notification(openbabel_unavailable_message(), "warning", 0)
        return ok

    def _calculate_dipole_moment(self):
        mol = self._canvas.molecule
        if not mol:
            QMessageBox.information(self, 'No molecule', "Load or build a molecule first.")
            return

        if not self._require_openbabel():
            return

        mu_vec, mu_mag, atom_charges = calculate_dipole_moment(mol)
        if mu_mag is None:
            QMessageBox.information(self, 'Dipole moment',
                "Could not compute dipole moment.\n"
                "Ensure the molecule has at least 2 atoms and valid coordinates.")
            return

        # Compute centre of mass in centred coordinates (matching render frame)
        centered = center_positions(mol.atoms)
        com = np.zeros(3)
        total_mass = 0.0
        for i, atom in enumerate(mol.atoms):
            mass = ATOMIC_MASSES.get(atom.element, 0.0)
            com += mass * centered[i]
            total_mass += mass
        if total_mass > 0:
            com /= total_mass

        # Store dipole arrow data on canvas (start=+ end, end=- end)
        DEBYE_PER_EANGSTROM = 4.803204
        mu_angstrom = mu_vec / DEBYE_PER_EANGSTROM
        self._canvas.dipole_arrow_3d = (com + mu_angstrom, com - mu_angstrom, mu_mag)

        formula = chemical_formula(mol)

        lines = [
            f"Molecule:  {formula}",
            f"Charge:    {mol.charge}",
            "",
            f"|μ| = {mu_mag:.4f} Debye",
            "",
            f"μx = {mu_vec[0]:+.4f} D",
            f"μy = {mu_vec[1]:+.4f} D",
            f"μz = {mu_vec[2]:+.4f} D",
            "",
            "Partial charges (EEM):",
        ]
        for elem, q in atom_charges:
            lines.append(f"  {elem:2s}  {q:+.4f} e")

        dlg = QDialog(self)
        dlg.setWindowTitle("Dipole moment")
        dlg.setMinimumWidth(380)
        layout = QVBoxLayout(dlg)

        text_label = QLabel("\n".join(lines))
        text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        text_label.setFont(_monospace_font())
        layout.addWidget(text_label)

        show_cb = QCheckBox("Show dipole moment")
        show_cb.setChecked(self._canvas.show_dipole_arrow)
        show_cb.toggled.connect(self._on_dipole_arrow_toggled)
        layout.addWidget(show_cb)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(dlg.accept)
        layout.addWidget(btn_box)

        dlg.exec()

    def _on_dipole_arrow_toggled(self, checked: bool):
        self._canvas.show_dipole_arrow = checked
        self._canvas.request_render()

    def _open_terminal(self):
        if self._terminal_dialog is not None:
            try:
                if self._terminal_dialog.isVisible():
                    self._terminal_dialog.raise_()
                    self._terminal_dialog.activateWindow()
                    return
            except RuntimeError:
                self._terminal_dialog = None
        workdir = os.path.dirname(os.path.abspath(self._current_path)) if self._current_path else ""
        self._terminal_dialog = TerminalDialog(
            workdir=workdir, current_file=self._current_path,
            mol=self._canvas.molecule, parent=self,
        )
        self._terminal_dialog.destroyed.connect(lambda: self._on_terminal_destroyed())
        self._terminal_dialog.show()

    def _on_terminal_destroyed(self):
        self._terminal_dialog = None

    def _on_anim_toggle(self, enabled: bool):
        if enabled:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
            self._canvas.animation_amplitude = 0.0
            self._canvas.request_render()

    def _on_vectors_toggle(self, enabled: bool):
        self._canvas._show_vectors = enabled
        self._canvas.request_render()

    def _on_anim_step(self):
        self._anim_phase += 0.25
        if self._anim_phase > 2*math.pi:
            self._anim_phase -= 2*math.pi
        self._canvas.animation_phase = self._anim_phase
        self._canvas.animation_amplitude = 0.6
        self._canvas.request_render()

    def _on_calculations_closed(self):
        self._anim_timer.stop()
        self._anim_phase = 0.0
        self._canvas.animation_phase = 0.0
        self._canvas.animation_amplitude = 0.0
        self._canvas.active_vectors = None
        self._canvas.arrows = None
        self._canvas.request_render()
        self._status.clearMessage()

    def _on_view_spectrum(self, kind):
        if kind == "ir": self._view_ir_spectrum()
        else: self._view_uvvis_spectrum()

    def _view_ir_spectrum(self):
        mol = self._canvas.molecule
        if not mol: return
        x = [m.frequency for m in mol.vibrational_modes]
        y = [m.intensity for m in mol.vibrational_modes]
        meta = f"Molecule: {mol.name}\nCalculation: Vibrational Frequencies (IR)"
        safe_name = get_safe_filename(mol.name)
        default_file = f"{safe_name}_ir_spectrum.txt"
        dlg = SpectrumDialog(x, y, "Frequency / cm-1", "Intensity (arb. units)", "IR Spectrum", meta, self)
        dlg.set_default_filename(default_file)
        dlg.exec()

    def _view_uvvis_spectrum(self):
        mol = self._canvas.molecule
        if not mol: return
        x = [s.wavelength_nm for s in mol.excited_states]
        y = [s.oscillator_strength for s in mol.excited_states]
        meta = f"Molecule: {mol.name}\nCalculation: Excited States (TDDFT)"
        safe_name = get_safe_filename(mol.name)
        default_file = f"{safe_name}_uvvis_spectrum.txt"
        dlg = SpectrumDialog(x, y, "Wavelength / nm", "Oscillator Strength / f", "UV-Vis Spectrum", meta, self)
        dlg.set_default_filename(default_file)
        dlg.exec()

    def _reset_state(self):
        self._canvas.arrows = None
        self._canvas.active_vectors = None
        self._canvas.request_render()
        if self._canvas.molecule:
            self._update_info_panel(self._canvas.molecule)

    def _show_vibration(self, mode: VibrationalMode):
        # Scale displacements for visibility
        self._canvas.active_vectors = mode.displacements
        scale = 1.2
        arrows = []
        for i, d in enumerate(mode.displacements):
            arrows.append((i, d[0]*scale, d[1]*scale, d[2]*scale, "#00ff00"))
        
        self._canvas.arrows = arrows
        self._canvas.request_render()
        self._status.showMessage(f"Visualizing mode {mode.index}: {mode.frequency:.1f} cm⁻¹")

    def _show_excited_state(self, state: ExcitedState):
        self._canvas.arrows = None
        self._canvas.active_vectors = None
        self._canvas.request_render()
        
        msg = f"Excited State {state.index}: {state.symmetry} | {state.energy_ev:.3f} eV | f={state.oscillator_strength:.4f}"
        self._status.showMessage(msg)
        
        formula = chemical_formula(self._canvas.molecule)
        # Keep name consistent, don't add (S1) here if user wants consistency
        # Or just use plain text
        self._lbl_name.setText(formula)

    def _on_legend_color_changed(self, elem: str, hex_color: str):
        self._color_overrides[elem] = hex_color
        self._canvas.color_overrides[elem] = hex_color
        self._canvas._cached_grad_stops.clear()
        self._canvas.request_render()
        if self._canvas.molecule:
            self._legend.update_for(self._canvas.molecule, self._color_overrides)
        self._chem_canvas.color_overrides[elem] = hex_color
        self._chem_canvas.request_render()

    def _on_atom_color_changed(self, atom_idx: int, hex_color: str):
        mol = self._canvas.molecule
        if mol is None or atom_idx < 0 or atom_idx >= len(mol.atoms):
            return
        mol.atoms[atom_idx].color = hex_color
        self._canvas._cached_grad_stops.clear()
        self._canvas.request_render()
        self._legend.update_for(mol, self._color_overrides)
        self._chem_canvas.request_render()
        self._status.showMessage(f"Atom {atom_idx + 1} colour set to {hex_color}.")

    def _toggle_theme(self):
        new_theme = "light" if self._current_theme == "dark" else "dark"
        self._apply_theme(new_theme)
        self._canvas.background = THEMES[new_theme]["CANVAS"]
        self._canvas.request_render()

    def _apply_theme(self, theme_name: str):
        self._current_theme = theme_name
        self.setStyleSheet(get_stylesheet(theme_name))
        
        if hasattr(self, "_act_theme"):
            self._act_theme.setText("Switch to Dark Mode" if theme_name == "light" else "Switch to Light Mode")
            
        # Update toolbar icon colors
        self._update_tool_icons()

        self.setProperty("theme", theme_name)
        self.style().unpolish(self)
        self.style().polish(self)

    def _update_tool_icons(self):
        if not hasattr(self, '_act_select_btn') or not hasattr(self, '_act_build_btn'):
            return
        assets_dir = os.path.join(os.path.dirname(__file__), "assets", "icons")
        color = '#ccd6f6' if self._current_theme == 'dark' else '#000000'
        self._act_select_btn.setIcon(load_colored_icon(os.path.join(assets_dir, "icon_select.svg"), color))
        self._act_build_btn.setIcon(load_colored_icon(os.path.join(assets_dir, "icon_draw.svg"), color))
        if hasattr(self, '_act_align_btn'):
            self._act_align_btn.setIcon(load_colored_icon(os.path.join(assets_dir, "icon_align.svg"), color))
        if hasattr(self, '_act_measure_btn'):
            self._act_measure_btn.setIcon(load_colored_icon(os.path.join(assets_dir, "icon_measure.svg"), color))
        if hasattr(self, '_act_clean'):
            self._act_clean.setIcon(load_colored_icon(os.path.join(assets_dir, "icon_clean.svg"), color))
        if hasattr(self, '_act_snapping'):
            self._act_snapping.setIcon(load_colored_icon(os.path.join(assets_dir, "icon_magnet.svg"), color))
        if hasattr(self, '_act_text_btn'):
            self._act_text_btn.setIcon(load_colored_icon(os.path.join(assets_dir, "icon_text.svg"), color))
        if hasattr(self, '_bond_type_combo'):
            for k, (kind, _label, _tip) in enumerate(BOND_TYPES):
                self._bond_type_combo.setItemIcon(k, bond_type_icon(kind, color))

    def _update_info_panel(self, mol):
        """Populate sidebar labels and status bar for a loaded molecule."""
        formula = chemical_formula(mol)
        mass    = molecular_mass(mol)
        charge  = mol.charge
        src     = self._current_source
        path    = self._current_path

        # Build name: plain text C60+ etc.
        if charge == 0:
            display_name = formula
        elif charge > 0:
            display_name = f"{formula}{charge}+" if charge > 1 else f"{formula}+"
        else:
            display_name = f"{formula}{abs(charge)}-" if charge < -1 else f"{formula}-"

        charge_str = "neutral" if charge == 0 else (f"+{charge}" if charge > 0 else str(charge))

        self._lbl_name.setText(display_name)
        self._lbl_charge.setText(f"Charge: {charge_str}")
        self._lbl_atoms.setText(f"Atoms: {len(mol.atoms)}")
        self._lbl_bonds.setText(f"Bonds: {len(mol.bonds)}")
        self._lbl_mass.setText(f"Mass: {mass:.3f} uma")
        self._lbl_src.setText(f"Source: {src}")

        # Update Calculations Group
        has_vib = bool(mol.vibrational_modes)
        has_td  = bool(mol.excited_states)
        has_energy = mol.g16_scf_energy is not None or mol.g16_opt_energy is not None
        has_any = has_vib or has_td or has_energy
        if has_any:
            self._calc_group.show()
            if has_energy:
                eng = mol.g16_opt_energy if mol.g16_opt_energy is not None else mol.g16_scf_energy
                self._lbl_energy.setText(f"Energy: {eng:.4f} Ha")
                self._lbl_energy.setVisible(True)
            else:
                self._lbl_energy.setVisible(False)
            self._lbl_vib.setVisible(has_vib)
            self._lbl_td.setVisible(has_td)
            self._lbl_vib.setText(f"Vibrations: {len(mol.vibrational_modes)} modes")
            self._lbl_td.setText(f"TD-DFT: {len(mol.excited_states)} states")
        else:
            self._calc_group.hide()

        # Update main window title
        self.setWindowTitle(f"Molvector — {display_name}")

        status_path = f"{path}  |  " if path else ""
        self._status.showMessage(
            f"{status_path}{display_name}  |  {len(mol.atoms)} atoms, "
            f"{len(mol.bonds)} bonds  |  {mass:.3f} uma  [{src}]"
        )
        self._legend.update_for(mol, self._color_overrides)
        self._update_quick_svg_widget()

    def _show_molecule_info(self):
        mol = self._canvas.molecule
        if mol is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return

        from collections import Counter
        formula = chemical_formula(mol)
        mass = molecular_mass(mol)
        charge = mol.charge
        charge_str = "neutral" if charge == 0 else (f"+{charge}" if charge > 0 else str(charge))
        counts = Counter(a.element for a in mol.atoms)
        elem_parts = [f"{e}\u2009{counts[e]}" for e in sorted(counts.keys())]
        elem_str = "  " + ", ".join(elem_parts) if mol.atoms else "—"
        src = self._current_source or "—"
        path = self._current_path or "—"

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Info \u2014 {formula}")
        dlg.setMinimumWidth(400)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(6)

        for label, value in [
            ("Formula:", formula),
            ("Charge:", charge_str),
            ("Mass:", f"{mass:.3f} uma"),
            ("Atoms:", str(len(mol.atoms))),
            ("Bonds:", str(len(mol.bonds))),
            ("Source:", src),
            ("Path:", path),
            ("Elements:", elem_str),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight: bold;")
            val = QLabel(value)
            val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            val.setWordWrap(True)
            row.addWidget(lbl)
            row.addWidget(val, 1)
            layout.addLayout(row)

        layout.addSpacing(10)
        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn.rejected.connect(dlg.reject)
        layout.addWidget(btn)
        dlg.exec()

    def _show_inchi(self):
        mol = self._canvas.molecule
        if mol is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return

        inchi = generate_inchi(mol)
        if inchi is None:
            self.show_notification(
                "Failed to generate InChI for this molecule.\n"
                "Ensure RDKit is installed:  pip install rdkit",
                "warning", 0
            )
            return

        formula = chemical_formula(mol)
        warnings = check_valence_issues(mol)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"InChI \u2014 {formula}")
        dlg.setMinimumWidth(480)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(6)

        lbl = QLabel("InChI:")
        lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl)

        le = QLineEdit(inchi)
        le.setReadOnly(True)
        layout.addWidget(le)

        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(inchi))
        layout.addWidget(copy_btn)

        nist_btn = QPushButton("Search in NIST WebBook")
        from urllib.parse import quote
        encoded = quote(inchi, safe='')
        nist_url = QUrl(f"https://webbook.nist.gov/cgi/cbook.cgi?InChI={encoded}&Units=SI")
        nist_btn.clicked.connect(lambda: QDesktopServices.openUrl(nist_url))
        layout.addWidget(nist_btn)

        if warnings:
            layout.addSpacing(6)
            warn_lbl = QLabel("Valence warnings:")
            warn_lbl.setStyleSheet("font-weight: bold; color: #cc4444;")
            layout.addWidget(warn_lbl)

            warn_text = QLabel("\n".join(warnings))
            warn_text.setStyleSheet("color: #cc4444;")
            warn_text.setWordWrap(True)
            layout.addWidget(warn_text)

        layout.addSpacing(6)
        close_btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_btn.rejected.connect(dlg.reject)
        layout.addWidget(close_btn)
        dlg.exec()

    def _edit_charge(self):
        mol = self._canvas.molecule
        if mol is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Charge")
        dlg.setMinimumWidth(240)
        layout = QVBoxLayout(dlg)

        lbl = QLabel("Molecular charge:")
        layout.addWidget(lbl)

        spin = QSpinBox()
        spin.setRange(-10, 10)
        spin.setValue(mol.charge)
        spin.setMinimumWidth(90)
        layout.addWidget(spin)

        layout.addSpacing(6)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            mol.charge = spin.value()
            self._update_info_panel(mol)
            self._canvas.update()
            self._status.showMessage(f"Charge set to {mol.charge}")

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Molecule File", "",
            "All supported (*.xyz *.gjf *.com *.log *.out *.pdb *.mol *.txt);;"
            "XYZ (*.xyz);;Gaussian input (*.gjf *.com);;"
            "Gaussian log (*.log *.out);;PDB (*.pdb);;MDL Molfile (*.mol);;Text (*.txt);;All files (*)"
        )
        if path:
            self._load_and_display(path)

    def _export_canvas(self):
        """The canvas whose drawing is exported: the 2D whiteboard while it is showing, else the 3D view."""
        return self._chem_canvas if self._chem_mode else self._canvas

    def _save_svg(self):
        if self._canvas.molecule is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return
        safe_name = get_safe_filename(self._canvas.molecule.name)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save SVG", f"{safe_name}_view.svg", "SVG files (*.svg)"
        )
        if not path:
            return
        try:
            data = self._export_canvas().get_svg_bytes(export_mode=True)
            with open(path, "wb") as f:
                f.write(data)
            self.show_notification(f"Saved {os.path.basename(path)}", "success")
        except Exception as e:
            log_path = _log_error(f"SVG save failed for {path}", e)
            self.show_notification("SVG save failed.", "error", 0, log_file=log_path)

    def _quick_svg_export(self):
        if self._canvas.molecule is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return
        dlg = QuickExportDialog(self._export_canvas(), self)
        dlg.show()

    def _get_quick_svg_path(self) -> str:
        """Write the current canvas view to a temp SVG file for the sidebar drag widget."""
        svg_data = self._export_canvas().get_svg_bytes(export_mode=True)
        if self._quick_svg_tmp_path and os.path.exists(self._quick_svg_tmp_path):
            try:
                os.unlink(self._quick_svg_tmp_path)
            except OSError:
                pass
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            f.write(svg_data)
            self._quick_svg_tmp_path = f.name
        return self._quick_svg_tmp_path

    def _update_quick_svg_widget(self):
        """Show/hide the sidebar Quick SVG export drag widget per the current setting."""
        enabled = bool(getattr(self, "_quick_svg_widget_enabled", False))
        self._quick_svg_drag.setVisible(enabled and self._canvas.molecule is not None)

    def _edit_shortcuts(self):
        _, saved_alt = self._load_shortcut_overrides()
        alt = dict(self.ALT_SHORTCUT_DEFAULTS)
        alt.update(saved_alt)
        alt = {k: v for k, v in alt.items() if k in self._shortcut_actions}
        dlg = ShortcutDialog(self._shortcut_actions, alt_defaults=alt, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            primary, alt = dlg.get_shortcuts()
            self._apply_shortcuts(primary)
            self._apply_alt_shortcuts(alt)
            self._save_shortcut_overrides(primary, alt)

    def _apply_shortcut_config(self):
        primary, alt = self._load_shortcut_overrides()
        if primary:
            self._apply_shortcuts(primary)
        if alt:
            self._apply_alt_shortcuts(alt)

    def _apply_shortcuts(self, overrides: dict):
        for aid, seq_str in overrides.items():
            if aid in self._shortcut_actions and seq_str:
                self._shortcut_actions[aid].setShortcut(seq_str)

    def _apply_alt_shortcuts(self, alt: dict):
        for aid, seq_str in alt.items():
            if aid in self._shortcut_actions and seq_str:
                action = self._shortcut_actions[aid]
                primary = action.shortcut().toString()
                if primary:
                    action.setShortcuts([QKeySequence(primary), QKeySequence(seq_str)])
                else:
                    action.setShortcut(seq_str)

    def _load_shortcut_overrides(self) -> tuple:
        cfg = paths.read_json(CONFIG_FILE, {})
        if not isinstance(cfg, dict):
            return {}, {}
        return cfg.get("shortcuts", {}), cfg.get("alt_shortcuts", {})

    def _save_shortcut_overrides(self, primary: dict, alt: dict):
        try:
            _update_config({"shortcuts": primary, "alt_shortcuts": alt})
        except OSError as e:
            log_path = _log_error("Could not save shortcuts", e)
            self.show_notification("Shortcuts could not be saved.", "error", 0, log_file=log_path)

    def _export_view(self):
        if self._canvas.molecule is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return

        safe_name = get_safe_filename(self._canvas.molecule.name)
        filters = "PDF files (*.pdf);;PNG files (*.png);;JPEG files (*.jpg *.jpeg);;SVG files (*.svg)"
        path, sel_filter = QFileDialog.getSaveFileName(
            self, "Export View", f"{safe_name}_view.pdf", filters
        )
        if not path:
            return
        path = _with_filter_extension(path, sel_filter, (".pdf", ".png", ".jpg", ".jpeg", ".svg"))

        try:
            # 1. Get SVG data from canvas
            svg_data = self._export_canvas().get_svg_bytes(export_mode=True)
            renderer = QSvgRenderer(QByteArray(svg_data))
            
            # 2. Determine size (use high res for raster)
            # We'll use a 2x or 3x scale for images to make them crisp
            view_size = renderer.defaultSize()
            if view_size.isEmpty():
                view_size = QSize(1200, 900)
            
            ext = os.path.splitext(path)[1].lower()
            
            if ext == ".pdf":
                pdf = QPdfWriter(path)
                pdf.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
                # Center the molecule on the page
                painter = QPainter(pdf)
                if not painter.isActive():
                    raise OSError(f"Could not write PDF file: {path}")
                try:
                    # Scale molecule to fit page comfortably
                    target_rect = painter.viewport()
                    # Maintain aspect ratio
                    svg_ratio = view_size.width() / view_size.height()
                    page_ratio = target_rect.width() / target_rect.height()
                    
                    if svg_ratio > page_ratio:
                        w = target_rect.width()
                        h = int(w / svg_ratio)
                    else:
                        h = target_rect.height()
                        w = int(h * svg_ratio)
                    
                    # Center it
                    x = (target_rect.width() - w) // 2
                    y = (target_rect.height() - h) // 2
                    renderer.render(painter, QRectF(float(x), float(y), float(w), float(h)))
                finally:
                    painter.end()
            
            elif ext in (".png", ".jpg", ".jpeg"):
                # Scale up for high quality
                scale_factor = 2.0
                img_size = view_size * scale_factor
                img = QImage(img_size, QImage.Format.Format_ARGB32)
                img.fill(Qt.GlobalColor.transparent if ext == ".png" else Qt.GlobalColor.white)
                
                painter = QPainter(img)
                try:
                    renderer.render(painter)
                finally:
                    painter.end()
                
                if not img.save(path):
                    raise OSError(f"Could not write image file: {path}")

            elif ext == ".svg":
                with open(path, "wb") as f:
                    f.write(svg_data)

            else:
                raise ValueError(f"Unsupported export format: {ext or '(none)'}")

            self.show_notification(f"Exported {os.path.basename(path)}", "success")
        except Exception as e:
            log_path = _log_error(f"Export failed for {path}", e)
            self.show_notification("Export failed.", "error", 0, log_file=log_path)

    def _save_as(self):
        mol = self._canvas.molecule
        if not mol:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return
            
        safe_name = get_safe_filename(mol.name)
        filters = "XYZ (*.xyz);;PDB (*.pdb);;MDL Molfile (*.mol);;Gaussian input (*.gjf *.com);;All files (*)"
        path, sel_filter = QFileDialog.getSaveFileName(
            self, "Save Molecule As", f"{safe_name}.xyz", filters
        )
        if not path:
            return
        path = _with_filter_extension(path, sel_filter, (".xyz", ".pdb", ".mol", ".gjf", ".com"))

        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".xyz":
                text = save_xyz(mol)
            elif ext == ".pdb":
                text = save_pdb(mol)
            elif ext == ".mol":
                text = save_mol(mol)
            elif ext in (".gjf", ".com"):
                text = save_gaussian_input(mol)
            else:
                # Default to XYZ if extension is unknown
                text = save_xyz(mol)
                
            # LF endings on every platform: these files go to Linux tools and clusters.
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            self.show_notification(f"Saved {os.path.basename(path)}", "success")
        except Exception as e:
            log_path = _log_error(f"Save failed for {path}", e)
            self.show_notification("Save failed.", "error", 0, log_file=log_path)

    # ── Tool mode actions ────────────────────────────────────────────────────

    def _update_status_for_mode(self):
        if self._chem_mode:
            if self._chem_canvas.text_mode:
                self._status.showMessage(
                    "2D Text — click an atom shown as letters to edit its label, color and boldness. "
                    "Enter to confirm, Esc to cancel."
                )
                self._hint.setText("Click label  edit atom\nEnter  confirm\nEsc  cancel")
            elif self._chem_canvas.build_mode:
                if self._chem_canvas.bond_type == "ring":
                    self._status.showMessage(
                        "2D Build, ring bond — drag from inside a ring out over one of its bonds; "
                        "release on an atom or empty space. Right-click a ring bond to delete it."
                    )
                else:
                    self._status.showMessage(
                        "2D Build — click empty space to add an atom, drag from an atom to bond it "
                        "(with the selected bond type), click a bond to apply the bond type, right-click to delete."
                    )
                self._hint.setText("Click  add atom\nDrag  bond atoms\nClick bond  apply bond type\nShift+drag  60° snap")
            elif self._chem_canvas.selection_mode:
                self._status.showMessage(
                    "2D Selection — click an atom to select it, click a bond or inside a ring "
                    "to select the whole ring (Shift adds to the selection)."
                )
                self._hint.setText("Click atom  select\nClick ring  select ring\nShift  add to selection")
            elif self._chem_canvas.align_mode:
                self._status.showMessage(
                    "2D Align — click a bond to make it vertical · Shift+click horizontal · "
                    f"{ALT_CLICK_HINT} 3 atoms, then Align plane, to draw the molecule in their plane"
                )
                self._hint.setText(f"Click bond  align vertically\nShift+click  align horizontally\n{ALT_CLICK_HINT}  mark atoms for plane")
            elif self._chem_canvas.measure_mode:
                self._status.showMessage(
                    "2D Measure — click or drag two atoms for distance · Shift+drag three atoms for angle · Esc to clear"
                )
                self._hint.setText("Click/drag  measure distance\nShift+drag  measure angle\nEsc  clear all")
            else:
                self._status.showMessage("Chemvector — 2D whiteboard view. Enable Build or Selection Mode to interact here.")
                self._hint.setText("Drag  pan\nCtrl+drag  rotate\nScroll  zoom\nDouble-click  reset view")
            return
        if self._canvas.build_mode:
            self._status.showMessage("Build: Left-click add atom · Drag to bond · Click bond change order · Scroll zoom")
            self._hint.setText("Left-click  add atom\nDrag  bond atoms\nClick bond  change order")
        elif self._canvas.selection_mode:
            self._status.showMessage("Selection: Click select atom · Drag rectangle-select · Scroll zoom")
            self._hint.setText("Click  select atom\nDrag  rectangle-select")
        elif self._canvas.align_mode:
            self._status.showMessage(f"Align: Click bond align vertically · Shift+click align horizontally · {ALT_CLICK_HINT} atoms to mark for plane alignment")
            self._hint.setText(f"Click bond  align vertically\nShift+click  align horizontally\n{ALT_CLICK_HINT}  mark atoms for plane")
        elif self._canvas.paint_mode:
            self._status.showMessage("Paint: Click atom to paint · Shift+click to remove paint")
            self._hint.setText("Click atom  paint\nShift+click  remove paint")
        elif self._canvas.measure_mode:
            self._status.showMessage("Measure: Click or drag atoms for distance · Shift+drag for angle · Esc to clear")
            self._hint.setText("Click/drag  measure distance\nShift+drag  measure angle\nEsc  clear all")
        else:
            self._status.showMessage("Left-drag rotate molecule · Right-drag pan · Shift+drag rotate ard. bond · Scroll zoom")
            self._hint.setText("Drag  rotate\nRight-drag  pan\nShift+drag  rotate ard. bond\nScroll  zoom")

    def _deactivate_all_modes(self):
        """Uncheck every mode's UI controls and canvas flags."""
        for btn, toggle, attr in [
            (self._act_select_btn, self._act_select_toggle, "selection_mode"),
            (self._act_build_btn,   self._act_build_toggle,   "build_mode"),
            (self._act_align_btn,   self._act_align_toggle,   "align_mode"),
            (self._act_measure_btn, self._act_measure_toggle, "measure_mode"),
        ]:
            btn.setChecked(False)
            toggle.setChecked(False)
            setattr(self._canvas, attr, False)
        self._chem_canvas.build_mode = False
        self._chem_canvas.selection_mode = False
        self._chem_canvas.paint_mode = False
        self._chem_canvas.align_mode = False
        self._chem_canvas.measure_mode = False
        self._act_text_btn.setChecked(False)
        self._chem_canvas.set_text_mode(False)
        self._chem_canvas._green_atoms.clear()
        self._chem_canvas._measure_atoms.clear()
        self._chem_canvas._angle_atoms.clear()
        self._chem_canvas._angle_leg1 = None
        self._chem_canvas._measure_from = None
        self._chem_canvas.update()
        self._act_paint_btn.setChecked(False)
        self._canvas.paint_mode = False
        self._canvas._measure_atoms.clear()
        self._canvas._angle_atoms.clear()
        self._canvas._angle_leg1 = None
        self._canvas._measure_from = None

        self._canvas._green_atoms.clear()

    def _deactivate_non_build_modes(self):
        """Like _deactivate_all_modes, but leaves Build/Selection Mode alone —
        used when switching between the 3D and 2D views, where those two
        carry over (applied to whichever canvas is now active) but
        Align/Measure are reset here (Align is re-applied by the caller)."""
        for btn, toggle, attr in [
            (self._act_align_btn,   self._act_align_toggle,   "align_mode"),
            (self._act_measure_btn, self._act_measure_toggle, "measure_mode"),
        ]:
            btn.setChecked(False)
            toggle.setChecked(False)
            setattr(self._canvas, attr, False)
        self._act_paint_btn.setChecked(False)
        self._canvas.paint_mode = False
        self._chem_canvas.paint_mode = False
        self._canvas._measure_atoms.clear()
        self._canvas._angle_atoms.clear()
        self._canvas._angle_leg1 = None
        self._canvas._measure_from = None
        self._canvas._green_atoms.clear()
        self._chem_canvas.align_mode = False
        self._chem_canvas.measure_mode = False
        self._act_text_btn.setChecked(False)
        self._chem_canvas.set_text_mode(False)
        self._chem_canvas._green_atoms.clear()
        self._chem_canvas._measure_atoms.clear()
        self._chem_canvas._angle_atoms.clear()
        self._chem_canvas._angle_leg1 = None
        self._chem_canvas._measure_from = None

    def _toggle_selection_mode(self, enabled: bool):
        """One Selection Mode toggle, shared by the 3D and 2D (Chemvector)
        views — it applies to whichever canvas is currently on screen. In
        2D: click an atom to select it, or click a bond/ring interior to
        select the whole ring."""
        if enabled:
            self._deactivate_all_modes()
        self._act_select_btn.setChecked(enabled)
        self._act_select_toggle.setChecked(enabled)
        if self._chem_mode:
            self._canvas.selection_mode = False
            self._chem_canvas.selection_mode = enabled
        else:
            self._chem_canvas.selection_mode = False
            self._canvas.selection_mode = enabled
            self._canvas._update_cursor(self._canvas._mouse_pos or QPoint(0, 0), Qt.KeyboardModifier.NoModifier)
        self._switch_mode_toolbar("selection" if enabled else None)
        self._update_status_for_mode()

    def _toggle_build_mode(self, enabled: bool):
        """One Build Mode toggle, shared by the 3D and 2D (Chemvector) views —
        it applies to whichever canvas is currently on screen."""
        if enabled:
            self._deactivate_all_modes()
        else:
            self._act_text_btn.setChecked(False)   # the text tool is a build subtool
        self._act_build_toggle.setChecked(enabled)
        self._act_build_btn.setChecked(enabled)
        if self._chem_mode:
            self._canvas.build_mode = False
            self._chem_canvas.build_mode = enabled
            self._chem_canvas.build_element = self._elem_combo.currentText() or "C"
            self._chem_canvas.setCursor(QCursor(
                Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor
            ))
        else:
            self._chem_canvas.build_mode = False
            self._canvas.build_mode = enabled
            self._canvas._update_cursor(self._canvas._mouse_pos or QPoint(0, 0), Qt.KeyboardModifier.NoModifier)
        for act in self._build_2d_only_actions:
            act.setVisible(self._chem_mode)
        self._switch_mode_toolbar("build" if enabled else None)
        self._update_status_for_mode()

    def _on_bond_type_change(self, kind: str):
        self._chem_canvas.bond_type = kind
        self._update_status_for_mode()

    def _toggle_text_mode(self, enabled: bool):
        """Text tool — a 2D Build subtool: while on, clicking an atom drawn as
        letters edits its label (text, color, bold) instead of adding atoms
        (the bond type doesn't apply then)."""
        enabled = enabled and self._chem_mode and self._chem_canvas.build_mode
        if self._act_text_btn.isChecked() != enabled:
            self._act_text_btn.setChecked(enabled)   # re-enters here via toggled
            return
        self._chem_canvas.set_text_mode(enabled)
        if not enabled and self._chem_canvas.build_mode:
            self._chem_canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self._bond_type_combo.setEnabled(not enabled)
        self._update_status_for_mode()

    def _toggle_align_mode(self, enabled: bool):
        """One Align Mode toggle, shared by the 3D and 2D views — it acts on
        whichever canvas is on screen (in 2D: in-plane, and the 3D camera
        follows)."""
        if enabled:
            self._deactivate_all_modes()
        self._act_align_toggle.setChecked(enabled)
        self._act_align_btn.setChecked(enabled)
        if self._chem_mode:
            self._canvas.align_mode = False
            self._chem_canvas.align_mode = enabled
            self._chem_canvas.setCursor(QCursor(
                Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor
            ))
        else:
            self._chem_canvas.align_mode = False
            self._canvas.align_mode = enabled
            self._canvas._update_cursor(self._canvas._mouse_pos or QPoint(0, 0), Qt.KeyboardModifier.NoModifier)
        self._switch_mode_toolbar("align" if enabled else None)
        self._update_status_for_mode()

    def _align_plane_to_screen(self):
        if self._chem_mode:
            c = self._chem_canvas
            if len(c._green_atoms) != 3:
                self.show_notification(f"Select exactly 3 atoms ({ALT_CLICK_HINT} in Align mode) to define a plane.", "warning", 0)
                return
            if not c.align_plane(sorted(c._green_atoms)):
                self.show_notification("The 3 selected atoms are collinear; cannot define a plane.", "warning", 0)
                return
            c._green_atoms.clear()
            c.update()
            return
        green = self._canvas._green_atoms
        if len(green) != 3:
            self.show_notification(f"Select exactly 3 atoms ({ALT_CLICK_HINT} in Alignment mode) to define a plane.", "warning", 0)
            return
        mol = self._canvas.molecule
        if mol is None:
            return
        indices = sorted(green)
        a1 = mol.atoms[indices[0]]
        a2 = mol.atoms[indices[1]]
        a3 = mol.atoms[indices[2]]
        v1 = np.array([a2.x - a1.x, a2.y - a1.y, a2.z - a1.z])
        v2 = np.array([a3.x - a1.x, a3.y - a1.y, a3.z - a1.z])
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            self.show_notification("The 3 selected atoms are collinear; cannot define a plane.", "warning", 0)
            return
        normal = normal / norm_len
        target = np.array([0.0, 0.0, 1.0])
        v_normal_view = self._canvas._rot @ normal
        dot = np.dot(v_normal_view, target)
        if dot < 0:
            target = -target
            dot = -dot
        if dot < 0.9999:
            sin_theta = np.linalg.norm(np.cross(v_normal_view, target))
            if sin_theta > 1e-10:
                cos_theta = dot
                axis = np.cross(v_normal_view, target) / sin_theta
                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
                R_corr = np.eye(3) + sin_theta * K + (1.0 - cos_theta) * (K @ K)
                self._canvas._rot = R_corr @ self._canvas._rot
                self._canvas.rotationChanged.emit()
                self._canvas.request_render()
        self._canvas._green_atoms.clear()
        self._canvas.update()

    def _toggle_paint_mode(self, enabled: bool):
        """Paint Mode is shared between the 3D and 2D views like Build/Selection
        Mode — it acts on whichever canvas is currently on screen. If atoms are
        already selected there, turning it on paints them immediately instead
        of entering an interactive click-to-paint mode."""
        canvas = self._chem_canvas if self._chem_mode else self._canvas
        if enabled:
            if canvas.selected_atoms:
                self._paint_selected_with_color()
                self._act_paint_btn.setChecked(False)
                return
        self._act_paint_btn.setChecked(enabled)
        canvas.paint_mode = enabled
        if self._chem_mode:
            self._chem_canvas.setCursor(QCursor(
                Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor
            ))
        else:
            self._canvas._update_cursor(self._canvas._mouse_pos or QPoint(0, 0), Qt.KeyboardModifier.NoModifier)
        self._update_status_for_mode()

    def _toggle_measure_mode(self, enabled: bool):
        """One Measure Mode toggle, shared by the 3D and 2D views — it acts
        on whichever canvas is on screen. Distances/angles are always the
        real 3D quantities, whichever view is used to pick the atoms."""
        if enabled:
            self._deactivate_all_modes()
        self._act_measure_toggle.setChecked(enabled)
        self._act_measure_btn.setChecked(enabled)
        canvas = self._chem_canvas if self._chem_mode else self._canvas
        other = self._canvas if self._chem_mode else self._chem_canvas
        canvas.measure_mode = enabled
        other.measure_mode = False
        self._switch_mode_toolbar("measure" if enabled else None)
        if not enabled:
            for c in (self._canvas, self._chem_canvas):
                c._measure_atoms.clear()
                c._measure_from = None
                c._mouse_pos = None
                c._angle_atoms.clear()
                c._angle_leg1 = None
                c.update()
        self._update_status_for_mode()
        if self._chem_mode:
            self._chem_canvas.setCursor(QCursor(
                Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor
            ))
        else:
            self._canvas._update_cursor(self._canvas._mouse_pos or QPoint(0, 0), Qt.KeyboardModifier.NoModifier)

    def _clear_measurements(self):
        for c in (self._canvas, self._chem_canvas):
            c._measure_atoms.clear()
            c._measure_from = None
            c._mouse_pos = None
            c._angle_atoms.clear()
            c._angle_leg1 = None
            c.update()

    def _on_paint_color_changed(self, hex_color: str):
        self._canvas._paint_color = hex_color
        self._chem_canvas._paint_color = hex_color

    def _paint_selected_with_color(self):
        mol = self._canvas.molecule
        if mol is None:
            return
        if self._chem_mode:
            # 2D's own paint layer — separate from Atom.color, so this never
            # touches (or is touched by) the 3D view's paint.
            sel = self._chem_canvas.selected_atoms
            if not sel:
                return
            hex_col = self._chem_canvas._paint_color
            for idx in sel:
                self._chem_canvas.atom_paint[idx] = hex_col
            self._chem_canvas.selected_atoms.clear()
            self._chem_canvas.request_render()
            self._status.showMessage(f"Painted {len(sel)} atom(s) with {hex_col} (2D only).")
            return
        sel = self._canvas.selected_atoms
        if not sel:
            return
        hex_col = self._canvas._paint_color
        for idx in sel:
            mol.atoms[idx].color = hex_col
        self._canvas.selected_atoms.clear()
        self._canvas.color_overrides = {}
        self._color_overrides = {}
        self._legend.update_for(mol, {})
        self._canvas.request_render()
        self._status.showMessage(f"Painted {len(sel)} atom(s) with {hex_col}.")

    def _on_build_elem_change(self, elem: str):
        if elem == "...":
            prev = self._canvas.build_element
            dlg = PeriodicTableDialog(self, prev, self._color_overrides)
            dlg.elementSelected.connect(self._on_pick_from_periodic_table)
            if not dlg.exec():
                self._elem_combo.setCurrentText(prev)
            return
        self._canvas.build_element = elem
        self._chem_canvas.build_element = elem

    def _on_pick_from_periodic_table(self, sym: str):
        idx = self._elem_combo.findText(sym)
        if idx < 0:
            others_idx = self._elem_combo.findText("...")
            self._elem_combo.insertItem(others_idx, sym)
        self._elem_combo.setCurrentText(sym)

    def _on_auto_h_toggle(self, checked: bool):
        self._canvas.auto_adjust_h = checked

    def _toggle_bond_order(self, checked: bool):
        self._canvas.bond_order_disabled = checked
        if checked:
            if self._canvas.molecule:
                self._canvas._saved_bond_orders = {(b.i, b.j): b.order for b in self._canvas.molecule.bonds}
                for b in self._canvas.molecule.bonds:
                    b.order = 1
            self._act_bond_order.setText("Enable Bond Order")
            self._status.showMessage("Bond order disabled — all bonds set to single")
        else:
            if self._canvas.molecule and self._canvas._saved_bond_orders is not None:
                for b in self._canvas.molecule.bonds:
                    saved = self._canvas._saved_bond_orders.get((b.i, b.j))
                    if saved is not None:
                        b.order = saved
                self._canvas._saved_bond_orders = None
            self._act_bond_order.setText("Disable Bond Order")
            self._status.showMessage("Bond order enabled — click bonds to cycle order")
        self._canvas.request_render()

    def _clear_molecule(self):
        if QMessageBox.question(self, "Clear", "Clear the entire molecule?") == QMessageBox.StandardButton.Yes:
            self._save_history()
            self._canvas.molecule = Molecule("New Molecule", atoms=[])
            self._canvas.request_render()
            self._update_info_panel(self._canvas.molecule)
            self._mark_chem_dirty()

    def _clean_molecule(self):
        mol = self._canvas.molecule
        if not mol or not mol.atoms:
            return
        if self._optimize_worker is not None:
            self._status.showMessage("Optimization already running…")
            return

        worker = _OptimizeWorker(mol, self._ff_max_steps, self._ff_tol, self)
        elements = list(worker.elements)
        worker.result.connect(
            lambda coords, steps, err: self._on_optimize_done(mol, elements, coords, steps, err))
        worker.finished.connect(self._on_optimize_thread_finished)
        self._optimize_worker = worker
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        self._status.showMessage("Optimizing geometry…")
        worker.start()

    def _on_optimize_thread_finished(self):
        if self._optimize_worker is not None:
            self._optimize_worker.deleteLater()
            self._optimize_worker = None
        QApplication.restoreOverrideCursor()

    def _on_optimize_done(self, mol, elements, coords, steps_taken, err):
        """Apply a finished optimization (GUI thread)."""
        self._status.clearMessage()
        if err is not None:
            if err == "unavailable":
                self.show_notification(openbabel_unavailable_message(), "warning", 0)
            else:
                log_path = _log_error("Molecule optimization failed", err)
                self.show_notification("Optimization failed.", "error", 0, log_file=log_path)
            return
        # The user may have edited or replaced the molecule meanwhile.
        current = self._canvas.molecule
        if (current is not mol or len(coords) != len(mol.atoms)
                or [a.element for a in mol.atoms] != elements):
            self.show_notification("The molecule changed during optimization; result discarded.", "warning")
            return

        self._save_history()
        for a, (x, y, z) in zip(mol.atoms, coords):
            a.x, a.y, a.z = x, y, z
        self._canvas.request_render()
        self._update_info_panel(self._canvas.molecule)
        self._mark_chem_dirty()
        self.show_notification(f"Molecule cleaned ({steps_taken} iterations).", "success")

    def _on_structure_changed(self):
        # UI updates only; history should be saved BEFORE the change occurs
        # to capture the 'before' state correctly.
        self._update_info_panel(self._canvas.molecule)
        if self._canvas.molecule is not None:
            self._legend.update_for(self._canvas.molecule, self._color_overrides)
        self._mark_chem_dirty()

    # ── Chemvector (2D whiteboard) view ─────────────────────────────────────

    def _mark_chem_dirty(self, reset_view: bool = False):
        """Flag that the 2D depiction is stale; it is regenerated lazily on next view / switch.

        While the 2D view is showing, an edit (clear, undo/redo, clean, ...)
        keeps its zoom/pan; only *reset_view* (a different molecule was
        opened) refits it.
        """
        self._chem_view_dirty = True
        if self._chem_mode:
            if reset_view:
                self._face_molecule_plane()
            self._sync_chem_view(keep_view=not reset_view)

    def _face_molecule_plane(self):
        """
        Turn the shared camera so the molecule's plane faces the screen, for
        the 2D view to be drawn in plane rather than as a tilted projection:
        the plane chosen with Align plane while it still applies, else the
        best-fit plane. The smallest turn is used, keeping the in-plane angle.
        """
        mol = self._canvas.molecule
        if mol is None or not mol.atoms:
            return
        plane = None if self._chem_view_dirty else self._chem_canvas._plane
        normal = plane[0] if plane is not None else best_fit_plane_normal(mol)
        if normal is None:
            return
        self._canvas._rot = face_plane_rotation(np.asarray(normal, dtype=float), self._canvas._rot)

    def _sync_chem_view(self, force: bool = False, keep_view: bool = False):
        """Push the current 3D molecule + appearance into the Chemvector canvas."""
        self._chem_canvas.background = self._canvas.background
        self._chem_canvas.color_overrides = dict(self._color_overrides)
        # Same scale (px per Angstrom) as the 3D view, so bonds match in length.
        self._chem_canvas.base_scale = self._canvas.base_scale
        if not force and not self._chem_view_dirty:
            # Structure unchanged — just turn the existing 2D layout to match
            # wherever the 3D view has been rotated to.
            self._chem_canvas.align_to_view(self._canvas._rot)
            return
        self._chem_canvas.load_molecule(
            self._canvas.molecule, keep_view=keep_view, view_rot=self._canvas._rot)
        self._chem_view_dirty = False

    def _on_chem_view_changed(self, rot):
        """The 2D view was turned (rotate / align) — give the 3D camera the same orientation."""
        self._canvas._rot = np.array(rot, dtype=float)
        self._canvas.request_render()

    def _update_view_mode_labels(self, chem_mode: bool):
        """Bold/brighten whichever of the "3D"/"2D" toolbar labels matches the active view."""
        active_style = "font-weight: bold;"
        dim_style = "color: palette(mid);"
        self._view_mode_lbl_3d.setStyleSheet(dim_style if chem_mode else active_style)
        self._view_mode_lbl_2d.setStyleSheet(active_style if chem_mode else dim_style)

    def _toggle_chem_mode(self, enabled: bool):
        self._chem_mode = enabled
        self._act_chem_mode.setChecked(enabled)
        self._view_mode_toggle.setChecked(enabled)
        self._update_view_mode_labels(enabled)

        build_was_active = self._act_build_toggle.isChecked()
        select_was_active = self._act_select_toggle.isChecked()
        align_was_active = self._act_align_toggle.isChecked()
        measure_was_active = self._act_measure_toggle.isChecked()
        self._deactivate_non_build_modes()

        if enabled:
            shared_zoom = self._canvas._zoom
            self._face_molecule_plane()
            self._sync_chem_view()  # a refit inside may reset zoom; the shared zoom wins
            self._on_chem_zoom_changed(shared_zoom)
            self._chem_canvas._zoom = shared_zoom
            self._chem_canvas._pan = self._canvas._pan.copy()   # same framing as the 3D view
            # Draw it now, before it is shown — a deferred render would flash
            # the picture from before the shared zoom/orientation changed.
            self._view_stack.setCurrentWidget(self._chem_canvas)
            self._chem_canvas.resize(self._view_stack.size())   # render at the size it will be shown at
            self._chem_canvas._render_timer.stop()
            self._chem_canvas._do_render()
            self._hint.setText("Drag  pan\nCtrl+drag  rotate\nScroll  zoom\nDouble-click  reset view")
            self._status.showMessage("Chemvector — 2D whiteboard view. Enable Build Mode to sketch here.")
            if not HAS_RDKIT and not getattr(self, '_rdkit_hint_shown', False):
                self._rdkit_hint_shown = True
                if native_guard.crashed("rdkit"):
                    _msg = ("Chem view is using a simplified layout — RDKit is installed "
                            "but was disabled because it crashes in this environment.")
                else:
                    _msg = ("Chem view is using a simplified layout — install RDKit "
                            "(pip install rdkit) for proper 2D depictions and stereo wedges.")
                self.show_notification(_msg, "info", 7000)
        else:
            # Zoom (and rotation) may have been changed from the 2D view while
            # this one was hidden — draw it with the current shared state.
            self._canvas._pan = self._chem_canvas._pan.copy()
            self._view_stack.setCurrentWidget(self._canvas)
            self._canvas.resize(self._view_stack.size())
            self._canvas._render_timer.stop()
            self._canvas._do_render()
            self._hint.setText("Drag  rotate\nRight-drag  pan\nShift+drag  rotate ard. bond\nScroll  zoom")
            self._update_status_for_mode()

        if build_was_active:
            # Build/Selection Mode are shared between the two views — carry
            # whichever was active over onto the now-active canvas instead
            # of dropping it.
            self._toggle_build_mode(True)
        elif select_was_active:
            self._toggle_selection_mode(True)
        elif align_was_active:
            self._toggle_align_mode(True)
        elif measure_was_active:
            self._toggle_measure_mode(True)

    def _reset_chem_view(self):
        self._chem_canvas.reset_view()

    def _toggle_chem_carbon_labels(self, enabled: bool):
        self._chem_canvas.show_carbon_labels = enabled
        self._chem_canvas.request_render()

    def _toggle_chem_hydrogens(self, enabled: bool):
        self._chem_canvas.show_hydrogens = enabled
        self._chem_canvas.request_render()

    def _check_chem_aromaticity(self):
        """2D Viewer only: perceive aromaticity from the molecule's current
        explicit (Kekulé) bond orders and show those rings as a circle in the
        2D view. This is a purely 2D-local overlay (ChemCanvas.aromatic_bonds)
        — it never writes to bond.order, so the shared Molecule and the 3D
        view are completely untouched."""
        mol = self._canvas.molecule
        if mol is None or not mol.atoms:
            self.show_notification("Load or build a molecule first.", "warning")
            return
        aromatic_bond_idxs = check_aromaticity(mol)
        new_bonds = aromatic_bond_idxs - self._chem_canvas.aromatic_bonds
        if not aromatic_bond_idxs:
            self._status.showMessage("Check Aromaticity — no aromatic rings found.")
            return
        self._chem_canvas.aromatic_bonds |= aromatic_bond_idxs
        self._chem_canvas.request_render()
        if not new_bonds:
            self._status.showMessage("Check Aromaticity — no new aromatic rings found.")
        else:
            self._status.showMessage(
                f"Check Aromaticity — marked {len(aromatic_bond_idxs)} bond(s) aromatic (2D view only)."
            )

    def _on_chem_structure_changed(self):
        """A 2D build-mode edit mutated the shared Molecule directly — refresh both views."""
        if self._canvas.molecule is None and self._chem_canvas.molecule is not None:
            self._canvas.molecule = self._chem_canvas.molecule   # first atom/text drawn on an empty 2D view
        self._update_info_panel(self._canvas.molecule)
        self._canvas.request_render()

    def _save_history(self):
        import copy
        if self._canvas.molecule:
            # We save a snapshot of the CURRENT state before it gets modified.
            snap = copy.deepcopy(self._canvas.molecule)
            self._history.append(snap)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            # Clear redo stack on NEW action
            self._redo_stack.clear()

    def _undo(self):
        if not self._history:
            self._status.showMessage("Nothing to undo.")
            return
        
        import copy
        # Save CURRENT state to redo stack before going back
        if self._canvas.molecule:
            self._redo_stack.append(copy.deepcopy(self._canvas.molecule))

        # Restore the most recent snapshot
        prev = self._history.pop()
        self._canvas.molecule = prev
        self._canvas.request_render()
        self._update_info_panel(self._canvas.molecule)
        self._mark_chem_dirty()
        self._status.showMessage("Undo successful.")

    def _redo(self):
        if not self._redo_stack:
            self._status.showMessage("Nothing to redo.")
            return
        
        import copy
        # Save CURRENT state to history before going forward
        if self._canvas.molecule:
            self._history.append(copy.deepcopy(self._canvas.molecule))
            if len(self._history) > self._max_history:
                self._history.pop(0)

        # Restore from redo stack
        next_state = self._redo_stack.pop()
        self._canvas.molecule = next_state
        self._canvas.request_render()
        self._update_info_panel(self._canvas.molecule)
        self._mark_chem_dirty()
        self._status.showMessage("Redo successful.")

    def _edit_ff_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Optimization Parameters")
        l = QVBoxLayout(dlg)
        
        from PyQt6.QtWidgets import QFormLayout, QSpinBox, QDialogButtonBox
        form = QFormLayout()
        
        s_steps = QSpinBox()
        s_steps.setRange(10, 100000)
        s_steps.setValue(self._ff_max_steps)
        form.addRow("Max Steps:", s_steps)
        
        l.addLayout(form)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        l.addWidget(btns)
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._ff_max_steps = s_steps.value()
            self._clean_molecule()

    # ── View actions ─────────────────────────────────────────────────────────

    def _set_bond_style(self, style: str):
        self._canvas.bond_style = style
        self._canvas.request_render()

    def _set_ball_size(self, scale: float):
        self._canvas.atom_scale = scale
        self._canvas.request_render()

    def _set_bond_width(self, w: float):
        self._canvas.bond_width_px = w
        self._canvas.request_render()

    # ── Edit actions ──────────────────────────────────────────────────────────

    def _load_appearance_config(self):
        cfg = SettingsDialog.load_config()
        if cfg is None:
            return
        g = cfg.get("general", {})
        o = cfg.get("overlay", {})
        a = cfg.get("atoms_bonds", {})
        c2 = cfg.get("chem2d", {})
        if c2:
            self._chem_canvas.label_color_mode = c2.get("label_colors", "cpk")
            self._chem_canvas.layout_mode = c2.get("layout", "match3d")
            self._chem_canvas.request_render()
        if g:
            theme = g.get("theme", "light")
            if theme != self._current_theme:
                self._apply_theme(theme)
            self._canvas.background = o.get("bg_color", g.get("bg_color", self._canvas.background))
            self._restore_on_startup = g.get("restore_molecule", False)
            self._start_view_pref = g.get("start_view", "3D")
            self._canvas._fast_render = g.get("fast_render", False)
            self._show_debug_menu = g.get("show_debug_menu", False)
            self._quick_svg_widget_enabled = g.get("quick_svg_export_widget", False)
            self._update_quick_svg_widget()
        if a:
            self._canvas.atom_scale = a.get("atom_scale", self._canvas.atom_scale)
            self._canvas.bond_width_px = a.get("bond_width_px", self._canvas.bond_width_px)
            self._canvas.bond_style = a.get("bond_style", self._canvas.bond_style)
            self._canvas.bond_color = a.get("bond_color", self._canvas.bond_color)
            self._canvas.atom_border_mode = a.get("atom_border_mode", self._canvas.atom_border_mode)
            self._canvas.atom_border_scale = a.get("atom_border_scale", self._canvas.atom_border_scale)
            self._canvas.atom_border_width = a.get("atom_border_width", self._canvas.atom_border_width)
            self._canvas.lighting_intensity = a.get("lighting_intensity", self._canvas.lighting_intensity)
            self._canvas.light_position = a.get("light_position", self._canvas.light_position)
            self._canvas.roughness = a.get("roughness", self._canvas.roughness)
            self._color_overrides = a.get("color_overrides", {})
            self._canvas.color_overrides = self._color_overrides
        if o:
            self._canvas.show_axes = o.get("show_axes", self._canvas.show_axes)
            self._canvas.show_principal_axes = o.get("show_principal_axes", self._canvas.show_principal_axes)
            self._canvas.axes_position = o.get("axes_position", self._canvas.axes_position)
            self._canvas.principal_axes_position = o.get("principal_axes_position", self._canvas.principal_axes_position)
            self._canvas.show_dipole_arrow = o.get("show_dipole_arrow", self._canvas.show_dipole_arrow)
        self._canvas.request_render()

    @staticmethod
    def _parse_session(data) -> tuple:
        """Validate a saved session: (Molecule or None, {canvas attribute: value}).

        Raises on anything malformed, before the UI is touched.
        """
        if not isinstance(data, (dict, str)):
            raise ValueError(f"unexpected session data: {type(data).__name__}")
        mol = None
        if isinstance(data, dict) and "molecule" in data:
            md = data["molecule"]
            atoms = [Atom(**a) for a in md.get("atoms", [])]
            bonds = [Bond(**b) for b in md.get("bonds", [])]
            for b in bonds:
                if not (0 <= b.i < len(atoms) and 0 <= b.j < len(atoms)):
                    raise ValueError(f"bond {b.i}-{b.j} refers to a missing atom")
            ring_bonds = md.get("ring_bonds", [])
            for rb in ring_bonds:
                if not all(0 <= k < len(atoms) for k in [rb["atom"], *rb["ring"]]):
                    raise ValueError("ring bond refers to a missing atom")
            mult = md.get("multiplicity")
            mol = Molecule(name=str(md.get("name", "")), atoms=atoms, bonds=bonds,
                           charge=int(md.get("charge", 0)),
                           multiplicity=int(mult) if mult is not None else None,
                           ring_bonds=ring_bonds)

        view = {}
        if isinstance(data, dict):
            shapes = {"rot": (3, 3), "pan": (2,), "axes_ref": (3, 3), "default_view": (3, 3)}
            for key, shape in shapes.items():
                if key in data:
                    arr = np.array(data[key], dtype=float)
                    if arr.shape != shape or not np.all(np.isfinite(arr)):
                        raise ValueError(f"bad '{key}' in session view")
                    view[f"_{key}"] = arr
            if "zoom" in data:
                zoom = float(data["zoom"])
                if not (math.isfinite(zoom) and zoom > 0):
                    raise ValueError("bad 'zoom' in session view")
                view["_zoom"] = zoom
        return mol, view

    def _maybe_restore_molecule(self):
        if not getattr(self, '_restore_on_startup', False):
            return
        path = self.LAST_MOLECULE_FILE
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.loads(f.read().strip())
            mol, view = self._parse_session(data)
        except Exception as e:
            # A corrupt or incompatible session file must never block startup.
            log_path = _log_error(f"Could not restore the last session from {path}", e)
            QTimer.singleShot(500, lambda: self.show_notification(
                "The last session could not be restored.", "warning", log_file=log_path))
            return

        if mol is not None:
            self._color_overrides = {}
            self._canvas.color_overrides = {}
            self._canvas.load_molecule(mol)
            self._legend.update_for(mol, {})
            self._current_source = "Session"
            self._current_path = data.get("path", "")
            self._update_info_panel(mol)
            self._update_calculations_menu(mol)
        else:
            mol_path = data.get("path", "") if isinstance(data, dict) else data
            if isinstance(mol_path, str) and mol_path and os.path.isfile(mol_path):
                self._restoring_session = True
                try:
                    self._load_and_display(mol_path)
                finally:
                    self._restoring_session = False
                mol = self._canvas.molecule

        if mol and mol.atoms and view:
            c = self._canvas
            for attr, value in view.items():
                setattr(c, attr, value)
            pct = int(c._zoom * 100)
            self._zoom_slider.blockSignals(True)
            self._zoom_slider.setValue(pct)
            self._zoom_slider.blockSignals(False)
            self._zoom_lbl.setText(f"{pct}%")
            c.request_render()
            QTimer.singleShot(500, lambda: self.show_notification("Session restored", "info"))

    def _edit_settings(self, tab_index=0):
        cfg = SettingsDialog.load_config() or {}
        restore = cfg.get("general", {}).get("restore_molecule", False)
        show_debug = cfg.get("general", {}).get("show_debug_menu", False)
        quick_svg_widget = getattr(self, "_quick_svg_widget_enabled", False)
        start_view_pref = getattr(self, "_start_view_pref", "3D")
        orig = (self._current_theme, self._canvas.background,
                self._canvas.atom_scale, self._canvas.bond_width_px,
                self._canvas.bond_style, dict(self._color_overrides),
                self._canvas.atom_border_mode, self._canvas.atom_border_scale,
                self._canvas.atom_border_width, self._canvas.bond_color,
                self._canvas.lighting_intensity, self._canvas.light_position,
                self._canvas.roughness,
                self._canvas.show_axes, self._canvas.show_principal_axes,
                self._canvas.axes_position, self._canvas.principal_axes_position,
                restore, self._canvas._fast_render, show_debug, quick_svg_widget,
                start_view_pref)

        def _live_update(theme, bg, ball, bw, style, colors, border_mode, border_scale,
                         border_width, bcol, lighting, pos, rough,
                         show_xyz, show_abc, xyz_pos, abc_pos):
            if theme != self._current_theme:
                self._apply_theme(theme)
            self._canvas.background = bg
            self._canvas.atom_scale = ball
            self._canvas.bond_width_px = bw
            self._canvas.bond_style = style
            self._canvas.bond_color = bcol
            self._canvas.atom_border_mode = border_mode
            self._canvas.atom_border_scale = border_scale
            self._canvas.atom_border_width = border_width
            self._canvas.lighting_intensity = lighting
            self._canvas.light_position = pos
            self._canvas.roughness = rough
            self._color_overrides = colors
            self._canvas.color_overrides = colors
            self._canvas.show_axes = show_xyz
            self._canvas.show_principal_axes = show_abc
            self._canvas.axes_position = xyz_pos
            self._canvas.principal_axes_position = abc_pos
            if self._canvas.molecule:
                self._legend.update_for(self._canvas.molecule, colors)
            self._canvas.request_render()

        orig_chem_labels = self._chem_canvas.label_color_mode
        orig_chem_layout = self._chem_canvas.layout_mode

        def _chem_live_update(labels, layout):
            self._chem_canvas.label_color_mode = labels
            self._chem_canvas.set_layout_mode(layout)
            self._chem_canvas.request_render()

        _, saved_alt = self._load_shortcut_overrides()
        alt_defaults = dict(self.ALT_SHORTCUT_DEFAULTS)
        alt_defaults.update(saved_alt)
        alt_defaults = {k: v for k, v in alt_defaults.items() if k in self._shortcut_actions}

        dlg = SettingsDialog(
            orig[0], orig[1], orig[2], orig[3], orig[4], orig[5],
            atom_border_mode=orig[6], atom_border_scale=orig[7], atom_border_width=orig[8],
            bond_color=orig[9], lighting_intensity=orig[10],
            light_position=orig[11], roughness=orig[12],
            show_axes=orig[13], show_principal_axes=orig[14],
            axes_position=orig[15], principal_axes_position=orig[16],
            restore_molecule=orig[17],
            fast_render=orig[18],
            show_debug_menu=orig[19],
            quick_svg_widget=orig[20],
            shortcut_actions=self._shortcut_actions,
            alt_shortcut_defaults=alt_defaults,
            live_callback=_live_update, parent=self,
            show_dipole_arrow=self._canvas.show_dipole_arrow,
            start_view=orig[21],
            chem_label_colors=orig_chem_labels, chem_layout=orig_chem_layout,
            chem_callback=_chem_live_update,
        )
        dlg.tabs.setCurrentIndex(tab_index)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._apply_theme(dlg.theme)
            self._canvas.background = dlg.bg_color
            self._canvas.atom_scale = dlg.ball_scale
            self._canvas.bond_width_px = dlg.bond_width
            self._canvas.bond_style = dlg.bond_style
            self._canvas.bond_color = dlg.bond_color
            self._canvas.atom_border_mode = dlg.border_mode
            self._canvas.atom_border_scale = dlg.border_scale
            self._canvas.atom_border_width = dlg.border_width
            self._canvas.lighting_intensity = dlg.lighting_intensity
            self._canvas.light_position = dlg.light_position
            self._canvas.roughness = dlg.roughness
            self._canvas.show_axes = dlg.show_axes
            self._canvas.show_principal_axes = dlg.show_principal_axes
            self._canvas.axes_position = dlg.axes_position
            self._canvas.principal_axes_position = dlg.principal_axes_position
            self._canvas.show_dipole_arrow = dlg.show_dipole_arrow
            self._color_overrides = dlg._color_overrides
            self._canvas.color_overrides = dlg._color_overrides
            self._canvas._cached_grad_stops.clear()
            self._canvas._fast_render = dlg.fast_render
            self._show_debug_menu = dlg.show_debug_menu
            self._quick_svg_widget_enabled = dlg.quick_svg_export_widget
            self._start_view_pref = dlg.start_view
            self._chem_canvas.label_color_mode = dlg.chem_label_colors
            self._chem_canvas.set_layout_mode(dlg.chem_layout)
            self._chem_canvas.request_render()
            self._update_quick_svg_widget()
            self._update_debug_menu()
            if self._canvas.molecule:
                self._legend.update_for(self._canvas.molecule, dlg._color_overrides)
            # Apply shortcut changes
            sc_primary, sc_alt = dlg.get_shortcut_overrides()
            self._apply_shortcuts(sc_primary)
            self._apply_alt_shortcuts(sc_alt)
            self._save_shortcut_overrides(sc_primary, sc_alt)
            self._canvas.request_render()
        else:
            (self._current_theme, self._canvas.background,
             self._canvas.atom_scale, self._canvas.bond_width_px,
             self._canvas.bond_style, self._color_overrides,
             self._canvas.atom_border_mode, self._canvas.atom_border_scale,
             self._canvas.atom_border_width, self._canvas.bond_color,
             self._canvas.lighting_intensity,
             self._canvas.light_position,
             self._canvas.roughness,
             self._canvas.show_axes,
             self._canvas.show_principal_axes,
             self._canvas.axes_position,
             self._canvas.principal_axes_position,
             _, self._canvas._fast_render, self._show_debug_menu,
             self._quick_svg_widget_enabled, self._start_view_pref) = orig
            self._chem_canvas.label_color_mode = orig_chem_labels
            self._chem_canvas.set_layout_mode(orig_chem_layout)
            self._chem_canvas.request_render()
            self._update_quick_svg_widget()
            self._update_debug_menu()
            self._canvas.color_overrides = self._color_overrides
            if self._canvas.molecule:
                self._legend.update_for(self._canvas.molecule, self._color_overrides)
            self._canvas.request_render()

    def _edit_style(self):
        self._edit_settings(tab_index=3)

    def _edit_overlay(self):
        self._edit_settings(tab_index=2)

    def _restart(self):
        self.close()
        if getattr(sys, "frozen", False):
            QProcess.startDetached(sys.executable, sys.argv[1:])
        else:
            QProcess.startDetached(sys.executable, [os.path.abspath(sys.argv[0])] + sys.argv[1:])

    def closeEvent(self, event):
        if self._optimize_worker is not None:
            # Destroying a running QThread aborts the process; let it finish.
            self._optimize_worker.wait()
        if self._canvas.molecule:
            try:
                c = self._canvas
                mol = c.molecule
                mol_data = {
                    "name": mol.name,
                    "charge": mol.charge,
                    "multiplicity": mol.multiplicity,
                    "atoms": [
                        {"element": a.element, "x": a.x, "y": a.y, "z": a.z, "color": a.color,
                         **({"charge": a.charge} if a.charge else {}),
                         **({"label_color": a.label_color} if a.label_color else {}),
                         **({"label_bold": a.label_bold} if a.label_bold is not None else {})}
                        for a in mol.atoms
                    ],
                    "bonds": [
                        {"i": b.i, "j": b.j, "order": b.order,
                         **({"stereo": b.stereo} if b.stereo else {})}
                        for b in mol.bonds
                    ],
                    "ring_bonds": mol.ring_bonds,
                }
                data = {
                    "molecule": mol_data,
                    "path": self._current_path,
                    "rot": c._rot.tolist(),
                    "zoom": c._zoom,
                    "pan": c._pan.tolist(),
                    "axes_ref": c._axes_ref.tolist(),
                    "default_view": c._default_view.tolist(),
                }
                paths.write_json(self.LAST_MOLECULE_FILE, data)
            except Exception:
                pass
        if getattr(self, "_quick_svg_tmp_path", None) and os.path.exists(self._quick_svg_tmp_path):
            try:
                os.unlink(self._quick_svg_tmp_path)
            except OSError:
                pass
        super().closeEvent(event)

    def _edit_atom_colors(self):
        mol = self._canvas.molecule
        orig_overrides = dict(self._color_overrides)

        def _live_update(overrides):
            self._color_overrides = overrides
            self._canvas.color_overrides = overrides
            if mol:
                self._legend.update_for(mol, overrides)
            self._canvas.request_render()

        dlg = AtomColorDialog(orig_overrides, live_callback=_live_update, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._color_overrides = dlg.get_overrides()
            self._canvas.color_overrides = self._color_overrides
            if mol:
                self._legend.update_for(mol, self._color_overrides)
            self._canvas.request_render()
        else:
            self._color_overrides = orig_overrides
            self._canvas.color_overrides = orig_overrides
            if mol:
                self._legend.update_for(mol, orig_overrides)
            self._canvas.request_render()

    def _reset_colors(self):
        saved = AtomColorDialog._load_defaults()
        self._color_overrides = saved
        self._canvas.color_overrides = saved
        if self._canvas.molecule:
            for atom in self._canvas.molecule.atoms:
                atom.color = None
            self._legend.update_for(self._canvas.molecule, saved)
        self._canvas.request_render()
        self._status.showMessage("Atom colours reset to CPK defaults.")

    def _paint_selected_atoms(self):
        mol = self._canvas.molecule
        if mol is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return
        canvas = self._chem_canvas if self._chem_mode else self._canvas
        sel = canvas.selected_atoms
        if not sel:
            QMessageBox.information(self, "No selection", "Select atoms first, then use Paint.")
            return
        col = QColorDialog.getColor(QColor("#ff6666"), self, "Pick Colour for Selected Atoms")
        if not col.isValid():
            return
        hex_col = col.name()
        if self._chem_mode:
            # 2D's own paint layer — separate from Atom.color.
            for idx in sel:
                self._chem_canvas.atom_paint[idx] = hex_col
            canvas.selected_atoms.clear()
            self._chem_canvas.request_render()
            self._status.showMessage(f"Painted {len(sel)} atom(s) with {hex_col} (2D only).")
            return
        for idx in sel:
            mol.atoms[idx].color = hex_col
        canvas.selected_atoms.clear()
        self._canvas.color_overrides = {}
        self._color_overrides = {}
        self._legend.update_for(mol, {})
        self._canvas.request_render()
        self._status.showMessage(f"Painted {len(sel)} atom(s) with {hex_col}.")

    def _clear_paint(self):
        mol = self._canvas.molecule
        if mol is None:
            QMessageBox.information(self, "No molecule", "Load or build a molecule first.")
            return
        if self._chem_mode:
            n = len(self._chem_canvas.atom_paint)
            if n == 0:
                self._status.showMessage("No painted atoms to clear.")
                return
            self._chem_canvas.atom_paint.clear()
            self._chem_canvas.request_render()
            self._status.showMessage(f"Cleared paint on {n} atom(s) (2D only).")
            return
        n = sum(1 for a in mol.atoms if a.color is not None)
        if n == 0:
            self._status.showMessage("No painted atoms to clear.")
            return
        for atom in mol.atoms:
            atom.color = None
        self._canvas.mulliken_legend = None
        self._canvas.mulliken_legend_in_export = False
        self._canvas._cached_grad_stops.clear()
        self._legend.update_for(mol, self._color_overrides)
        self._canvas.request_render()
        self._status.showMessage(f"Cleared paint on {n} atom(s).")

    # ── View actions ──────────────────────────────────────────────────────────

    def _pick_background(self):
        col = QColorDialog.getColor(
            QColor(self._canvas.background), self, "Pick Background Colour"
        )
        if col.isValid():
            self._canvas.background = col.name()
            self._canvas.request_render()

    # ── Help ──────────────────────────────────────────────────────────────────

    def _open_test_molecule(self):
        test_path = os.path.join(os.path.dirname(__file__), "test_files", "c5h6.xyz")
        if os.path.isfile(test_path):
            self._load_and_display(test_path)

    def _show_about(self):
        QMessageBox.about(self, "About Molvector",
            "<b>Molvector</b> — 3D Molecule Viewer<br><br>"
            "Ball-and-stick rendering, G16 input generator.<br>"
            "Parsers: XYZ · Gaussian input (.gjf/.com) · Gaussian log (.log/.out) · Mol file<br><br>"
            "Dependencies: PyQt6 · NumPy · svgwrite · molstrudel · RDKit · OpenBabel"
        )

    def _send_test_notification(self):
        types = [
            ("This is an info notification.", "info"),
            ("This is a success notification.", "success"),
            ("This is a warning notification.", "warning"),
            ("This is an error notification.", "error"),
        ]
        for i, (msg, ntype) in enumerate(types):
            QTimer.singleShot(i * 300, lambda m=msg, t=ntype: self.show_notification(m, t))

    def _update_debug_menu(self):
        mb = self.menuBar()
        if self._show_debug_menu:
            if not hasattr(self, '_debug_menu') or self._debug_menu is None:
                self._debug_menu = mb.addMenu("&Debug")
                act_test_notif = QAction("Send Test Notifications", self)
                act_test_notif.triggered.connect(self._send_test_notification)
                self._debug_menu.addAction(act_test_notif)
                self._debug_menu.addAction(self._act_open_test_folder)
        else:
            if hasattr(self, '_debug_menu') and self._debug_menu is not None:
                mb.removeAction(self._debug_menu.menuAction())
                self._debug_menu.deleteLater()
                self._debug_menu = None

    def show_notification(self, message: str, notification_type: str = "info", duration: int = None,
                          log_file: str = None):
        if duration is None:
            if notification_type == "error":
                duration = 0
            elif notification_type == "warning":
                duration = 6000
            else:
                duration = 4000
        button_text = None
        button_callback = None
        if log_file:
            display_msg = f"{message}\n{log_file}"
            button_text = "Open log"
            button_callback = lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(log_file))
        else:
            display_msg = message
        self._notification_container.add_notification(display_msg, notification_type, duration,
                                                     button_text=button_text, button_callback=button_callback)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_notification_container') and self._notification_container:
            self._notification_container._reposition()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonDblClick and obj is self._zoom_slider:
            self._zoom_slider.setValue(100)
            return True
        return super().eventFilter(obj, event)

    def _on_zoom_change(self, value):
        # One zoom for both views: the slider drives the 3D and 2D canvases alike.
        z = value / 100.0
        self._canvas._zoom = z
        self._chem_canvas._zoom = z
        self._zoom_lbl.setText(f"{value}%")
        self._canvas.note_zoom_activity()
        (self._chem_canvas if self._chem_mode else self._canvas).request_render(delay_ms=0)

    def _on_chem_zoom_changed(self, zoom: float):
        """The 2D view was zoomed — mirror it onto the 3D camera and the slider."""
        self._canvas._zoom = zoom
        self._canvas.note_zoom_activity()
        if not self._chem_mode:
            self._canvas.request_render(delay_ms=0)
        pct = int(round(zoom * 100))
        self._zoom_lbl.setText(f"{pct}%")
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(pct)
        self._zoom_slider.blockSignals(False)

    def _reset_view(self):
        if self._chem_mode:
            self._chem_canvas.reset_view()
            self._on_chem_zoom_changed(self._chem_canvas._zoom)
            return
        self._canvas.reset_view()
        self._zoom_slider.setValue(100)
        self._zoom_lbl.setText("100%")

    def _on_rotation_changed(self):
        self._chem_canvas._zoom = self._canvas._zoom  # linked zoom
        pct = int(self._canvas._zoom * 100)
        self._zoom_lbl.setText(f"{pct}%")
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(pct)
        self._zoom_slider.blockSignals(False)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    native_guard.enable_crash_traceback()
    app = QApplication(sys.argv)
    app.setApplicationName("Molvector")
    # Wayland ignores setWindowIcon and finds the icon via the .desktop file name.
    app.setDesktopFileName("molvector")
    # Keep the platform's own UI font ("sans-serif" is not a real family on
    # Windows/macOS and costs a slow alias lookup).  macOS keeps its native
    # 13pt; elsewhere 10pt, the size the layouts were tuned for.
    if platform.system() != "Darwin":
        font = app.font()
        font.setPointSize(10)
        app.setFont(font)

    if platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Molvector")
        except Exception:
            pass


    win = MainWindow()
    _ErrorReporter(win).install()
    set_icons(app, win)
    win.show()

    # Optional: open file from command line
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        win._load_and_display(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()