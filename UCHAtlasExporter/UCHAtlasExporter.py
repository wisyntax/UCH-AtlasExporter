"""
Krita plugin: UCH Atlas Exporter.

Installed via Krita's Python Plugin Manager. Adds "Export As UltimateOutfit
Atlas" under Tools > Scripts.

Requires the character document to be the active document -
exports whatever's currently visible on canvas

UI follows kritaSpritesheetManager's philosophy: sensible defaults, pressing
OK does a full correct export. Extra controls live behind "Use Custom Export
Settings".

Pipeline:
  1. Read AllCharacterAnimStates.json for frame names/count (browse dialog
     if missing). Character auto-detected from a matching group layer name,
     or picked manually.
  2. Capture each in-range frame's visible canvas content.
  3. Compute each frame's tight content bounding box (full canvas if Trim
     is off).
  4. Pack boxes into one atlas (Shelf, MaxRects, or Auto - tries both,
     keeps the smaller).
  5. Compute each frame's Sprite pivot as a fraction of its packed crop box.
  6. Render the atlas PNG and metadata.svg.
"""

import json
import math
import os
from pathlib import Path

from krita import Extension, Krita
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QImage, QPainter
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

# ---------------- CONFIGURATION ----------------
# AllCharacterAnimStates.json is bundled next to this file and resolved via
# __file__ (works because this is an installed module, not a Scripter
# script). Falls back to a browse dialog if the bundled copy is missing.
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_ANIM_STATES_JSON = os.path.join(PLUGIN_DIR, "AllCharacterAnimStates.json")

CANVAS_SIZE = 650  # fallback only - real runs use the active document's width
DEFAULT_ATLAS_PADDING = 1
DEFAULT_PIXELS_PER_UNIT = 210  # matches the game's standard sprite PPU
PLUGIN_VERSION = "1.0.1"  # bump on meaningful changes; shown in the dialog title
MAX_SANE_ATLAS_DIMENSION = 8192  # see pick_better_packing()
# ------------------------------------------------

# ---- session-persisted dialog settings ----
# In-memory dict (no disk I/O) that outlives any single dialog instance -
# persists across dialog opens until Krita restarts, or Reset is clicked.
# Exception: the character selection is tagged to the document it was
# picked for (see doc_identity_key()) and only reused for that same doc -
# switching documents re-detects instead of carrying the old pick over.
_SESSION_SETTINGS = {}
# ------------------------------------------------

# ---- shared with build_character_base.py: keeps frame-index alignment ----
def resolve_state_frames(state_entry, states_by_name, _chain=None):
    if state_entry.get("frames") is not None:
        return state_entry["frames"]
    same_as = state_entry.get("same_as")
    if not same_as:
        return []
    _chain = _chain or set()
    if same_as in _chain:
        return []
    _chain.add(state_entry["state"])
    target = states_by_name.get(same_as)
    if not target:
        return []
    frames = resolve_state_frames(target, states_by_name, _chain)
    return list(reversed(frames)) if state_entry.get("reversed") else list(frames)


def build_flat_frame_list(char_data):
    states_by_name = {s["state"]: s for s in char_data["states"]}
    flat = []
    for state_entry in char_data["states"]:
        flat.extend(resolve_state_frames(state_entry, states_by_name))
    for key in ("OK Cursor", "Bad Cursor", "Notebook Cursor", "Portrait"):
        name = char_data.get("cursor_portrait", {}).get(key)
        if name:
            flat.append(name)
    flat.extend(char_data.get("unused_frames", []))
    return flat


def build_alias_rects(char_data):
    aliases = {}
    for alias_map in char_data.get("frame_aliases", {}).values():
        for base_frame, real_frame in alias_map.items():
            aliases.setdefault(base_frame, []).append(real_frame)
    return aliases


def format_bytes(num_bytes):
    """Human-readable byte count (KB uses 1024, matching how Unity/most
    engines report texture memory)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0


def estimate_texture_ram_bytes(atlas_w, atlas_h):
    """Estimated Texture2D memory: width * height * 4 bytes (uncompressed
    RGBA32, no mipmaps - Unity's default Sprite import). RAM/VRAM cost, not
    on-disk PNG size; actual cost differs if import settings compress or
    mipmap."""
    return atlas_w * atlas_h * 4


def alpha_bbox(img):
    """Tight (x, y, w, h) bounding box of non-transparent pixels in img,
    or None if fully transparent."""
    w, h = img.width(), img.height()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    buf = bytes(ptr)
    stride = img.bytesPerLine()

    try:
        import numpy as np

        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, stride)[:, : w * 4]
        alpha = arr[:, 3::4]
        rows = np.any(alpha, axis=1)
        cols = np.any(alpha, axis=0)
        if not rows.any():
            return None
        y0, y1 = int(rows.argmax()), h - 1 - int(rows[::-1].argmax())
        x0, x1 = int(cols.argmax()), w - 1 - int(cols[::-1].argmax())
        return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    except ImportError:
        min_x, min_y, max_x, max_y = w, h, -1, -1
        for row in range(h):
            row_start = row * stride
            row_alpha = buf[row_start + 3 : row_start + w * 4 : 4]
            if any(row_alpha):
                min_y = row if min_y == h else min_y
                max_y = row
                for col, a in enumerate(row_alpha):
                    if a:
                        if col < min_x:
                            min_x = col
                        if col > max_x:
                            max_x = col
        if max_y == -1:
            return None
        return (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)


def shelf_pack_classic(boxes, padding=1):
    """Algorithm A: row-based shelf packer."""
    total_area = sum((w + padding) * (h + padding) for w, h in boxes.values())
    target_width = max(1, math.ceil(math.sqrt(total_area)))
    order = sorted(boxes.keys(), key=lambda i: boxes[i][1], reverse=True)

    positions = {}
    shelf_x = padding
    shelf_y = padding
    shelf_h = 0
    atlas_w = 0

    for i in order:
        w, h = boxes[i]
        if shelf_x > padding and shelf_x + w + padding > target_width:
            shelf_y += shelf_h + padding
            shelf_x = padding
            shelf_h = 0
        positions[i] = (shelf_x, shelf_y)
        shelf_x += w + padding
        shelf_h = max(shelf_h, h)
        atlas_w = max(atlas_w, shelf_x)

    return positions, atlas_w, shelf_y + shelf_h + padding


def shelf_pack_maxrects(boxes, padding, rule, order):
    """Free-rectangle ('MaxRects') packer; which free rect wins for each
    box is parameterized (see MAXRECTS_RULES/_maxrects_score).

    PADDING WARNING: fit-checks, placement, and free-rect splitting all use
    the box's PADDED footprint (w + padding, h + padding), never raw w/h -
    adjacent boxes touched with no gap when this didn't hold (real bug,
    now fixed). w_needed/h_needed below are that padded footprint. Kept in
    one function - only _maxrects_score() varies per rule - so every rule
    inherits the same padding-correct behavior.
    """
    total_area = sum((w + padding * 2) * (h + padding * 2) for w, h in boxes.values())
    target_width = max(
        max(b[0] for b in boxes.values()) + padding * 2,
        math.ceil(math.sqrt(total_area)),
    )

    max_used_w = 0
    max_used_h = 0
    free_rects = [{"x": padding, "y": padding, "w": target_width - padding * 2, "h": 1000000}]
    positions = {}

    for i in order:
        w, h = boxes[i]
        w_needed = w + padding
        h_needed = h + padding

        best_idx = -1
        best_score = None
        for idx, rect in enumerate(free_rects):
            score = _maxrects_score(rule, rect, w_needed, h_needed)
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_idx = idx

        if best_idx == -1:
            old_w = target_width
            target_width += w_needed
            free_rects.append({"x": old_w - padding, "y": padding, "w": w_needed, "h": 1000000})
            for idx, rect in enumerate(free_rects):
                score = _maxrects_score(rule, rect, w_needed, h_needed)
                if score is not None and (best_score is None or score < best_score):
                    best_score = score
                    best_idx = idx

        best_rect = free_rects[best_idx]
        px, py = best_rect["x"], best_rect["y"]
        positions[i] = (px, py)
        max_used_w = max(max_used_w, px + w + padding)
        max_used_h = max(max_used_h, py + h + padding)

        new_free_rects = []
        for rect in free_rects:
            rx, ry, rw, rh = rect["x"], rect["y"], rect["w"], rect["h"]
            if px < rx + rw and px + w_needed > rx and py < ry + rh and py + h_needed > ry:
                if px > rx:
                    new_free_rects.append({"x": rx, "y": ry, "w": px - rx, "h": rh})
                if px + w_needed < rx + rw:
                    new_free_rects.append(
                        {"x": px + w_needed, "y": ry, "w": (rx + rw) - (px + w_needed), "h": rh}
                    )
                if py > ry:
                    new_free_rects.append({"x": rx, "y": ry, "w": rw, "h": py - ry})
                if py + h_needed < ry + rh:
                    new_free_rects.append(
                        {"x": rx, "y": py + h_needed, "w": rw, "h": (ry + rh) - (py + h_needed)}
                    )
            else:
                new_free_rects.append(rect)

        free_rects = []
        for r in new_free_rects:
            contained = False
            for other in new_free_rects:
                if r is other:
                    continue
                if (
                    r["x"] >= other["x"]
                    and r["y"] >= other["y"]
                    and r["x"] + r["w"] <= other["x"] + other["w"]
                    and r["y"] + r["h"] <= other["y"] + other["h"]
                ):
                    contained = True
                    break
            if not contained:
                free_rects.append(r)

    return positions, max_used_w, max_used_h


# The four classic MaxRects placement rules (Jylanki, "A Thousand Ways to
# Pack the Bin"). Trying all four and keeping the smallest (see
# maxrects_pack_best) packs mixed box sizes tighter than hard-coding "bl".
MAXRECTS_RULES = ("bssf", "blsf", "baf", "bl")


def _maxrects_score(rule, rect, w_needed, h_needed):
    """Score for placing a w_needed x h_needed box (already padding-
    inclusive) into this free rect, per the chosen heuristic. Lower score
    wins, ties broken by the second tuple element. None if it doesn't fit."""
    if rect["w"] < w_needed or rect["h"] < h_needed:
        return None

    if rule == "bl":  # Bottom-Left: topmost, then leftmost
        return (rect["y"], rect["x"])

    leftover_w = rect["w"] - w_needed
    leftover_h = rect["h"] - h_needed
    short_leftover = min(leftover_w, leftover_h)
    long_leftover = max(leftover_w, leftover_h)

    if rule == "bssf":  # Best Short Side Fit: tightest remaining short gap
        return (short_leftover, long_leftover)
    if rule == "blsf":  # Best Long Side Fit: tightest remaining long gap
        return (long_leftover, short_leftover)
    if rule == "baf":  # Best Area Fit: smallest free rect that still fits
        return (rect["w"] * rect["h"], short_leftover)

    raise ValueError(f"Unknown MaxRects rule: {rule!r}")


def maxrects_pack_best(boxes, padding):
    """Runs the MaxRects packer once per rule in MAXRECTS_RULES, picks the
    winner via pick_better_packing() (the same "absurd strip" risk that
    guards against can show up between rules too). Returns (positions,
    atlas_w, atlas_h, winning_rule_name)."""
    order = sorted(boxes.keys(), key=lambda i: (boxes[i][1], boxes[i][0]), reverse=True)

    candidates = []
    for rule in MAXRECTS_RULES:
        positions, w, h = shelf_pack_maxrects(boxes, padding, rule, order)
        candidates.append((rule, positions, w, h))

    rule, positions, w, h = pick_better_packing(candidates)
    return positions, w, h, rule


def pick_better_packing(candidates, max_dim=MAX_SANE_ATLAS_DIMENSION):
    """Chooses between multiple (label, positions, w, h) packing results.

    Smallest-area alone can pick a technically-minimal but unusable
    layout, e.g. a 652x66403 strip past a GPU's max texture dimension
    despite equal pixel count to a roughly-square packing. So: among
    candidates under max_dim on both sides, pick smallest area; if none
    qualify, keep whichever has the smallest longer side.

    candidates: list of (label, positions, atlas_w, atlas_h) tuples.
    """
    within_limit = [c for c in candidates if max(c[2], c[3]) <= max_dim]
    if within_limit:
        return min(within_limit, key=lambda c: c[2] * c[3])
    # Nothing fits the safe limit - pick whichever is least oversized.
    fallback = min(candidates, key=lambda c: max(c[2], c[3]))
    print(
        f"[warn] every candidate atlas layout exceeds the {max_dim}px safe dimension "
        f"guideline - using {fallback[0]} ({fallback[2]}x{fallback[3]}) as the least-bad "
        f"option. Consider trimming, a smaller canvas, or exporting a frame subset."
    )
    return fallback


# --------------------------- export dialog ---------------------------


class TightAtlasExportDialog(QDialog):
    def __init__(self, all_characters, doc, parent=None):
        super().__init__(parent)
        self.all_characters = all_characters
        self.char_data_by_label = {c["label"]: c for c in all_characters}
        self.doc = doc
        self._auto_export_name = ""
        self._auto_atlas_name = ""
        self.detected_canvas_size = doc.width() if doc else CANVAS_SIZE

        self.setWindowTitle(f"UCH Atlas Exporter (v{PLUGIN_VERSION})")
        self.setMinimumSize(480, 120)

        outer = QVBoxLayout(self)
        top = QGridLayout()
        row = 0

        # --- Character ---
        # Reused only if the active doc matches the one it was picked for
        # (doc_identity_key()) - otherwise auto-detect fresh. Reset also
        # re-detects on demand.
        self.characterCombo = QComboBox()
        labels = [c["label"] for c in all_characters]
        current_doc_key = self.doc_identity_key(doc)
        # Also gates frame_start/frame_end restoration below - a freshly
        # auto-detected character (different/unseen document) always
        # starts at its full frame range rather than inheriting whatever
        # partial range was last set for a previous document/character.
        self._reused_session_character = (
            "character_label" in _SESSION_SETTINGS
            and current_doc_key is not None
            and _SESSION_SETTINGS.get("character_doc_key") == current_doc_key
        )
        if self._reused_session_character:
            initial_label = _SESSION_SETTINGS["character_label"]
        else:
            initial_label = self.detect_character_label(doc, all_characters)
        if initial_label and initial_label in labels:
            self.characterCombo.addItems(labels)
            self.characterCombo.setCurrentIndex(labels.index(initial_label))
        else:
            self.characterCombo.addItem("-- Select Character --")
            self.characterCombo.addItems(labels)
        self.characterCombo.currentIndexChanged.connect(self.on_character_changed)
        top.addWidget(QLabel("Character:"), row, 0)
        top.addWidget(self.characterCombo, row, 1)
        row += 1

        self.atlasName = QLineEdit()
        self.atlasName.setToolTip(
            "Filename for the exported atlas linked to the metadata.\n"
            "'.png' is added automatically if you leave it off."
        )
        top.addWidget(QLabel("Export atlas name:"), row, 0)
        top.addWidget(self.atlasName, row, 1)
        row += 1

        self.exportName = QLineEdit()
        self.exportName.setToolTip("Name of the export folder (holds override.png + metadata.svg)")
        top.addWidget(QLabel("Export folder name:"), row, 0)
        top.addWidget(self.exportName, row, 1)
        row += 1

        self.exportDirTx = QLineEdit()
        self.exportDirTx.setToolTip("Directory the export folder above will be created in.")
        top.addWidget(QLabel("Export directory:"), row, 0)
        top.addWidget(self.exportDirTx, row, 1)
        row += 1

        dirButtons = QHBoxLayout()
        self.exportDirButt = QPushButton("Change export directory")
        self.exportDirResetButt = QPushButton("Reset to current directory")
        self.exportDirResetButt.setToolTip("Reset export to the current .kra document's directory.")
        self.exportDirButt.clicked.connect(self.browse_export_dir)
        self.exportDirResetButt.clicked.connect(self.reset_export_dir)
        dirButtons.addWidget(self.exportDirButt)
        dirButtons.addWidget(self.exportDirResetButt)
        top.addLayout(dirButtons, row, 1)
        row += 1

        outer.addLayout(top)

        self.customSettings = QCheckBox("Use Custom Export Settings")
        self.customSettings.setChecked(_SESSION_SETTINGS.get("custom_settings", False))
        self.customSettings.stateChanged.connect(self.toggle_hideable)
        outer.addWidget(self.customSettings)

        self.hideableWidget = QFrame()
        self.hideableWidget.setFrameShape(QFrame.Panel)
        self.hideableWidget.setFrameShadow(QFrame.Sunken)
        hideLayout = QGridLayout(self.hideableWidget)
        hrow = 0

        self.canvasSizeSpin = QSpinBox()
        self.canvasSizeSpin.setRange(1, 100000)
        self.canvasSizeSpin.setValue(_SESSION_SETTINGS.get("canvas_size", self.detected_canvas_size))
        self.canvasSizeSpin.setToolTip(
            "Canvas size (px) each frame is captured at.\n"
            "Defaults to the document's own width - assumes a square canvas."
        )
        hideLayout.addWidget(QLabel("Canvas size:"), hrow, 0)
        hideLayout.addWidget(self.canvasSizeSpin, hrow, 1)
        hrow += 1

        self.paddingSpin = QSpinBox()
        self.paddingSpin.setRange(0, 64)
        self.paddingSpin.setValue(_SESSION_SETTINGS.get("padding", DEFAULT_ATLAS_PADDING))
        self.paddingSpin.setToolTip("Transparent margin (px) added around each frame when packing.")
        hideLayout.addWidget(QLabel("Padding:"), hrow, 0)
        hideLayout.addWidget(self.paddingSpin, hrow, 1)
        hrow += 1

        self.pixelsPerUnitSpin = QSpinBox()
        self.pixelsPerUnitSpin.setRange(1, 100000)
        self.pixelsPerUnitSpin.setValue(_SESSION_SETTINGS.get("pixels_per_unit", DEFAULT_PIXELS_PER_UNIT))
        self.pixelsPerUnitSpin.setToolTip(
            "pixelsPerUnit written into metadata.svg's <image> tag -\n"
            "sets the atlas's in-game scale.\n"
            f"Defaults to {DEFAULT_PIXELS_PER_UNIT}, the game's standard sprite PPU -\n"
            "only change it if this outfit needs a different scale."
        )
        hideLayout.addWidget(QLabel("Pixels per unit:"), hrow, 0)
        hideLayout.addWidget(self.pixelsPerUnitSpin, hrow, 1)
        hrow += 1

        self.packerCombo = QComboBox()
        self.packerCombo.addItems(["Auto", "Force Shelf", "Force MaxRects"])
        self.packerCombo.setCurrentText(_SESSION_SETTINGS.get("packer_mode", "Auto"))
        self.packerCombo.setToolTip(
            "Auto: tries both packers, keeps whichever atlas is smaller.\n"
            "Force Shelf / Force MaxRects: skips the comparison, uses only that packer."
        )
        hideLayout.addWidget(QLabel("Packer mode:"), hrow, 0)
        hideLayout.addWidget(self.packerCombo, hrow, 1)
        hrow += 1

        self.startSpin = QSpinBox()
        self.endSpin = QSpinBox()
        self.startSpin.setToolTip("First frame index (inclusive) to export")
        self.endSpin.setToolTip(
            "Last frame index (inclusive) to export.\n"
            "A partial range is preview/debug only - frames outside it\n"
            "won't appear in metadata.svg, so it's not a full replacement."
        )
        frameRange = QHBoxLayout()
        frameRange.addWidget(self.startSpin)
        frameRange.addWidget(QLabel("to"))
        frameRange.addWidget(self.endSpin)
        hideLayout.addWidget(QLabel("Frame range:"), hrow, 0)
        hideLayout.addLayout(frameRange, hrow, 1)
        hrow += 1

        self.trimCheck = QCheckBox("Trim each frame to its content")
        self.trimCheck.setChecked(_SESSION_SETTINGS.get("trim", True))
        self.trimCheck.setToolTip("Unchecked falls back to the old full-canvas-per-frame export.")
        hideLayout.addWidget(self.trimCheck, hrow, 0, 1, 2)
        hrow += 1

        self.skipAutoHideCheck = QCheckBox("Don't hide lines and notes layers")
        self.skipAutoHideCheck.setChecked(_SESSION_SETTINGS.get("skip_auto_hide", False))
        self.skipAutoHideCheck.setToolTip(
            "Unchecked (default): hides 'Lines' and any '_notes' layer while\n"
            "capturing, then restores them - keeps guides/notes out of the\n"
            "atlas even if they're visible in the working file.\n"
            "Checked: skips that and leaves visibility exactly as-is."
        )
        hideLayout.addWidget(self.skipAutoHideCheck, hrow, 0, 1, 2)
        hrow += 1

        self.noOverwriteCheck = QCheckBox("Don't overwrite existing files")
        self.noOverwriteCheck.setChecked(_SESSION_SETTINGS.get("no_overwrite", False))
        self.noOverwriteCheck.setToolTip(
            "Checked: if the export folder already exists, auto-suffix a new one (_2, _3, ...) instead.\n"
            "Unchecked: overwrite override.png/metadata.svg in the existing folder."
        )
        hideLayout.addWidget(self.noOverwriteCheck, hrow, 0, 1, 2)
        hrow += 1

        outer.addWidget(self.hideableWidget)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.try_accept)
        buttonBox.rejected.connect(self.reject)
        self.resetButt = QPushButton("Reset Settings")
        self.resetButt.setToolTip("Resets every setting in this dialog back to defaults.")
        self.resetButt.clicked.connect(self.reset_to_defaults)
        buttonBox.addButton(self.resetButt, QDialogButtonBox.ResetRole)
        outer.addWidget(buttonBox)

        # finished fires on both Accept and Cancel, so field values persist
        # even if the user backs out - matches kritaSpritesheetManager.
        self.finished.connect(self.save_session_settings)

        # Persisted export dir wins over the doc-folder default.
        if "export_dir" in _SESSION_SETTINGS:
            self.exportDirTx.setText(_SESSION_SETTINGS["export_dir"])
        else:
            self.reset_export_dir()
        self.toggle_hideable()
        # Populate defaults for the selected character (signal may not fire
        # if the index didn't change).
        self.on_character_changed(self.characterCombo.currentIndex())
        # Persisted frame range only carries over when reusing the same
        # character for the same document (_reused_session_character, set
        # above) - a freshly auto-detected character keeps the full range
        # on_character_changed just set, instead of inheriting a stale
        # partial range left over from a different document/character.
        if self._reused_session_character:
            if "frame_start" in _SESSION_SETTINGS:
                self.startSpin.setValue(max(self.startSpin.minimum(), min(_SESSION_SETTINGS["frame_start"], self.startSpin.maximum())))
            if "frame_end" in _SESSION_SETTINGS:
                self.endSpin.setValue(max(self.endSpin.minimum(), min(_SESSION_SETTINGS["frame_end"], self.endSpin.maximum())))

    @staticmethod
    def doc_identity_key(doc):
        """Best-effort identifier for 'is this the same document as last
        time'. Prefers the on-disk file path; falls back to the document's
        display name for an unsaved doc (Krita auto-increments
        "Untitled"/"Untitled-2"). None if there's no active document."""
        if not doc:
            return None
        return doc.fileName() or f"__unsaved__:{doc.name()}"

    @staticmethod
    def detect_character_label(doc, all_characters):
        """Scans the doc's top-level group layers for a name matching a
        character label in the JSON. Returns the match, or None."""
        if not doc:
            return None
        for child in doc.rootNode().childNodes():
            if child.type() == "grouplayer":
                for c in all_characters:
                    if c["label"] == child.name():
                        return c["label"]
        return None

    def reset_to_defaults(self):
        """Resets every persisted field in this open dialog to defaults,
        live, and re-runs character auto-detection immediately."""
        self.customSettings.setChecked(False)
        self.canvasSizeSpin.setValue(self.detected_canvas_size)
        self.paddingSpin.setValue(DEFAULT_ATLAS_PADDING)
        self.pixelsPerUnitSpin.setValue(DEFAULT_PIXELS_PER_UNIT)
        self.packerCombo.setCurrentText("Auto")
        self.trimCheck.setChecked(True)
        self.noOverwriteCheck.setChecked(False)
        self.skipAutoHideCheck.setChecked(False)
        self.reset_export_dir()

        labels = [c["label"] for c in self.all_characters]
        detected_label = self.detect_character_label(self.doc, self.all_characters)
        self.characterCombo.blockSignals(True)
        self.characterCombo.clear()
        if detected_label and detected_label in labels:
            self.characterCombo.addItems(labels)
            self.characterCombo.setCurrentIndex(labels.index(detected_label))
        else:
            self.characterCombo.addItem("-- Select Character --")
            self.characterCombo.addItems(labels)
        self.characterCombo.blockSignals(False)
        # blockSignals() above means on_character_changed didn't fire -
        # call it manually.
        self._auto_export_name = ""
        self._auto_atlas_name = ""
        self.atlasName.clear()
        self.exportName.clear()
        self.on_character_changed(self.characterCombo.currentIndex())

    def toggle_hideable(self):
        self.hideableWidget.setVisible(self.customSettings.isChecked())
        self.adjustSize()

    def browse_export_dir(self):
        chosen = QFileDialog.getExistingDirectory(self, "Choose Export Directory", self.exportDirTx.text())
        if chosen:
            self.exportDirTx.setText(chosen)

    def reset_export_dir(self):
        if self.doc and self.doc.fileName():
            self.exportDirTx.setText(str(Path(self.doc.fileName()).parent))
        else:
            self.exportDirTx.setText(os.path.expanduser("~"))

    def default_atlas_name(self, label):
        # Prefer the doc's .kra filename so saved variants never collide;
        # fall back to the character label if unsaved.
        if self.doc and self.doc.fileName():
            return Path(self.doc.fileName()).stem
        return label

    def default_export_folder_name(self, label):
        # Same pattern as default_atlas_name().
        if self.doc and self.doc.fileName():
            return f"{Path(self.doc.fileName()).stem} atlas export"
        return f"{label} atlas export"

    def on_character_changed(self, _index):
        label = self.characterCombo.currentText()
        if label not in self.char_data_by_label:
            return

        # Only auto-update if the user hasn't typed over the last auto-value.
        export_default = self.default_export_folder_name(label)
        if self.exportName.text() == self._auto_export_name:
            self.exportName.setText(export_default)
        self._auto_export_name = export_default

        atlas_default = self.default_atlas_name(label)
        if self.atlasName.text() == self._auto_atlas_name:
            self.atlasName.setText(atlas_default)
        self._auto_atlas_name = atlas_default

        n_frames = len(build_flat_frame_list(self.char_data_by_label[label]))
        for spin in (self.startSpin, self.endSpin):
            spin.setMinimum(0)
            spin.setMaximum(max(0, n_frames - 1))
        self.startSpin.setValue(0)
        self.endSpin.setValue(max(0, n_frames - 1))

    def try_accept(self):
        if self.characterCombo.currentText() not in self.char_data_by_label:
            QMessageBox.warning(self, "Export Atlas", "Pick a character first.")
            return
        if not self.exportDirTx.text().strip():
            QMessageBox.warning(self, "Export Atlas", "Choose an export directory first.")
            return
        self.accept()

    def save_session_settings(self, _result=None):
        """Persists the dialog's field values into _SESSION_SETTINGS.
        Hooked to QDialog.finished (fires on Accept and Cancel, so
        settings persist even if the user backs out)."""
        _SESSION_SETTINGS["character_label"] = self.characterCombo.currentText()
        _SESSION_SETTINGS["character_doc_key"] = self.doc_identity_key(self.doc)
        _SESSION_SETTINGS["custom_settings"] = self.customSettings.isChecked()
        _SESSION_SETTINGS["canvas_size"] = self.canvasSizeSpin.value()
        _SESSION_SETTINGS["padding"] = self.paddingSpin.value()
        _SESSION_SETTINGS["pixels_per_unit"] = self.pixelsPerUnitSpin.value()
        _SESSION_SETTINGS["packer_mode"] = self.packerCombo.currentText()
        _SESSION_SETTINGS["trim"] = self.trimCheck.isChecked()
        _SESSION_SETTINGS["no_overwrite"] = self.noOverwriteCheck.isChecked()
        _SESSION_SETTINGS["skip_auto_hide"] = self.skipAutoHideCheck.isChecked()
        _SESSION_SETTINGS["export_dir"] = self.exportDirTx.text().strip()
        _SESSION_SETTINGS["frame_start"] = self.startSpin.value()
        _SESSION_SETTINGS["frame_end"] = self.endSpin.value()

    def get_settings(self):
        label = self.characterCombo.currentText()
        custom = self.customSettings.isChecked()
        return {
            "label": label,
            "char_data": self.char_data_by_label[label],
            "export_name": self.exportName.text().strip() or self.default_export_folder_name(label),
            "export_dir": self.exportDirTx.text().strip(),
            "canvas_size": self.canvasSizeSpin.value() if custom else self.detected_canvas_size,
            "padding": self.paddingSpin.value() if custom else DEFAULT_ATLAS_PADDING,
            "pixels_per_unit": self.pixelsPerUnitSpin.value() if custom else DEFAULT_PIXELS_PER_UNIT,
            "packer_mode": self.packerCombo.currentText() if custom else "Auto",
            "frame_start": self.startSpin.value() if custom else 0,
            "frame_end": self.endSpin.value() if custom else (len(build_flat_frame_list(self.char_data_by_label[label])) - 1),
            "trim": self.trimCheck.isChecked() if custom else True,
            "no_overwrite": self.noOverwriteCheck.isChecked() if custom else False,
            "skip_auto_hide": self.skipAutoHideCheck.isChecked() if custom else False,
            "atlas_name": self.atlasName.text().strip() or self.default_atlas_name(label),
        }


# ------------------------------- export -------------------------------


def find_nodes_by_name(root_node, name, node_type=None):
    """Recursively finds every descendant node with an exact name match,
    optionally restricted to a Node.type() (e.g. "grouplayer")."""
    found = []
    for child in root_node.childNodes():
        if child.name() == name and (node_type is None or child.type() == node_type):
            found.append(child)
        found.extend(find_nodes_by_name(child, name, node_type))
    return found


def find_nodes_ending_with(root_node, suffix):
    """Recursively finds every descendant node whose name ends with suffix."""
    found = []
    for child in root_node.childNodes():
        if child.name().endswith(suffix):
            found.append(child)
        found.extend(find_nodes_ending_with(child, suffix))
    return found


def collect_auto_hide_nodes(doc):
    """Locates the "Lines" group and any "*_notes" layer(s) to hide before
    capture (see AUTO-HIDE in run_export). Finding neither isn't an error -
    the caller logs a warning instead."""
    lines_nodes = find_nodes_by_name(doc.rootNode(), "Lines", node_type="grouplayer")
    notes_nodes = find_nodes_ending_with(doc.rootNode(), "_notes")
    return lines_nodes, notes_nodes


def resolve_output_dir(save_root, export_name, no_overwrite):
    out_dir = os.path.join(save_root, export_name)
    if not no_overwrite or not os.path.exists(out_dir):
        return out_dir
    n = 2
    while True:
        candidate = os.path.join(save_root, f"{export_name}_{n}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def run_export(doc, settings):
    label = settings["label"]
    char_data = settings["char_data"]
    frame_names = build_flat_frame_list(char_data)
    alias_rects = build_alias_rects(char_data)

    start = max(0, min(settings["frame_start"], len(frame_names) - 1))
    end = max(start, min(settings["frame_end"], len(frame_names) - 1))
    frame_range = range(start, end + 1)
    is_subset = start != 0 or end != len(frame_names) - 1

    out_dir = resolve_output_dir(settings["export_dir"], settings["export_name"], settings["no_overwrite"])
    os.makedirs(out_dir, exist_ok=True)

    raw_atlas_name = settings["atlas_name"].strip() or label
    if raw_atlas_name.lower().endswith(".png"):
        raw_atlas_name = raw_atlas_name[: -len(".png")].strip() or label
    image_filename = raw_atlas_name + ".png"

    canvas_size = settings["canvas_size"]
    anchor_x = canvas_size / 2.0
    anchor_y = float(canvas_size)

    # AUTO-HIDE: hide "Lines"/"*_notes" before capture so neither leaks
    # into the atlas; restored in `finally` regardless of outcome. Skipped
    # if "Don't hide lines and notes layers" is checked.
    hidden_nodes = []
    if not settings.get("skip_auto_hide", False):
        lines_nodes, notes_nodes = collect_auto_hide_nodes(doc)
        if not lines_nodes:
            print("[warn] auto-hide: no 'Lines' group layer found - nothing to hide there.")
        if not notes_nodes:
            print("[warn] auto-hide: no '*_notes' layer found - nothing to hide there.")
        for node in lines_nodes + notes_nodes:
            if node.visible():
                hidden_nodes.append(node)
                node.setVisible(False)
        if hidden_nodes:
            doc.refreshProjection()

    # PHASE 1: capture each in-range frame and its content bounding box
    # (or full canvas if Trim is off).
    frame_images = {}
    frame_boxes = {}

    try:
        for frame_idx in frame_range:
            doc.setCurrentTime(frame_idx)
            doc.refreshProjection()

            pixel_data = doc.rootNode().projectionPixelData(0, 0, canvas_size, canvas_size)
            img = QImage(pixel_data, canvas_size, canvas_size, QImage.Format_RGBA8888).rgbSwapped()
            frame_images[frame_idx] = img

            if settings["trim"]:
                box = alpha_bbox(img)
                if box is None:
                    print(f"[warn] frame {frame_idx} ({frame_names[frame_idx]}) is fully transparent.")
                    box = (int(anchor_x), int(anchor_y) - 1, 1, 1)
            else:
                box = (0, 0, canvas_size, canvas_size)

            frame_boxes[frame_idx] = box
    finally:
        # Always restore visibility, even if capture raised partway through.
        for node in hidden_nodes:
            node.setVisible(True)
        if hidden_nodes:
            doc.refreshProjection()

    # PHASE 2: pack.
    box_sizes = {i: (b[2], b[3]) for i, b in frame_boxes.items()}
    padding = settings["padding"]
    mode = settings["packer_mode"]

    if mode == "Force Shelf":
        positions, atlas_w, atlas_h = shelf_pack_classic(box_sizes, padding)
        packer_used = "Shelf Packer"
        print(f"-> {packer_used} ({atlas_w}x{atlas_h})")
        if max(atlas_w, atlas_h) > MAX_SANE_ATLAS_DIMENSION:
            print(
                f"[warn] atlas exceeds the {MAX_SANE_ATLAS_DIMENSION}px safe dimension "
                f"guideline - a game engine may downscale or reject this texture."
            )
    elif mode == "Force MaxRects":
        positions, atlas_w, atlas_h, rule = maxrects_pack_best(box_sizes, padding)
        packer_used = f"MaxRects Packer [{rule}]"
        print(f"-> {packer_used} ({atlas_w}x{atlas_h})")
        if max(atlas_w, atlas_h) > MAX_SANE_ATLAS_DIMENSION:
            print(
                f"[warn] atlas exceeds the {MAX_SANE_ATLAS_DIMENSION}px safe dimension "
                f"guideline - a game engine may downscale or reject this texture."
            )
    else:
        pos_c, w_c, h_c = shelf_pack_classic(box_sizes, padding)
        pos_m, w_m, h_m, rule_m = maxrects_pack_best(box_sizes, padding)
        # Not a pure smallest-area pick - see pick_better_packing().
        packer_used, positions, atlas_w, atlas_h = pick_better_packing(
            [
                ("Shelf Packer", pos_c, w_c, h_c),
                (f"MaxRects Packer [{rule_m}]", pos_m, w_m, h_m),
            ]
        )
        print(f"-> {packer_used} ({atlas_w}x{atlas_h} = {atlas_w*atlas_h}px)")

    print(f"Packed {len(frame_range)} frame(s) into a {atlas_w}x{atlas_h} atlas.")
    if is_subset:
        print(
            f"[note] Exporting a subset (frames {start}-{end} of 0-{len(frame_names)-1}) - "
            f"this metadata.svg is a preview only, not a full replacement."
        )

    # PHASE 3: render the atlas and build metadata.svg.
    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        '<svg version="1.1" xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg" '
        'xmlns:svg="http://www.w3.org/2000/svg">',
    ]

    for frame_idx in frame_range:
        name = frame_names[frame_idx]
        crop_x, crop_y, crop_w, crop_h = frame_boxes[frame_idx]
        packed_x, packed_y = positions[frame_idx]

        pivot_x = (anchor_x - crop_x) / crop_w
        pivot_y = (crop_y + crop_h - anchor_y) / crop_h

        for rect_id in [name] + alias_rects.get(name, []):
            svg_lines.append(
                f'  <rect id="{rect_id}" width="{crop_w}" height="{crop_h}" '
                f'x="{packed_x}" y="{packed_y}" '
                f'offset-x="{pivot_x:.6f}" offset-y="{pivot_y:.6f}" />'
            )

    svg_lines.append(
        f'  <image xlink:href="{image_filename}" pixelsPerUnit="{settings["pixels_per_unit"]}" id="{label}" sodipodi:insensitive="true" />'
    )
    svg_lines.append("</svg>")
    metadata_svg = "\n".join(svg_lines)

    atlas_img = QImage(atlas_w, atlas_h, QImage.Format_RGBA8888)
    atlas_img.fill(0)
    painter = QPainter(atlas_img)
    for frame_idx in frame_range:
        crop_x, crop_y, crop_w, crop_h = frame_boxes[frame_idx]
        packed_x, packed_y = positions[frame_idx]
        crop = frame_images[frame_idx].copy(crop_x, crop_y, crop_w, crop_h)
        painter.drawImage(packed_x, packed_y, crop)
    painter.end()

    atlas_path = os.path.join(out_dir, image_filename)
    atlas_img.save(atlas_path)

    with open(os.path.join(out_dir, "metadata.svg"), "w", encoding="utf-8") as f:
        f.write(metadata_svg)

    ram_bytes = estimate_texture_ram_bytes(atlas_w, atlas_h)

    result_box = QMessageBox()
    result_box.setWindowTitle("Export Atlas")
    result_box.setIcon(QMessageBox.Information)
    result_box.setText(
        f"{label}: {len(frame_range)} frame(s) exported.\n"
        f"Atlas size: {atlas_w}x{atlas_h}\n"
        f"Packer used: {packer_used}\n"
        f"Estimated in-game texture memory: {format_bytes(ram_bytes)}\n"
        f"Saved to: {out_dir}"
    )
    open_folder_button = result_box.addButton("Open Export Folder", QMessageBox.ActionRole)
    result_box.addButton(QMessageBox.Ok)
    result_box.setDefaultButton(QMessageBox.Ok)
    result_box.exec_()
    if result_box.clickedButton() == open_folder_button:
        QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))

# ------------------------------- extension -------------------------------


class UCHAtlasExporterExtension(Extension):
    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        pass

    def createActions(self, window):
        action = window.createAction(
            "UCHAtlasExporter", "Export As UltimateOutfit Atlas", "tools/scripts"
        )
        action.triggered.connect(self.export_tight_atlas)

    def export_tight_atlas(self):
        anim_states_path = BUNDLED_ANIM_STATES_JSON
        if not os.path.isfile(anim_states_path):
            chosen, _ = QFileDialog.getOpenFileName(
                None,
                "Locate AllCharacterAnimStates.json",
                os.path.expanduser("~"),
                "JSON files (*.json)",
            )
            if not chosen:
                print("No AllCharacterAnimStates.json chosen - aborted.")
                return
            anim_states_path = chosen

        doc = Krita.instance().activeDocument()
        if not doc:
            QMessageBox.warning(
                None, "Export Atlas", "No active document - open the character's built .kra file first."
            )
            return

        with open(anim_states_path, encoding="utf-8") as f:
            all_characters = json.load(f)

        dialog = TightAtlasExportDialog(all_characters, doc)
        if dialog.exec_() != QDialog.Accepted:
            print("Cancelled.")
            return

        settings = dialog.get_settings()
        run_export(doc, settings)
