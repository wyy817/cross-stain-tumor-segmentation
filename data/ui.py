# viewer_team_ui.py
# PyQt5 UI for "Team(A/B) -> multiple Actors" workflow (2.0).
#
# Synced with NEW stage.py:
# - Camera.center (pixel center)
# - Actor.center(type='world'|'local') default 'world'
# - Actor has NO center_world()
# - Actor scaling is inside pose (fx, fy)
#
# UI behavior:
# - 1 team: force Merge + Move Camera only.
# - 2 teams: View(Merge/Split) independent from Move(Camera/Actor).
#
# Import as requested:
#   from stage import Camera, ColorTeam

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from stage import Camera, ColorTeam


# ============================================================
# Helpers
# ============================================================

def _is_mask_file(path: str) -> bool:
    p = path.lower()
    name = os.path.basename(p)
    if "mask" in name:
        return True
    ext = os.path.splitext(p)[1]
    if ext in (".kfb",):
        return True
    return False


def _infer_color_and_mask(paths: List[str]) -> Tuple[str, Optional[str]]:
    if not paths:
        raise ValueError("Empty selection")
    if len(paths) == 1:
        return paths[0], None

    a, b = paths[0], paths[1]
    a_is_mask = _is_mask_file(a)
    b_is_mask = _is_mask_file(b)

    if a_is_mask and not b_is_mask:
        return b, a
    if b_is_mask and not a_is_mask:
        return a, b

    return a, b


def _vp_default_path(color_path: str) -> str:
    base, _ = os.path.splitext(color_path)
    return base + ".vp"


def _actor_pose(actor) -> np.ndarray:
    if hasattr(actor, "pose"):
        return np.asarray(actor.pose(), dtype=np.float64)
    if hasattr(actor, "_pose"):
        return np.asarray(actor._pose, dtype=np.float64)
    return np.eye(3, dtype=np.float64)


def _actor_center_world(actor) -> Tuple[float, float]:
    if hasattr(actor, "center"):
        c = actor.center("world")
        return float(c[0]), float(c[1])
    return 0.0, 0.0


def _actor_center_local(actor) -> np.ndarray:
    if hasattr(actor, "center"):
        c = actor.center("local")
        return np.asarray(c, dtype=np.float64)
    return np.array([0.0, 0.0], dtype=np.float64)


def _pose_scale_xy(p: np.ndarray) -> Tuple[float, float]:
    # Column norms of 2x2 part approximate sx, sy even with rotation.
    A = np.asarray(p[:2, :2], dtype=np.float64)
    sx = float(np.linalg.norm(A[:, 0]))
    sy = float(np.linalg.norm(A[:, 1]))
    sx = max(sx, 1e-12)
    sy = max(sy, 1e-12)
    return sx, sy


def _pose_angle_deg(p: np.ndarray) -> float:
    A = np.asarray(p[:2, :2], dtype=np.float64)
    det = float(np.linalg.det(A))

    # If there's reflection (flip), remove it for angle display:
    # Our Apply uses sign on sx, so the first column direction flips => +180deg jump.
    # Fix: when det<0, flip col0 back before computing atan2.
    col0 = A[:, 0].copy()
    if det < 0:
        col0 *= -1.0

    ang = math.degrees(math.atan2(float(col0[1]), float(col0[0])))

    while ang <= -180:
        ang += 360
    while ang > 180:
        ang -= 360
    return ang


def _blend_rgba(images: List[np.ndarray]) -> np.ndarray:
    if not images:
        out = np.zeros((512, 512, 4), dtype=np.uint8)
        out[..., 3] = 255
        return out

    imgs = [im for im in images if im is not None and im.ndim == 3 and im.shape[-1] == 4]
    if not imgs:
        h, w = images[0].shape[:2]
        out = np.zeros((h, w, 4), dtype=np.uint8)
        out[..., 3] = 255
        return out

    h, w = imgs[0].shape[:2]
    alphas = np.stack([im[..., 3:4].astype(np.float32) for im in imgs], axis=0)  # (N,H,W,1)
    rgbs = np.stack([im[..., :3].astype(np.float32) for im in imgs], axis=0)     # (N,H,W,3)

    summary = alphas.sum(axis=0, keepdims=False)  # (H,W,1)
    bg = (summary <= 0).astype(np.float32)        # (H,W,1)

    denom = np.maximum(summary, 1e-6)
    weights = alphas / denom[None, ...]

    out_rgb = (weights * rgbs).sum(axis=0) + bg * 255.0
    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)

    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :3] = out_rgb
    out[..., 3] = np.where(bg[..., 0] > 0, 255, np.clip(summary[..., 0], 0, 255)).astype(np.uint8)
    return out


def _to_qimage_rgba(img_rgba: np.ndarray) -> QtGui.QImage:
    img = np.ascontiguousarray(img_rgba, dtype=np.uint8)
    h, w = img.shape[:2]
    bytes_per_line = 4 * w
    qimg = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGBA8888)
    return qimg.copy()


# ============================================================
# Canvas
# ============================================================

class Canvas(QtWidgets.QWidget):
    sig_mouse_press = QtCore.pyqtSignal(QtCore.QPoint, int, int)
    sig_mouse_release = QtCore.pyqtSignal(QtCore.QPoint, int, int)
    sig_mouse_move = QtCore.pyqtSignal(QtCore.QPoint, int, int)
    sig_wheel = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._qimg: Optional[QtGui.QImage] = None
        self._hint: str = ""

    def set_image(self, img_rgba: Optional[np.ndarray], hint: str = "") -> None:
        self._hint = hint or ""
        self._qimg = None if img_rgba is None else _to_qimage_rgba(img_rgba)
        self.update()

    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))
        if self._qimg is not None:
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            painter.drawImage(self.rect(), self._qimg)
        if self._hint:
            painter.setPen(QtGui.QColor(220, 220, 220))
            painter.setFont(QtGui.QFont("Consolas", 10))
            painter.drawText(10, 20, self._hint)
        painter.end()

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.sig_mouse_press.emit(ev.pos(), int(ev.button()), int(ev.modifiers()))

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.sig_mouse_release.emit(ev.pos(), int(ev.button()), int(ev.modifiers()))

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.sig_mouse_move.emit(ev.pos(), int(ev.buttons()), int(ev.modifiers()))

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        self.sig_wheel.emit(int(ev.angleDelta().y()), int(ev.modifiers()))


# ============================================================
# Team state
# ============================================================

@dataclass
class TeamState:
    team_id: str
    color_path: str
    mask_path: Optional[str]
    team: ColorTeam
    vp_path: str
    visible: Dict[int, bool]


# ============================================================
# Main window
# ============================================================

class TeamViewer(QtWidgets.QMainWindow):
    MODE_MERGE = "merge"
    MODE_SPLIT = "split"

    MOVE_CAMERA = "move_camera"
    MOVE_ACTOR = "move_actor"

    RIGHT_DRAG_THRESHOLD_PX = 6

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Team Viewer (2.0 UI)")

        self.camera = Camera(x=0.0, y=0.0, z=200.0, w=1024, h=1024)

        self.teamA: Optional[TeamState] = None
        self.teamB: Optional[TeamState] = None

        self.view_mode = self.MODE_MERGE
        self.move_mode = self.MOVE_CAMERA

        self.selected_actor: Optional[Tuple[str, int]] = None  # (team_id, actor_index)

        self._dragging = False
        self._drag_button: Optional[int] = None
        self._last_pos: Optional[QtCore.QPoint] = None
        self._right_press_pos: Optional[QtCore.QPoint] = None

        self._build_ui()
        self._wire_events()

        self._sync_enable_states()
        self.refresh()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        root = QtWidgets.QHBoxLayout(central)

        panel = QtWidgets.QVBoxLayout()
        root.addLayout(panel, 0)

        self.gbA = QtWidgets.QGroupBox("Team A")
        self.gbB = QtWidgets.QGroupBox("Team B")
        panel.addWidget(self.gbA)
        panel.addWidget(self.gbB)

        self._build_team_box(self.gbA, "A")
        self._build_team_box(self.gbB, "B")

        gb_view = QtWidgets.QGroupBox("View Mode")
        panel.addWidget(gb_view)
        view_lay = QtWidgets.QVBoxLayout(gb_view)
        self.rb_merge = QtWidgets.QRadioButton("Merge (default)")
        self.rb_split = QtWidgets.QRadioButton("Split (Left=A, Right=B)")
        self.rb_merge.setChecked(True)
        view_lay.addWidget(self.rb_merge)
        view_lay.addWidget(self.rb_split)

        gb_move = QtWidgets.QGroupBox("Interaction Mode")
        panel.addWidget(gb_move)
        move_lay = QtWidgets.QVBoxLayout(gb_move)
        self.rb_move_camera = QtWidgets.QRadioButton("Move Camera (pan/zoom)")
        self.rb_move_actor = QtWidgets.QRadioButton("Move Actor (drag/rotate/scale)")
        self.rb_move_camera.setChecked(True)
        move_lay.addWidget(self.rb_move_camera)
        move_lay.addWidget(self.rb_move_actor)

        # Two independent radio groups
        self.bg_view = QtWidgets.QButtonGroup(self)
        self.bg_view.setExclusive(True)
        self.bg_view.addButton(self.rb_merge)
        self.bg_view.addButton(self.rb_split)

        self.bg_move = QtWidgets.QButtonGroup(self)
        self.bg_move.setExclusive(True)
        self.bg_move.addButton(self.rb_move_camera)
        self.bg_move.addButton(self.rb_move_actor)

        gb_pose = QtWidgets.QGroupBox("Selected Actor Pose")
        panel.addWidget(gb_pose)
        play = QtWidgets.QGridLayout(gb_pose)

        self.ed_x = QtWidgets.QDoubleSpinBox()
        self.ed_y = QtWidgets.QDoubleSpinBox()
        self.ed_sx = QtWidgets.QDoubleSpinBox()
        self.ed_sy = QtWidgets.QDoubleSpinBox()
        self.ed_ang = QtWidgets.QDoubleSpinBox()

        for w in (self.ed_x, self.ed_y, self.ed_sx, self.ed_sy, self.ed_ang):
            w.setDecimals(6)
            w.setRange(-1e12, 1e12)

        self.ed_sx.setRange(1e-6, 1e12)
        self.ed_sy.setRange(1e-6, 1e12)
        self.ed_ang.setRange(-180.0, 180.0)

        play.addWidget(QtWidgets.QLabel("x"), 0, 0)
        play.addWidget(self.ed_x, 0, 1)
        play.addWidget(QtWidgets.QLabel("y"), 1, 0)
        play.addWidget(self.ed_y, 1, 1)
        play.addWidget(QtWidgets.QLabel("sx"), 2, 0)
        play.addWidget(self.ed_sx, 2, 1)
        play.addWidget(QtWidgets.QLabel("sy"), 3, 0)
        play.addWidget(self.ed_sy, 3, 1)
        play.addWidget(QtWidgets.QLabel("angle(deg)"), 4, 0)
        play.addWidget(self.ed_ang, 4, 1)

        self.btn_apply_pose = QtWidgets.QPushButton("Apply Pose")
        self.btn_reset_actor = QtWidgets.QPushButton("Reset Actor")
        self.btn_flip_lr = QtWidgets.QPushButton("Flip LR")
        self.btn_flip_ud = QtWidgets.QPushButton("Flip UD")
        self.btn_reset_cam = QtWidgets.QPushButton("Reset Camera")

        play.addWidget(self.btn_apply_pose, 5, 0, 1, 2)
        play.addWidget(self.btn_reset_actor, 6, 0, 1, 2)
        play.addWidget(self.btn_flip_lr, 7, 0, 1, 1)
        play.addWidget(self.btn_flip_ud, 7, 1, 1, 1)
        play.addWidget(self.btn_reset_cam, 8, 0, 1, 2)
        play.setRowStretch(9, 1)

        panel.addStretch(1)

        canvas_col = QtWidgets.QVBoxLayout()
        root.addLayout(canvas_col, 1)

        self.canvas_main = Canvas()
        self.canvas_main.setMinimumSize(900, 650)
        canvas_col.addWidget(self.canvas_main, 1)

        self.canvas_hint = QtWidgets.QLabel("")
        self.canvas_hint.setStyleSheet("color:#ddd; padding:4px;")
        canvas_col.addWidget(self.canvas_hint, 0)

    def _build_team_box(self, gb: QtWidgets.QGroupBox, team_id: str) -> None:
        lay = QtWidgets.QVBoxLayout(gb)

        row = QtWidgets.QHBoxLayout()
        lay.addLayout(row)

        btn_open = QtWidgets.QPushButton("Open Team")
        btn_close = QtWidgets.QPushButton("Close Team")
        btn_load = QtWidgets.QPushButton("Load vp")
        btn_save = QtWidgets.QPushButton("Save vp")

        row.addWidget(btn_open)
        row.addWidget(btn_close)
        row.addWidget(btn_load)
        row.addWidget(btn_save)

        lab = QtWidgets.QLabel("(empty)")
        lab.setWordWrap(True)
        lay.addWidget(lab)

        lst = QtWidgets.QListWidget()
        lst.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        lay.addWidget(lst)

        if team_id == "A":
            self.btn_open_A = btn_open
            self.btn_close_A = btn_close
            self.btn_load_A = btn_load
            self.btn_save_A = btn_save
            self.lab_A = lab
            self.list_A = lst
        else:
            self.btn_open_B = btn_open
            self.btn_close_B = btn_close
            self.btn_load_B = btn_load
            self.btn_save_B = btn_save
            self.lab_B = lab
            self.list_B = lst

    def _wire_events(self) -> None:
        self.btn_open_A.clicked.connect(lambda: self.open_team("A"))
        self.btn_close_A.clicked.connect(lambda: self.close_team("A"))
        self.btn_load_A.clicked.connect(lambda: self.load_vp("A"))
        self.btn_save_A.clicked.connect(lambda: self.save_vp("A"))

        self.btn_open_B.clicked.connect(lambda: self.open_team("B"))
        self.btn_close_B.clicked.connect(lambda: self.close_team("B"))
        self.btn_load_B.clicked.connect(lambda: self.load_vp("B"))
        self.btn_save_B.clicked.connect(lambda: self.save_vp("B"))

        self.list_A.itemChanged.connect(lambda it: self._on_item_check_changed("A", it))
        self.list_B.itemChanged.connect(lambda it: self._on_item_check_changed("B", it))
        self.list_A.currentRowChanged.connect(lambda row: self._on_select_actor("A", row))
        self.list_B.currentRowChanged.connect(lambda row: self._on_select_actor("B", row))

        self.rb_merge.toggled.connect(self._on_view_mode_changed)
        self.rb_split.toggled.connect(self._on_view_mode_changed)
        self.rb_move_camera.toggled.connect(self._on_move_mode_changed)
        self.rb_move_actor.toggled.connect(self._on_move_mode_changed)

        self.btn_apply_pose.clicked.connect(self.apply_pose_from_ui)
        self.btn_reset_actor.clicked.connect(self.reset_selected_actor)
        self.btn_flip_lr.clicked.connect(lambda: self.flip_selected("lr"))
        self.btn_flip_ud.clicked.connect(lambda: self.flip_selected("ud"))
        self.btn_reset_cam.clicked.connect(self.reset_camera)

        self.canvas_main.sig_mouse_press.connect(self.on_mouse_press)
        self.canvas_main.sig_mouse_release.connect(self.on_mouse_release)
        self.canvas_main.sig_mouse_move.connect(self.on_mouse_move)
        self.canvas_main.sig_wheel.connect(self.on_wheel)

    # ---------------- state ----------------

    def _team_count(self) -> int:
        return int(self.teamA is not None) + int(self.teamB is not None)

    def _two_teams_ready(self) -> bool:
        return self.teamA is not None and self.teamB is not None

    def _get_state(self, team_id: str) -> Optional[TeamState]:
        return self.teamA if team_id == "A" else self.teamB

    def _team_center(self, st: TeamState) -> Tuple[float, float]:
        actors = getattr(st.team, "_actors", [])
        if not actors:
            return 0.0, 0.0
        return _actor_center_world(actors[0])

    def _get_selected_actor_obj(self):
        if self.selected_actor is None:
            return None
        tid, idx = self.selected_actor
        st = self._get_state(tid)
        if st is None:
            return None
        actors = getattr(st.team, "_actors", [])
        if idx < 0 or idx >= len(actors):
            return None
        return actors[idx]

    # ---------------- team ops ----------------

    def open_team(self, team_id: str) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            f"Open Team {team_id}: select 1 file (color) or 2 files (color+mask)",
            "",
            "WSI/Mask Files (*.mrxs *.svs *.tif *.tiff *.kfb *.png *.jpg *.jpeg *.bmp);;All Files (*.*)",
        )
        if not paths:
            return
        if len(paths) > 2:
            paths = paths[:2]

        color_path, mask_path = _infer_color_and_mask(paths)
        color_path = os.path.abspath(color_path)
        mask_path = os.path.abspath(mask_path) if mask_path else None
        vp_path = _vp_default_path(color_path)

        try:
            team = ColorTeam(color_path, mask_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open Team Failed", f"{e}")
            return

        st = TeamState(
            team_id=team_id,
            color_path=color_path,
            mask_path=mask_path,
            team=team,
            vp_path=vp_path,
            visible={},
        )

        actors = getattr(team, "_actors", [])
        for i in range(len(actors)):
            st.visible[i] = True

        if os.path.exists(vp_path):
            try:
                team.load(vp_path)
            except Exception:
                pass

        if team_id == "A":
            self.teamA = st
        else:
            self.teamB = st

        if self._team_count() == 1:
            cx, cy = self._team_center(st)
            self.camera.move(cx, cy, max(float(self.camera.z), 200.0))

        self._rebuild_actor_lists()
        self._sync_enable_states()
        self.refresh()

    def close_team(self, team_id: str) -> None:
        st = self._get_state(team_id)
        if st is None:
            return
        try:
            st.team.save(st.vp_path)
        except Exception:
            pass

        if team_id == "A":
            self.teamA = None
            self.list_A.clear()
            self.lab_A.setText("(empty)")
        else:
            self.teamB = None
            self.list_B.clear()
            self.lab_B.setText("(empty)")

        if self.selected_actor is not None and self.selected_actor[0] == team_id:
            self.selected_actor = None

        self._sync_enable_states()
        self.refresh()

    def save_vp(self, team_id: str) -> None:
        st = self._get_state(team_id)
        if st is None:
            return
        try:
            st.team.save(st.vp_path)
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved vp:\n{st.vp_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Failed", f"{e}")

    def load_vp(self, team_id: str) -> None:
        st = self._get_state(team_id)
        if st is None:
            return
        vp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"Load vp for Team {team_id}",
            os.path.dirname(st.vp_path),
            "VP Files (*.vp *.json);;All Files (*.*)",
        )
        if not vp:
            return
        try:
            st.team.load(vp)
            st.vp_path = vp
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Failed", f"{e}")
            return

        self._rebuild_actor_lists()
        self._sync_enable_states()
        self.refresh()

    # ---------------- lists ----------------

    def _team_label_text(self, st: TeamState) -> str:
        mask_str = "None" if st.mask_path is None else os.path.basename(st.mask_path)
        return f"color: {os.path.basename(st.color_path)}\nmask: {mask_str}\nvp: {os.path.basename(st.vp_path)}"

    def _fill_list(self, lst: QtWidgets.QListWidget, st: TeamState) -> None:
        lst.blockSignals(True)
        lst.clear()
        actors = getattr(st.team, "_actors", [])
        for i in range(len(actors)):
            item = QtWidgets.QListWidgetItem(f"Actor {i}")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if st.visible.get(i, True) else QtCore.Qt.Unchecked)
            lst.addItem(item)
        lst.blockSignals(False)

    def _rebuild_actor_lists(self) -> None:
        if self.teamA is not None:
            self.lab_A.setText(self._team_label_text(self.teamA))
            self._fill_list(self.list_A, self.teamA)
        else:
            self.lab_A.setText("(empty)")
            self.list_A.clear()

        if self.teamB is not None:
            self.lab_B.setText(self._team_label_text(self.teamB))
            self._fill_list(self.list_B, self.teamB)
        else:
            self.lab_B.setText("(empty)")
            self.list_B.clear()

        self._sync_pose_ui_from_selected()

    def _on_item_check_changed(self, team_id: str, item: QtWidgets.QListWidgetItem) -> None:
        st = self._get_state(team_id)
        if st is None:
            return
        row = self.list_A.row(item) if team_id == "A" else self.list_B.row(item)
        st.visible[row] = (item.checkState() == QtCore.Qt.Checked)
        self.refresh()

    def _on_select_actor(self, team_id: str, row: int) -> None:
        if team_id == "A":
            if self.list_B.currentRow() >= 0:
                self.list_B.blockSignals(True)
                self.list_B.setCurrentRow(-1)
                self.list_B.blockSignals(False)
        else:
            if self.list_A.currentRow() >= 0:
                self.list_A.blockSignals(True)
                self.list_A.setCurrentRow(-1)
                self.list_A.blockSignals(False)

        if row < 0:
            if self.selected_actor is not None and self.selected_actor[0] == team_id:
                self.selected_actor = None
        else:
            self.selected_actor = (team_id, row)

        self._sync_enable_states()
        self._sync_pose_ui_from_selected()
        self.refresh()

    # ---------------- view/move modes ----------------

    def _on_view_mode_changed(self) -> None:
        if self._two_teams_ready() and self.rb_split.isChecked():
            self.view_mode = self.MODE_SPLIT
        else:
            self.view_mode = self.MODE_MERGE
        self.refresh()

    def _on_move_mode_changed(self) -> None:
        if self._two_teams_ready() and self.rb_move_actor.isChecked():
            self.move_mode = self.MOVE_ACTOR
        else:
            self.move_mode = self.MOVE_CAMERA
        self._sync_enable_states()
        self.refresh()

    def _sync_enable_states(self) -> None:
        two = self._two_teams_ready()

        if not two:
            # force Merge + Move Camera
            self.rb_split.setEnabled(False)
            self.rb_merge.setEnabled(True)
            if not self.rb_merge.isChecked():
                self.rb_merge.setChecked(True)
            self.view_mode = self.MODE_MERGE

            self.rb_move_actor.setEnabled(False)
            self.rb_move_camera.setEnabled(True)
            if not self.rb_move_camera.isChecked():
                self.rb_move_camera.setChecked(True)
            self.move_mode = self.MOVE_CAMERA
        else:
            self.rb_merge.setEnabled(True)
            self.rb_split.setEnabled(True)
            self.rb_move_camera.setEnabled(True)
            self.rb_move_actor.setEnabled(True)

        has_sel = (self._get_selected_actor_obj() is not None) and two
        for w in (
            self.ed_x, self.ed_y, self.ed_sx, self.ed_sy, self.ed_ang,
            self.btn_apply_pose, self.btn_reset_actor, self.btn_flip_lr, self.btn_flip_ud
        ):
            w.setEnabled(has_sel)

        self.btn_reset_cam.setEnabled(self._team_count() >= 1)

    # ---------------- pose UI ----------------

    def _sync_pose_ui_from_selected(self) -> None:
        a = self._get_selected_actor_obj()
        if a is None:
            self.ed_x.setValue(0.0)
            self.ed_y.setValue(0.0)
            self.ed_sx.setValue(1.0)
            self.ed_sy.setValue(1.0)
            self.ed_ang.setValue(0.0)
            return

        p = _actor_pose(a)
        cx, cy = _actor_center_world(a)
        sx, sy = _pose_scale_xy(p)
        ang = _pose_angle_deg(p)

        self.ed_x.setValue(cx)
        self.ed_y.setValue(cy)
        self.ed_sx.setValue(sx)
        self.ed_sy.setValue(sy)
        self.ed_ang.setValue(ang)

    def apply_pose_from_ui(self) -> None:
        """
        IMPORTANT FIX:
          - Do NOT apply via incremental rotate/scale (ambiguous decomposition).
          - Directly rebuild pose from (x,y,sx,sy,angle) using the actor local center as pivot.
          - This guarantees the UI sx/sy values will not "auto change" after Apply.
        """
        if not self._two_teams_ready():
            return
        a = self._get_selected_actor_obj()
        if a is None or not hasattr(a, "set"):
            return

        target_x = float(self.ed_x.value())
        target_y = float(self.ed_y.value())
        target_sx = float(self.ed_sx.value())
        target_sy = float(self.ed_sy.value())
        target_ang = float(self.ed_ang.value())

        # Preserve reflection (flip) if current pose has negative determinant
        p_cur = _actor_pose(a)
        A_cur = np.asarray(p_cur[:2, :2], dtype=np.float64)
        det = float(np.linalg.det(A_cur))
        sign = -1.0 if det < 0 else 1.0

        # Build rotation
        th = math.radians(target_ang)
        c = math.cos(th)
        s = math.sin(th)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)

        # Build scale with preserved reflection on x axis
        S = np.array([[sign * target_sx, 0.0], [0.0, target_sy]], dtype=np.float64)
        A_new = R @ S

        # Make local center map to target world center
        c_local = _actor_center_local(a)  # (2,)
        t_new = np.array([target_x, target_y], dtype=np.float64) - (A_new @ c_local)

        pose = np.eye(3, dtype=np.float64)
        pose[:2, :2] = A_new
        pose[:2, 2] = t_new

        a.set(pose)
        self._sync_pose_ui_from_selected()
        self.refresh()

    def reset_selected_actor(self) -> None:
        if not self._two_teams_ready():
            return
        a = self._get_selected_actor_obj()
        if a is None:
            return
        if hasattr(a, "reset"):
            a.reset()
        self._sync_pose_ui_from_selected()
        self.refresh()

    def flip_selected(self, flip_type: str) -> None:
        if not self._two_teams_ready():
            return
        a = self._get_selected_actor_obj()
        if a is None:
            return
        if hasattr(a, "flip"):
            a.flip(flip_type)
        self._sync_pose_ui_from_selected()
        self.refresh()

    def reset_camera(self) -> None:
        st = self.teamA if self.teamA is not None else self.teamB
        if st is None:
            return
        cx, cy = self._team_center(st)
        self.camera.move(cx, cy, max(float(self.camera.z), 200.0))
        self.refresh()

    # ---------------- rendering ----------------

    def _render_merge(self, cam_w: int, cam_h: int) -> np.ndarray:
        self.camera.set(cam_w, cam_h)
        imgs: List[np.ndarray] = []

        for st in (self.teamA, self.teamB):
            if st is None:
                continue
            actors = getattr(st.team, "_actors", [])
            for i, act in enumerate(actors):
                if not st.visible.get(i, True):
                    continue
                if hasattr(act, "render"):
                    try:
                        im = act.render(self.camera)
                        if im is not None and im.ndim == 3 and im.shape[-1] == 4:
                            imgs.append(im.astype(np.uint8))
                    except Exception:
                        continue

        if not imgs:
            out = np.zeros((cam_h, cam_w, 4), dtype=np.uint8)
            out[..., 3] = 255
            return out

        return _blend_rgba(imgs)

    def _render_split(self, cam_w: int, cam_h: int) -> np.ndarray:
        half_w = max(1, cam_w // 2)

        # Left: Team A
        self.camera.set(half_w, cam_h)
        left_imgs: List[np.ndarray] = []
        if self.teamA is not None:
            actors = getattr(self.teamA.team, "_actors", [])
            for i, act in enumerate(actors):
                if not self.teamA.visible.get(i, True):
                    continue
                if hasattr(act, "render"):
                    try:
                        im = act.render(self.camera)
                        if im is not None and im.ndim == 3 and im.shape[-1] == 4:
                            left_imgs.append(im.astype(np.uint8))
                    except Exception:
                        pass
        left = _blend_rgba(left_imgs) if left_imgs else np.zeros((cam_h, half_w, 4), dtype=np.uint8)
        left[..., 3] = np.where(left[..., 3] == 0, 255, left[..., 3]).astype(np.uint8)

        # Right: Team B
        self.camera.set(cam_w - half_w, cam_h)
        right_imgs: List[np.ndarray] = []
        if self.teamB is not None:
            actors = getattr(self.teamB.team, "_actors", [])
            for i, act in enumerate(actors):
                if not self.teamB.visible.get(i, True):
                    continue
                if hasattr(act, "render"):
                    try:
                        im = act.render(self.camera)
                        if im is not None and im.ndim == 3 and im.shape[-1] == 4:
                            right_imgs.append(im.astype(np.uint8))
                    except Exception:
                        pass
        right = _blend_rgba(right_imgs) if right_imgs else np.zeros((cam_h, cam_w - half_w, 4), dtype=np.uint8)
        right[..., 3] = np.where(right[..., 3] == 0, 255, right[..., 3]).astype(np.uint8)

        return np.concatenate([left, right], axis=1)

    def refresh(self) -> None:
        w = max(1, self.canvas_main.width())
        h = max(1, self.canvas_main.height())

        if self._team_count() == 0:
            img = np.zeros((h, w, 4), dtype=np.uint8)
            img[..., 3] = 255
            self.canvas_main.set_image(img, hint="Open Team A / Team B")
            self.canvas_hint.setText("")
            return

        if self.view_mode == self.MODE_SPLIT and self._two_teams_ready():
            img = self._render_split(w, h)
        else:
            img = self._render_merge(w, h)

        hint = f"view={self.view_mode}, move={self.move_mode} | cam=({self.camera.x:.2f},{self.camera.y:.2f}, z={self.camera.z:.2f})"
        if self.selected_actor is not None:
            a = self._get_selected_actor_obj()
            if a is not None:
                p = _actor_pose(a)
                cx, cy = _actor_center_world(a)
                sx, sy = _pose_scale_xy(p)
                ang = _pose_angle_deg(p)
                hint += f" | sel={self.selected_actor[0]}:{self.selected_actor[1]} center=({cx:.2f},{cy:.2f}) s=({sx:.4f},{sy:.4f}) ang={ang:.2f}"

        self.canvas_main.set_image(img, hint=hint)

        if self._two_teams_ready():
            self.canvas_hint.setText("Left-drag: pan camera OR move selected actor (by mode). Right-drag (actor mode): rotate. Wheel: zoom camera OR scale actor.")
        else:
            self.canvas_hint.setText("Single team: camera-only. Left-drag pan, wheel zoom.")

        self.camera.set(w, h)

    # ---------------- input handling ----------------

    def on_mouse_press(self, pos: QtCore.QPoint, button: int, mods: int) -> None:
        if self._team_count() == 0:
            return
        self._dragging = True
        self._drag_button = button
        self._last_pos = pos
        self._right_press_pos = pos if button == int(QtCore.Qt.RightButton) else None

    def on_mouse_release(self, pos: QtCore.QPoint, button: int, mods: int) -> None:
        self._dragging = False
        self._drag_button = None
        self._last_pos = None
        self._right_press_pos = None

    def on_mouse_move(self, pos: QtCore.QPoint, buttons: int, mods: int) -> None:
        if not self._dragging or self._last_pos is None:
            return

        dx = float(pos.x() - self._last_pos.x())
        dy = float(pos.y() - self._last_pos.y())
        self._last_pos = pos

        # Single team => camera only
        if self._team_count() == 1:
            self._pan_camera(dx, dy)
            self.refresh()
            return

        # Two teams
        if self._drag_button == int(QtCore.Qt.RightButton):
            if self.move_mode != self.MOVE_ACTOR:
                return
            a = self._get_selected_actor_obj()
            if a is None or not hasattr(a, "rotate"):
                return

            if self._right_press_pos is not None:
                ddx = float(pos.x() - self._right_press_pos.x())
                ddy = float(pos.y() - self._right_press_pos.y())
                if (ddx * ddx + ddy * ddy) < float(self.RIGHT_DRAG_THRESHOLD_PX * self.RIGHT_DRAG_THRESHOLD_PX):
                    return

            delta_deg = dx * 0.2
            a.rotate(delta_deg)
            self._sync_pose_ui_from_selected()
            self.refresh()
            return

        if self._drag_button != int(QtCore.Qt.LeftButton):
            return

        if self.move_mode == self.MOVE_CAMERA:
            self._pan_camera(dx, dy)
        else:
            self._drag_selected_actor(dx, dy)

        self._sync_pose_ui_from_selected()
        self.refresh()

    def _pan_camera(self, dx_px: float, dy_px: float) -> None:
        # world_per_pixel = camera.z (as in stage.py mapping)
        dz = max(float(self.camera.z), 1e-6)
        self.camera.move(self.camera.x - dx_px * dz, self.camera.y - dy_px * dz, self.camera.z)

    def _drag_selected_actor(self, dx_px: float, dy_px: float) -> None:
        a = self._get_selected_actor_obj()
        if a is None or not hasattr(a, "move"):
            return
        dz = max(float(self.camera.z), 1e-6)
        d_world_x = dx_px * dz
        d_world_y = dy_px * dz

        cx, cy = _actor_center_world(a)
        try:
            a.move(cx + d_world_x, cy + d_world_y)
        except Exception:
            pass

    def on_wheel(self, delta: int, mods: int) -> None:
        if self._team_count() == 0:
            return

        step = 0.1
        sign = -1.0 if delta > 0 else 1.0  # wheel up => zoom in

        # Single team OR camera mode => zoom camera.z
        if self._team_count() == 1 or self.move_mode == self.MOVE_CAMERA:
            z = max(float(self.camera.z), 1e-6)
            inv = 1.0 / z
            inv *= (1.0 + sign * step)
            inv = max(inv, 1e-12)
            new_z = max(1.0, 1.0 / inv)
            self.camera.move(self.camera.x, self.camera.y, new_z)
            self.refresh()
            return

        # Actor mode => scale selected actor (uniform)
        a = self._get_selected_actor_obj()
        if a is None or not hasattr(a, "scale"):
            return

        factor = max(1e-6, 1.0 + sign * step)
        a.scale(factor, factor)
        self._sync_pose_ui_from_selected()
        self.refresh()

    # ---------------- close ----------------

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        for st in (self.teamA, self.teamB):
            if st is None:
                continue
            try:
                st.team.save(st.vp_path)
            except Exception:
                pass
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication([])
    w = TeamViewer()
    w.resize(1500, 950)
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
