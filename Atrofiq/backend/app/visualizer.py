#!/usr/bin/env python3
import sys, os
from collections import OrderedDict
import numpy as np
import nibabel as nib
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Brain Visualiser with optional image/segmentation preload")
    parser.add_argument("-img", "--image", type=str, help="Path to the image NIfTI file")
    parser.add_argument("-seg", "--segmentation", type=str, help="Path to the segmentation NIfTI file")
    return parser.parse_args()


APP_STYLE = """
QWidget{ background: #0f1416; color: #e6eef6; font-family: 'Segoe UI', 'Helvetica Neue', Arial; }
QFrame#panel{ background: #111417; border-radius: 8px; }
QPushButton{ background: #2b6ea3; color: #ffffff; padding: 6px 10px; border-radius: 6px; }
QPushButton:hover{ background: #3b7fb8; }
QComboBox, QSlider, QListWidget, QSpinBox, QLineEdit{ background: #171b1d; color: #e6eef6; border-radius: 6px; }
QLabel#title{ color: #ffffff; font-weight: 600; font-size: 12pt }
QLabel#small{ color: #9fb0c8; font-size: 9pt }
"""

def robust_normalize(volume, low_pct=1.0, high_pct=99.0):
    if volume is None:
        return None
    v = np.asarray(volume, dtype=np.float32)
    pmin = np.percentile(v, low_pct)
    pmax = np.percentile(v, high_pct)
    if pmax > pmin:
        v = np.clip(v, pmin, pmax)
        v = (v - pmin) / (pmax - pmin)
    else:
        v = np.zeros_like(v)
    return v

def get_freesurfer_labels():
    """Return FreeSurfer labels with colors from the color lookup table"""
    return OrderedDict([
        (0, ('Unknown', (0, 0, 0, 0))),
        (1, ('Left-Cerebral-Exterior', (70, 130, 180, 200))),
        (2, ('Left-Cerebral-White-Matter', (245, 245, 245, 200))),
        (3, ('Left-Cerebral-Cortex', (205, 62, 78, 200))),
        (4, ('Left-Lateral-Ventricle', (120, 18, 134, 200))),
        (5, ('Left-Inf-Lat-Vent', (196, 58, 250, 200))),
        (6, ('Left-Cerebellum-Exterior', (0, 148, 0, 200))),
        (7, ('Left-Cerebellum-White-Matter', (220, 248, 164, 200))),
        (8, ('Left-Cerebellum-Cortex', (230, 148, 34, 200))),
        (9, ('Left-Thalamus-unused', (0, 118, 14, 200))),
        (10, ('Left-Thalamus', (0, 118, 14, 200))),
        (11, ('Left-Caudate', (122, 186, 220, 200))),
        (12, ('Left-Putamen', (236, 13, 176, 200))),
        (13, ('Left-Pallidum', (12, 48, 255, 200))),
        (14, ('3rd-Ventricle', (204, 182, 142, 200))),
        (15, ('4th-Ventricle', (42, 204, 164, 200))),
        (16, ('Brain-Stem', (119, 159, 176, 200))),
        (17, ('Left-Hippocampus', (220, 216, 20, 200))),
        (18, ('Left-Amygdala', (103, 255, 255, 200))),
        (19, ('Left-Insula', (80, 196, 98, 200))),
        (20, ('Left-Operculum', (60, 58, 210, 200))),
        (24, ('CSF', (60, 60, 60, 200))),
        (26, ('Left-Accumbens-area', (255, 165, 0, 200))),
        (28, ('Left-VentralDC', (165, 42, 42, 200))),
        (40, ('Right-Cerebral-Exterior', (70, 130, 180, 200))),
        (41, ('Right-Cerebral-White-Matter', (245, 245, 245, 200))),
        (42, ('Right-Cerebral-Cortex', (205, 62, 78, 200))),
        (43, ('Right-Lateral-Ventricle', (120, 18, 134, 200))),
        (44, ('Right-Inf-Lat-Vent', (196, 58, 250, 200))),
        (45, ('Right-Cerebellum-Exterior', (0, 148, 0, 200))),
        (46, ('Right-Cerebellum-White-Matter', (220, 248, 164, 200))),
        (47, ('Right-Cerebellum-Cortex', (230, 148, 34, 200))),
        (48, ('Right-Thalamus-unused', (0, 118, 14, 200))),
        (49, ('Right-Thalamus', (0, 118, 14, 200))),
        (50, ('Right-Caudate', (122, 186, 220, 200))),
        (51, ('Right-Putamen', (236, 13, 176, 200))),
        (52, ('Right-Pallidum', (13, 48, 255, 200))),
        (53, ('Right-Hippocampus', (220, 216, 20, 200))),
        (54, ('Right-Amygdala', (103, 255, 255, 200))),
        (55, ('Right-Insula', (80, 196, 98, 200))),
        (58, ('Right-Accumbens-area', (255, 165, 0, 200))),
        (60, ('Right-VentralDC', (165, 42, 42, 200))),
        (77, ('WM-hypointensities', (200, 70, 255, 200))),
        (85, ('Optic-Chiasm', (234, 169, 30, 200))),
        (251, ('CC_Posterior', (0, 0, 64, 200))),
        (252, ('CC_Mid_Posterior', (0, 0, 112, 200))),
        (253, ('CC_Central', (0, 0, 160, 200))),
        (254, ('CC_Mid_Anterior', (0, 0, 208, 200))),
        (255, ('CC_Anterior', (0, 0, 255, 200))),
        (1000, ('ctx-lh-unknown', (25, 5, 25, 200))),
        (1001, ('ctx-lh-bankssts', (25, 100, 40, 200))),
        (1002, ('ctx-lh-caudalanteriorcingulate', (125, 100, 160, 200))),
        (1003, ('ctx-lh-caudalmiddlefrontal', (100, 25, 0, 200))),
        (1005, ('ctx-lh-cuneus', (220, 20, 100, 200))),
        (1006, ('ctx-lh-entorhinal', (220, 20, 10, 200))),
        (1007, ('ctx-lh-fusiform', (180, 220, 140, 200))),
        (1008, ('ctx-lh-inferiorparietal', (220, 60, 220, 200))),
        (1009, ('ctx-lh-inferiortemporal', (180, 40, 120, 200))),
        (1010, ('ctx-lh-isthmuscingulate', (140, 20, 140, 200))),
        (1011, ('ctx-lh-lateraloccipital', (20, 30, 140, 200))),
        (1012, ('ctx-lh-lateralorbitofrontal', (35, 75, 50, 200))),
        (1013, ('ctx-lh-lingual', (225, 140, 140, 200))),
        (1014, ('ctx-lh-medialorbitofrontal', (200, 35, 75, 200))),
        (1015, ('ctx-lh-middletemporal', (160, 100, 50, 200))),
        (1016, ('ctx-lh-parahippocampal', (20, 220, 60, 200))),
        (1017, ('ctx-lh-paracentral', (60, 220, 60, 200))),
        (1018, ('ctx-lh-parsopercularis', (220, 180, 140, 200))),
        (1019, ('ctx-lh-parsorbitalis', (20, 100, 50, 200))),
        (1020, ('ctx-lh-parstriangularis', (220, 60, 20, 200))),
        (1021, ('ctx-lh-pericalcarine', (120, 100, 60, 200))),
        (1022, ('ctx-lh-postcentral', (220, 20, 20, 200))),
        (1023, ('ctx-lh-posteriorcingulate', (220, 180, 220, 200))),
        (1024, ('ctx-lh-precentral', (60, 20, 220, 200))),
        (1025, ('ctx-lh-precuneus', (160, 140, 180, 200))),
        (1026, ('ctx-lh-rostralanteriorcingulate', (80, 20, 140, 200))),
        (1027, ('ctx-lh-rostralmiddlefrontal', (75, 50, 125, 200))),
        (1028, ('ctx-lh-superiorfrontal', (20, 220, 160, 200))),
        (1029, ('ctx-lh-superiorparietal', (20, 180, 140, 200))),
        (1030, ('ctx-lh-superiortemporal', (140, 220, 220, 200))),
        (1031, ('ctx-lh-supramarginal', (80, 160, 20, 200))),
        (1032, ('ctx-lh-frontalpole', (100, 0, 100, 200))),
        (1033, ('ctx-lh-temporalpole', (70, 70, 70, 200))),
        (1034, ('ctx-lh-transversetemporal', (150, 150, 200, 200))),
        (1035, ('ctx-lh-insula', (255, 192, 32, 200))),
        (2000, ('ctx-rh-unknown', (25, 5, 25, 200))),
        (2001, ('ctx-rh-bankssts', (25, 100, 40, 200))),
        (2002, ('ctx-rh-caudalanteriorcingulate', (125, 100, 160, 200))),
        (2003, ('ctx-rh-caudalmiddlefrontal', (100, 25, 0, 200))),
        (2005, ('ctx-rh-cuneus', (220, 20, 100, 200))),
        (2006, ('ctx-rh-entorhinal', (220, 20, 10, 200))),
        (2007, ('ctx-rh-fusiform', (180, 220, 140, 200))),
        (2008, ('ctx-rh-inferiorparietal', (220, 60, 220, 200))),
        (2009, ('ctx-rh-inferiortemporal', (180, 40, 120, 200))),
        (2010, ('ctx-rh-isthmuscingulate', (140, 20, 140, 200))),
        (2011, ('ctx-rh-lateraloccipital', (20, 30, 140, 200))),
        (2012, ('ctx-rh-lateralorbitofrontal', (35, 75, 50, 200))),
        (2013, ('ctx-rh-lingual', (225, 140, 140, 200))),
        (2014, ('ctx-rh-medialorbitofrontal', (200, 35, 75, 200))),
        (2015, ('ctx-rh-middletemporal', (160, 100, 50, 200))),
        (2016, ('ctx-rh-parahippocampal', (20, 220, 60, 200))),
        (2017, ('ctx-rh-paracentral', (60, 220, 60, 200))),
        (2018, ('ctx-rh-parsopercularis', (220, 180, 140, 200))),
        (2019, ('ctx-rh-parsorbitalis', (20, 100, 50, 200))),
        (2020, ('ctx-rh-parstriangularis', (220, 60, 20, 200))),
        (2021, ('ctx-rh-pericalcarine', (120, 100, 60, 200))),
        (2022, ('ctx-rh-postcentral', (220, 20, 20, 200))),
        (2023, ('ctx-rh-posteriorcingulate', (220, 180, 220, 200))),
        (2024, ('ctx-rh-precentral', (60, 20, 220, 200))),
        (2025, ('ctx-rh-precuneus', (160, 140, 180, 200))),
        (2026, ('ctx-rh-rostralanteriorcingulate', (80, 20, 140, 200))),
        (2027, ('ctx-rh-rostralmiddlefrontal', (75, 50, 125, 200))),
        (2028, ('ctx-rh-superiorfrontal', (20, 220, 160, 200))),
        (2029, ('ctx-rh-superiorparietal', (20, 180, 140, 200))),
        (2030, ('ctx-rh-superiortemporal', (140, 220, 220, 200))),
        (2031, ('ctx-rh-supramarginal', (80, 160, 20, 200))),
        (2032, ('ctx-rh-frontalpole', (100, 0, 100, 200))),
        (2033, ('ctx-rh-temporalpole', (70, 70, 70, 200))),
        (2034, ('ctx-rh-transversetemporal', (150, 150, 200, 200))),
        (2035, ('ctx-rh-insula', (255, 192, 32, 200))),
    ])

class ViewMouseHandler(QtCore.QObject):
    def __init__(self, parent_app, view_index, viewbox, parent=None):
        super().__init__(parent)
        self.app = parent_app
        self.view_idx = view_index
        self.vb = viewbox
        self._dragging = False
        self._button = None
        self._last_scene_pos = None

    def eventFilter(self, obj, event):
        t = event.type()

        # mouse press
        if t == QtCore.QEvent.GraphicsSceneMousePress:
            self._button = event.button()
            self._dragging = True
            try:
                self._last_scene_pos = event.scenePos()
            except Exception:
                try:
                    self._last_scene_pos = event.screenPos()
                except Exception:
                    self._last_scene_pos = None
            # Consume only middle button press to prevent default handling
            return self._button == QtCore.Qt.MiddleButton

        # mouse release
        if t == QtCore.QEvent.GraphicsSceneMouseRelease:
            self._dragging = False
            self._button = None
            self._last_scene_pos = None
            # Consume only middle button release
            return event.button() == QtCore.Qt.MiddleButton

        # mouse move -> drags
        if t == QtCore.QEvent.GraphicsSceneMouseMove:
            if not self._dragging or self._last_scene_pos is None:
                return False
            try:
                current = event.scenePos()
            except Exception:
                try:
                    current = event.screenPos()
                except Exception:
                    return False
            last = self._last_scene_pos

            # LEFT BUTTON: zoom by vertical movement
            if self._button == QtCore.Qt.LeftButton:
                try:
                    scene_dy = current.y() - last.y()
                    factor = 1.0 - (scene_dy / 300.0)
                    factor = max(0.01, factor)
                    cursor_pt = self.vb.mapSceneToView(current)
                    self.vb.scaleBy((factor, factor), center=cursor_pt)
                except Exception:
                    pass
                self._last_scene_pos = current
                return False  # Allow default handling for left button

            # RIGHT BUTTON: pan
            elif self._button == QtCore.Qt.RightButton:
                try:
                    last_data = self.vb.mapSceneToView(last)
                    curr_data = self.vb.mapSceneToView(current)
                    dx = curr_data.x() - last_data.x()
                    dy = curr_data.y() - last_data.y()
                    self.vb.translateBy(x=-dx, y=-dy)
                except Exception:
                    pass
                self._last_scene_pos = current
                return False  # Allow default handling for right button

            # MIDDLE BUTTON: window/level (contrast and brightness)
            elif self._button == QtCore.Qt.MiddleButton:
                try:
                    scene_dx = current.x() - last.x()
                    scene_dy = current.y() - last.y()
                    delta_center = -scene_dy / 300.0 * self.app._wl_range
                    delta_width = scene_dx / 300.0 * self.app._wl_range
                    self.app._wl_center += delta_center
                    self.app._wl_width = max(1e-6, self.app._wl_width + delta_width)
                    self.app._wl_center = np.clip(self.app._wl_center, self.app._wl_vmin, self.app._wl_vmax)
                    center_norm = (self.app._wl_center - self.app._wl_vmin) / self.app._wl_range
                    width_norm = self.app._wl_width / self.app._wl_range
                    self.app.wl_center_slider.blockSignals(True)
                    self.app.wl_center_slider.setValue(int(center_norm * 1000))
                    self.app.wl_center_slider.blockSignals(False)
                    self.app.wl_width_slider.blockSignals(True)
                    self.app.wl_width_slider.setValue(int(width_norm * 1000))
                    self.app.wl_width_slider.blockSignals(False)
                    self.app.update_wl_labels()
                    self.app.update_views()
                except Exception:
                    pass
                self._last_scene_pos = current
                return True  # Consume middle button move to prevent panning

        # wheel -> slice change
        if t == QtCore.QEvent.GraphicsSceneWheel:
            delta = 0
            try:
                delta = event.angleDelta().y()
            except Exception:
                try:
                    delta = event.delta()
                except Exception:
                    delta = 0
            steps = int(delta / 120) if delta != 0 else 0
            if steps == 0:
                return False
            if self.app.data is None:
                return False

            axis_len = self.app.data.shape[[2,1,0][self.view_idx]]
            cur = int(self.app.current_slices[self.view_idx])
            new_idx = np.clip(cur - steps, 0, axis_len - 1)
            self.app.current_slices[self.view_idx] = int(new_idx)

            # sync slider
            try:
                s = self.app.sliders[self.view_idx]
                s.blockSignals(True)
                s.setValue(int(new_idx))
                s.blockSignals(False)
            except Exception:
                pass
            self.app.update_views()
            return True

        return False
    

class MedicalVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Brain Visualiser')
        self.resize(1400, 900)
        self.setStyleSheet(APP_STYLE)
        self.img = None
        self.data = None
        self.seg_img = None
        self.seg = None
        self.affine = None
        self.active_view = 0
        
        # Use FreeSurfer color lookup table
        self.freesurfer_labels = get_freesurfer_labels()
        self.label_names = OrderedDict()
        self.label_colors = {}
        
        for label_id, (name, color) in self.freesurfer_labels.items():
            self.label_names[label_id] = name
            self.label_colors[label_id] = color
            
        self.selected_labels = set([k for k in self.label_names.keys() if k != 0])
        self.current_slices = [0, 0, 0]
        self.img_views = []
        self.sliders = []
        self.seg_opacity_slider = None
        self.label_tree = None
        self.rotations = [0, 0, 0]
        self.wl_center_slider = None
        self.wl_width_slider = None
        self._toggle_all_state = True
        self._wl_vmin = 0.0
        self._wl_vmax = 1.0
        self._wl_range = 1.0
        self._wl_center = None
        self._wl_width = None
        self.mouse_handlers = []
        self._build_ui()
    
    def load_nifti_from_path(self, fp):
        """Load an image NIfTI file directly from a given path (no file dialog)."""
        try:
            img = nib.load(fp)
            data = img.get_fdata()
            self.affine = img.affine
            data_ras, affine_ras = self._reorient_to_ras(data, img.affine)
            self.affine = affine_ras
            self.raw_data = np.asarray(data_ras, dtype=np.float32)
            self.data = robust_normalize(self.raw_data, low_pct=0.5, high_pct=99.5).astype(np.float32)
            self.img = nib.Nifti1Image(self.data, self.affine)
            self.current_slices = [
                int(self.data.shape[2] // 2),
                int(self.data.shape[1] // 2),
                int(self.data.shape[0] // 2),
            ]

            # default rotation
            self.rotations = [1, 1, 1]

            # update sliders
            try:
                self.sliders[0].setRange(0, max(0, self.data.shape[2] - 1))
                self.sliders[0].setValue(self.current_slices[0])
                self.sliders[0].setVisible(True)

                self.sliders[1].setRange(0, max(0, self.data.shape[1] - 1))
                self.sliders[1].setValue(self.current_slices[1])
                self.sliders[1].setVisible(True)

                self.sliders[2].setRange(0, max(0, self.data.shape[0] - 1))
                self.sliders[2].setValue(self.current_slices[2])
                self.sliders[2].setVisible(True)
            except Exception:
                pass

            # WL (contrast/brightness) setup
            vmin = float(np.nanmin(self.raw_data))
            vmax = float(np.nanmax(self.raw_data))
            center = (vmin + vmax) / 2.0
            width = max(1.0, vmax - vmin)

            self._wl_vmin = vmin
            self._wl_vmax = vmax
            self._wl_range = vmax - vmin if vmax > vmin else 1.0
            self._wl_center = center
            self._wl_width = width

            self.wl_center_slider.setEnabled(True)
            self.wl_width_slider.setEnabled(True)
            try:
                self.wl_center_slider.blockSignals(True)
                self.wl_width_slider.blockSignals(True)
                self.wl_center_slider.setValue(int((center - vmin) / self._wl_range * 1000))
                self.wl_width_slider.setValue(int(width / self._wl_range * 1000))
            finally:
                try:
                    self.wl_center_slider.blockSignals(False)
                    self.wl_width_slider.blockSignals(False)
                except Exception:
                    pass

            self.update_wl_labels()
            self.setWindowTitle(f'Brain Visualiser — {os.path.basename(fp)}')
            self.update_views()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load NIfTI:\n{e}')


    def load_segmentation_from_path(self, fp):
        """Load a segmentation NIfTI file directly from a given path (no file dialog)."""
        if self.data is None:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please load an image first')
            return

        try:
            img = nib.load(fp)
            seg = img.get_fdata().astype(np.int32)
            seg_ras, affine_ras = self._reorient_to_ras(seg, img.affine, is_seg=True)
            if seg_ras.shape != self.data.shape:
                QtWidgets.QMessageBox.warning(
                    self, 'Warning',
                    'Segmentation shape does not match image shape — visualization could be incorrect'
                )

            self.seg_img = nib.Nifti1Image(seg_ras, affine_ras)
            self.seg = seg_ras

            # Add new labels if present
            uniques = np.unique(seg_ras)
            new_lbls = False
            for lid in map(int, uniques):
                if lid not in self.label_names:
                    self.label_names[int(lid)] = f'Label_{int(lid)}'
                    # random color for unknown labels
                    import random
                    self.label_colors[int(lid)] = (
                        random.randint(50, 255),
                        random.randint(50, 255),
                        random.randint(50, 255),
                        200
                    )
                    new_lbls = True

            if new_lbls:
                self.selected_labels = set([k for k in self.label_names.keys() if k != 0])
                self.populate_label_tree()

            self.update_views()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load segmentation:\n{e}')


    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)
        hbox.setContentsMargins(10, 10, 10, 10)
        hbox.setSpacing(10)
        
        viewer_frame = QtWidgets.QFrame()
        viewer_frame.setObjectName('panel')
        vlay = QtWidgets.QVBoxLayout(viewer_frame)
        vlay.setContentsMargins(8, 8, 8, 8)
        vlay.setSpacing(8)
        
        toolbar = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel('Brain Visualiser')
        title.setObjectName('title')
        toolbar.addWidget(title)
        toolbar.addStretch()
        btn_load = QtWidgets.QPushButton('Load NIfTI')
        btn_load.clicked.connect(self.load_nifti)
        btn_load_seg = QtWidgets.QPushButton('Load Segmentation')
        btn_load_seg.clicked.connect(self.load_segmentation)
        toolbar.addWidget(btn_load)
        toolbar.addWidget(btn_load_seg)
        vlay.addLayout(toolbar)

        page2d = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(page2d)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        
        for i, name in enumerate(['Axial (XY)', 'Coronal (XZ)', 'Sagittal (YZ)']):
            view_container = QtWidgets.QWidget()
            vv_l = QtWidgets.QVBoxLayout(view_container)
            vv_l.setContentsMargins(0, 0, 0, 0)
            lbl = QtWidgets.QLabel(name)
            lbl.setStyleSheet('font-weight:600; color:#ffffff')
            vv_l.addWidget(lbl)
            iv = pg.ImageView(view=pg.PlotItem())
            iv.ui.histogram.hide()
            iv.ui.roiBtn.hide()
            iv.ui.menuBtn.hide()
            iv.getView().setAspectLocked(True)
            iv.getView().invertY(False)
            iv.getView().setMouseEnabled(x=True, y=True)
            vv_l.addWidget(iv)
            btn_row = QtWidgets.QHBoxLayout()
            btn_rotate = QtWidgets.QPushButton('Rotate')
            btn_rotate.clicked.connect(lambda _, idx=i: self.rotate_view(idx))
            btn_row.addStretch()
            btn_row.addWidget(btn_rotate)
            vv_l.addLayout(btn_row)
            grid.addWidget(view_container, 0, i)
            self.img_views.append(iv)
        vlay.addWidget(page2d, 1)
        hbox.addWidget(viewer_frame, 1)

        ctrl = QtWidgets.QFrame()
        ctrl.setObjectName('panel')
        ctrl.setFixedWidth(360)
        rlay = QtWidgets.QVBoxLayout(ctrl)
        rlay.setContentsMargins(12, 12, 12, 12)
        rlay.setSpacing(8)
        
        header = QtWidgets.QLabel('Segmentation & Display Controls')
        header.setObjectName('title')
        rlay.addWidget(header)
        
        search_row = QtWidgets.QHBoxLayout()
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText('Search labels...')
        self.search_edit.textChanged.connect(self.filter_label_tree)
        search_row.addWidget(self.search_edit)
        self.btn_toggle_all = QtWidgets.QPushButton('Select All')
        self.btn_toggle_all.clicked.connect(self.toggle_select_all)
        search_row.addWidget(self.btn_toggle_all)
        rlay.addLayout(search_row)
        
        self.label_tree = QtWidgets.QTreeWidget()
        self.label_tree.setHeaderHidden(True)
        self.label_tree.itemChanged.connect(self.on_tree_item_changed)
        rlay.addWidget(self.label_tree, 1)
        
        rlay.addWidget(QtWidgets.QLabel('Segmentation Opacity'))
        self.seg_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.seg_opacity_slider.setRange(0, 100)
        self.seg_opacity_slider.setValue(60)
        self.seg_opacity_slider.setTracking(True)
        self.seg_opacity_slider.valueChanged.connect(self.update_views)
        rlay.addWidget(self.seg_opacity_slider)
        
        rlay.addWidget(QtWidgets.QLabel('Contrast & Brightness (use middle mouse or sliders)'))
        
        b_row = QtWidgets.QHBoxLayout()
        self.b_label = QtWidgets.QLabel('Brightness: 0')
        self.b_label.setFixedWidth(110)
        self.wl_center_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.wl_center_slider.setRange(0, 1000)
        self.wl_center_slider.setEnabled(False)
        self.wl_center_slider.setTracking(True)
        self.wl_center_slider.valueChanged.connect(self.on_wl_changed)
        self.wl_center_slider.sliderMoved.connect(self.update_views)
        b_row.addWidget(self.b_label)
        b_row.addWidget(self.wl_center_slider)
        rlay.addLayout(b_row)
        
        c_row = QtWidgets.QHBoxLayout()
        self.c_label = QtWidgets.QLabel('Contrast: 0')
        self.c_label.setFixedWidth(110)
        self.wl_width_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.wl_width_slider.setRange(1, 1000)
        self.wl_width_slider.setEnabled(False)
        self.wl_width_slider.setTracking(True)
        self.wl_width_slider.valueChanged.connect(self.on_wl_changed)
        self.wl_width_slider.sliderMoved.connect(self.update_views)
        c_row.addWidget(self.c_label)
        c_row.addWidget(self.wl_width_slider)
        rlay.addLayout(c_row)
        
        rlay.addWidget(QtWidgets.QLabel('Slice: use mouse wheel over image or sliders'))
        for i, name in enumerate(['Axial', 'Coronal', 'Sagittal']):
            lab = QtWidgets.QLabel(name)
            lab.setObjectName('small')
            rlay.addWidget(lab)
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(0, 100)
            s.setValue(50)
            s.setTracking(True)
            s.valueChanged.connect(self.on_slice_change)
            s.sliderMoved.connect(self.update_views)
            self.sliders.append(s)
            rlay.addWidget(s)
        
        for s in self.sliders:
            s.setVisible(False)
            
        btns = QtWidgets.QHBoxLayout()
        btn_fit = QtWidgets.QPushButton('Reset View')
        btn_fit.clicked.connect(self.reset_views)
        btns.addWidget(btn_fit)
        rlay.addLayout(btns)
        rlay.addStretch()
        hbox.addWidget(ctrl)
        
        self.populate_label_tree()
        QtCore.QTimer.singleShot(0, self._install_mouse_handlers)

    def _install_mouse_handlers(self):
        self.mouse_handlers = []
        for i, iv in enumerate(self.img_views):
            vb = iv.getView()
            scene = vb.scene()
            handler = ViewMouseHandler(self, i, vb)
            scene.installEventFilter(handler)
            self.mouse_handlers.append(handler)

    def populate_label_tree(self):
        self.label_tree.blockSignals(True)
        self.label_tree.clear()
        for lid, name in self.label_names.items():
            if lid == 0:
                continue
            item = QtWidgets.QTreeWidgetItem(self.label_tree)
            item.setText(0, name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.Checked if lid in self.selected_labels else QtCore.Qt.Unchecked)
            item.setData(0, QtCore.Qt.UserRole, ('label', int(lid)))
            if lid in self.label_colors:
                pix = QtGui.QPixmap(18, 12)
                color = self.label_colors[lid]
                pix.fill(QtGui.QColor(color[0], color[1], color[2], color[3]))
                item.setIcon(0, QtGui.QIcon(pix))
        self.label_tree.blockSignals(False)
        
        root = self.label_tree.invisibleRootItem()
        all_checked = True
        for i in range(root.childCount()):
            it = root.child(i)
            if it.checkState(0) != QtCore.Qt.Checked:
                all_checked = False
                break
        self._toggle_all_state = not all_checked
        self.btn_toggle_all.setText('Select All' if self._toggle_all_state else 'Deselect All')

    def filter_label_tree(self, text):
        if text is None:
            text = ''
        text = text.strip().lower()
        root = self.label_tree.invisibleRootItem()
        for i in range(root.childCount()):
            it = root.child(i)
            name = it.text(0).lower()
            it.setHidden(False if text == '' or text in name else True)

    def toggle_select_all(self):
        select = self._toggle_all_state
        self.label_tree.blockSignals(True)
        root = self.label_tree.invisibleRootItem()
        for i in range(root.childCount()):
            it = root.child(i)
            it.setCheckState(0, QtCore.Qt.Checked if select else QtCore.Qt.Unchecked)
            data = it.data(0, QtCore.Qt.UserRole)
            if data and data[0] == 'label':
                if select:
                    self.selected_labels.add(int(data[1]))
                else:
                    self.selected_labels.discard(int(data[1]))
        self.label_tree.blockSignals(False)
        self._toggle_all_state = not select
        self.btn_toggle_all.setText('Select All' if self._toggle_all_state else 'Deselect All')
        self.update_views()

    def on_tree_item_changed(self, item, col):
        data = item.data(0, QtCore.Qt.UserRole)
        if not data:
            return
        if data[0] == 'label':
            lid = int(data[1])
            if item.checkState(0) == QtCore.Qt.Checked:
                self.selected_labels.add(lid)
            else:
                self.selected_labels.discard(lid)
        
        root = self.label_tree.invisibleRootItem()
        all_checked = True
        for i in range(root.childCount()):
            it = root.child(i)
            if it.checkState(0) != QtCore.Qt.Checked:
                all_checked = False
                break
        self._toggle_all_state = not all_checked
        self.btn_toggle_all.setText('Select All' if self._toggle_all_state else 'Deselect All')
        self.update_views()

    def load_nifti(self):
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open NIfTI', os.getcwd(), 'NIfTI files (*.nii *.nii.gz)')
        if not fp:
            return
        try:
            img = nib.load(fp)
            data = img.get_fdata()
            self.affine = img.affine
            data_ras, affine_ras = self._reorient_to_ras(data, img.affine)
            self.affine = affine_ras
            self.raw_data = np.asarray(data_ras, dtype=np.float32)
            self.data = robust_normalize(self.raw_data, low_pct=0.5, high_pct=99.5).astype(np.float32)
            self.img = nib.Nifti1Image(self.data, self.affine)
            self.current_slices = [int(self.data.shape[2]//2), int(self.data.shape[1]//2), int(self.data.shape[0]//2)]
            
            # Set default rotation to 1 for each view (axial, coronal, sagittal)
            self.rotations = [1, 1, 1]
            
            try:
                self.sliders[0].setRange(0, max(0, self.data.shape[2]-1))
                self.sliders[0].setValue(self.current_slices[0])
                self.sliders[0].setVisible(True)
                self.sliders[1].setRange(0, max(0, self.data.shape[1]-1))
                self.sliders[1].setValue(self.current_slices[1])
                self.sliders[1].setVisible(True)
                self.sliders[2].setRange(0, max(0, self.data.shape[0]-1))
                self.sliders[2].setValue(self.current_slices[2])
                self.sliders[2].setVisible(True)
            except Exception:
                pass
            
            vmin = float(np.nanmin(self.raw_data))
            vmax = float(np.nanmax(self.raw_data))
            center = (vmin + vmax) / 2.0
            width = max(1.0, vmax - vmin)
            self._wl_vmin = vmin
            self._wl_vmax = vmax
            self._wl_range = vmax - vmin if vmax > vmin else 1.0
            self._wl_center = center
            self._wl_width = width
            
            self.wl_center_slider.setEnabled(True)
            self.wl_width_slider.setEnabled(True)
            try:
                self.wl_center_slider.blockSignals(True)
                self.wl_width_slider.blockSignals(True)
                self.wl_center_slider.setValue(int((center - vmin) / self._wl_range * 1000))
                self.wl_width_slider.setValue(int((width) / self._wl_range * 1000))
            finally:
                try:
                    self.wl_center_slider.blockSignals(False)
                    self.wl_width_slider.blockSignals(False)
                except Exception:
                    pass
            
            self.update_wl_labels()
            self.setWindowTitle(f'Brain Visualiser — {os.path.basename(fp)}')
            self.update_views()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load NIfTI:\n{e}')

    def load_segmentation(self):
        if self.data is None:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please load an image first')
            return
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Segmentation NIfTI', os.getcwd(), 'NIfTI files (*.nii *.nii.gz)')
        if not fp:
            return
        try:
            img = nib.load(fp)
            seg = img.get_fdata().astype(np.int32)
            seg_ras, affine_ras = self._reorient_to_ras(seg, img.affine, is_seg=True)
            if seg_ras.shape != self.data.shape:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'Segmentation shape does not match image shape — visualization could be incorrect')
            self.seg_img = nib.Nifti1Image(seg_ras, affine_ras)
            self.seg = seg_ras
            
            # Add any new labels found in the segmentation
            uniques = np.unique(seg_ras)
            new_lbls = False
            for lid in map(int, uniques):
                if lid not in self.label_names:
                    self.label_names[int(lid)] = f'Label_{int(lid)}'
                    # Assign a random color for unknown labels
                    import random
                    self.label_colors[int(lid)] = (
                        random.randint(50, 255),
                        random.randint(50, 255), 
                        random.randint(50, 255),
                        200
                    )
                    new_lbls = True
            
            if new_lbls:
                self.selected_labels = set([k for k in self.label_names.keys() if k != 0])
                self.populate_label_tree()
            
            self.update_views()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load segmentation:\n{e}')

    def _reorient_to_ras(self, data, affine, is_seg=False):
        try:
            ornt_current = nib.orientations.io_orientation(affine)
            target_axcodes = ('R', 'A', 'S')
            ornt_target = nib.orientations.axcodes2ornt(target_axcodes)
            transform = nib.orientations.ornt_transform(ornt_current, ornt_target)
            data_ras = nib.orientations.apply_orientation(data, transform)
            inv_aff = nib.orientations.inv_ornt_aff(transform, data.shape)
            affine_ras = affine.dot(inv_aff)
            if is_seg:
                data_ras = np.asarray(data_ras, dtype=np.int32)
            else:
                data_ras = np.asarray(data_ras, dtype=np.float32)
            return data_ras, affine_ras
        except Exception:
            if is_seg:
                return np.asarray(data, dtype=np.int32), affine
            return np.asarray(data, dtype=np.float32), affine

    def on_slice_change(self, _):
        if self.data is None:
            return
        try:
            self.current_slices = [int(self.sliders[0].value()), int(self.sliders[1].value()), int(self.sliders[2].value())]
        except Exception:
            pass
        self.update_views()

    def reset_views(self):
        """
        Reset everything:
        - reset pan/zoom (autoRange) for each view
        - reset rotations
        - reset current slices to image centers and update sliders
        - reset window/level sliders and internal WL values
        - reset segmentation opacity to default
        - reset active_view to 0 (axial)
        - refresh labels / views
        """
        if self.data is None:
            # nothing loaded — still reset UI controls to safe defaults
            self.rotations = [0, 0, 0]
            self.current_slices = [0, 0, 0]
            try:
                for s in self.sliders:
                    s.setValue(0)
            except Exception:
                pass
            return

        # 1) Reset pan/zoom for each view
        for iv in self.img_views:
            try:
                iv.getView().autoRange()        # resets view transform (pan/zoom)
            except Exception:
                try:
                    iv.view.autoRange()
                except Exception:
                    pass

        # 2) Reset rotations
        self.rotations = [0, 0, 0]

        # 3) Reset slices to the image center and update sliders
        self.current_slices = [
            int(self.data.shape[2] // 2),
            int(self.data.shape[1] // 2),
            int(self.data.shape[0] // 2),
        ]
        try:
            self.sliders[0].blockSignals(True)
            self.sliders[1].blockSignals(True)
            self.sliders[2].blockSignals(True)
            self.sliders[0].setRange(0, max(0, self.data.shape[2] - 1))
            self.sliders[1].setRange(0, max(0, self.data.shape[1] - 1))
            self.sliders[2].setRange(0, max(0, self.data.shape[0] - 1))
            self.sliders[0].setValue(self.current_slices[0])
            self.sliders[1].setValue(self.current_slices[1])
            self.sliders[2].setValue(self.current_slices[2])
        finally:
            try:
                self.sliders[0].blockSignals(False)
                self.sliders[1].blockSignals(False)
                self.sliders[2].blockSignals(False)
            except Exception:
                pass

        # 4) Reset window/level (contrast & brightness) to loaded defaults
        try:
            vmin = getattr(self, '_wl_vmin', 0.0)
            vmax = getattr(self, '_wl_vmax', vmin + 1.0)
            full = getattr(self, '_wl_range', max(1.0, vmax - vmin))
            center = (vmin + vmax) / 2.0
            width = max(1.0, vmax - vmin)

            self._wl_center = center
            self._wl_width = width

            # update sliders safely
            if hasattr(self, 'wl_center_slider') and self.wl_center_slider is not None:
                try:
                    self.wl_center_slider.blockSignals(True)
                    self.wl_center_slider.setValue(int((center - vmin) / full * 1000))
                finally:
                    try: self.wl_center_slider.blockSignals(False)
                    except Exception: pass

            if hasattr(self, 'wl_width_slider') and self.wl_width_slider is not None:
                try:
                    self.wl_width_slider.blockSignals(True)
                    self.wl_width_slider.setValue(int((width) / full * 1000))
                finally:
                    try: self.wl_width_slider.blockSignals(False)
                    except Exception: pass

            self.update_wl_labels()
        except Exception:
            pass

        # 5) Reset segmentation opacity to a reasonable default (60)
        try:
            if self.seg_opacity_slider is not None:
                self.seg_opacity_slider.blockSignals(True)
                self.seg_opacity_slider.setValue(60)
                self.seg_opacity_slider.blockSignals(False)
        except Exception:
            pass

        # 6) Reset active view index (optional)
        self.active_view = 0

        # 7) Finally refresh the displays
        self.update_views()


    def rotate_view(self, idx):
        self.rotations[idx] = (self.rotations[idx] + 1) % 4
        self.update_views()

    def update_wl_labels(self):
        if not hasattr(self, '_wl_range'):
            return
        vmin = getattr(self, '_wl_vmin', 0.0)
        full = getattr(self, '_wl_range', 1.0)
        try:
            center_val = vmin + (self.wl_center_slider.value() / 1000.0) * full
            width_val = max(1e-6, (self.wl_width_slider.value() / 1000.0) * max(1.0, full))
        except Exception:
            center_val = getattr(self, '_wl_center', vmin + full / 2.0)
            width_val = getattr(self, '_wl_width', max(1.0, full))
        self.b_label.setText(f'Brightness: {center_val:.1f}')
        self.c_label.setText(f'Contrast: {width_val:.1f}')
        self._wl_center = center_val
        self._wl_width = width_val

    def on_wl_changed(self, _):
        self.update_wl_labels()
        self.update_views()

    def update_views(self):
        if self.data is None:
            for iv in self.img_views:
                iv.clear()
            return
        
        base_a = self.data[:, :, int(self.current_slices[0])]
        base_c = self.data[:, int(self.current_slices[1]), :]
        base_s = self.data[int(self.current_slices[2]), :, :]
        planes_base = [base_a, base_c, base_s]
        
        try:
            cmap = pg.colormap.get('gray')
            lut = cmap.getLookupTable(0.0, 1.0, 256)
        except Exception:
            lut = None

        raw_min = float(np.nanmin(self.raw_data)) if getattr(self, 'raw_data', None) is not None else 0.0
        raw_max = float(np.nanmax(self.raw_data)) if getattr(self, 'raw_data', None) is not None else 1.0
        full_range = raw_max - raw_min if raw_max - raw_min != 0 else 1.0

        center_val = getattr(self, '_wl_center', None)
        width_val = getattr(self, '_wl_width', None)
        if center_val is None or width_val is None:
            try:
                center_val = self._wl_vmin + (self.wl_center_slider.value() / 1000.0) * getattr(self, '_wl_range', full_range)
                width_val = max(1e-6, (self.wl_width_slider.value() / 1000.0) * max(1.0, getattr(self, '_wl_range', full_range)))
            except Exception:
                center_val = raw_min + full_range / 2.0
                width_val = full_range

        level_min = center_val - width_val / 2.0
        level_max = center_val + width_val / 2.0
        if raw_max - raw_min != 0:
            lvl_min_n = (level_min - raw_min) / (raw_max - raw_min)
            lvl_max_n = (level_max - raw_min) / (raw_max - raw_min)
        else:
            lvl_min_n, lvl_max_n = 0.0, 1.0

        for i, iv in enumerate(self.img_views):
            rot = self.rotations[i]
            plane = planes_base[i].copy()
            if rot != 0:
                plane = np.rot90(plane, -rot)
            
            iv.setImage(plane.T, autoLevels=False, autoRange=False)
            try:
                iv.getImageItem().setLevels([lvl_min_n, lvl_max_n])
            except Exception:
                pass
            if lut is not None:
                try:
                    iv.getImageItem().setLookupTable(lut)
                except Exception:
                    pass

            # segmentation overlay
            if self.seg is not None:
                if i == 0:
                    seg_slice = self.seg[:, :, int(self.current_slices[0])].copy()
                elif i == 1:
                    seg_slice = self.seg[:, int(self.current_slices[1]), :].copy()
                else:
                    seg_slice = self.seg[int(self.current_slices[2]), :, :].copy()
                
                if rot != 0:
                    seg_slice = np.rot90(seg_slice, -rot)
                
                rgba = np.zeros(seg_slice.shape + (4,), dtype=np.uint8)
                for lid in np.unique(seg_slice):
                    lid = int(lid)
                    if lid in self.selected_labels and lid in self.label_colors:
                        col = self.label_colors[lid]
                        rgba[seg_slice == lid] = col
                
                overlay = getattr(iv, 'seg_overlay', None)
                if overlay is None:
                    overlay = pg.ImageItem()
                    overlay.setZValue(10)
                    iv.addItem(overlay)
                    iv.seg_overlay = overlay
                overlay.setImage(rgba.transpose(1, 0, 2))
                overlay.setOpacity(self.seg_opacity_slider.value() / 100.0)
            else:
                overlay = getattr(iv, 'seg_overlay', None)
                if overlay is not None:
                    try:
                        iv.removeItem(overlay)
                        delattr(iv, 'seg_overlay')
                    except Exception:
                        pass

    def keyPressEvent(self, ev):
        if self.data is None:
            return

        key = ev.key()
        step_slice = 1
        step_w = getattr(self, '_wl_range', 1.0) * 0.02  # 2% per contrast step

        # Shift -> cycle active view
        if key == QtCore.Qt.Key_Shift:
            self.active_view = (getattr(self, "active_view", 0) + 1) % 3
            QtWidgets.QMessageBox.information(
                self, "Active View",
                f"Now controlling: {['Axial','Coronal','Sagittal'][self.active_view]}"
            )
            return

        # Up / Down -> slice change for active view
        if key == QtCore.Qt.Key_Up:
            s = self.sliders[self.active_view]
            idx = min(s.maximum(), s.value() + step_slice)
            s.setValue(int(idx))
            self.current_slices[self.active_view] = idx
            self.update_views()
            return

        if key == QtCore.Qt.Key_Down:
            s = self.sliders[self.active_view]
            idx = max(s.minimum(), s.value() - step_slice)
            s.setValue(int(idx))
            self.current_slices[self.active_view] = idx
            self.update_views()
            return

        # Left / Right -> contrast (width) change (global)
        if key == QtCore.Qt.Key_Right:
            current_width = getattr(self, '_wl_width', (self._wl_vmax - self._wl_vmin))
            new_width = current_width + step_w
            v = int((new_width) / getattr(self, '_wl_range', 1.0) * 1000.0)
            v = np.clip(v, self.wl_width_slider.minimum(), self.wl_width_slider.maximum())
            self.wl_width_slider.setValue(int(v))
            self.update_wl_labels()
            self.update_views()
            return

        if key == QtCore.Qt.Key_Left:
            current_width = getattr(self, '_wl_width', (self._wl_vmax - self._wl_vmin))
            new_width = max(1e-3, current_width - step_w)
            v = int((new_width) / getattr(self, '_wl_range', 1.0) * 1000.0)
            v = np.clip(v, self.wl_width_slider.minimum(), self.wl_width_slider.maximum())
            self.wl_width_slider.setValue(int(v))
            self.update_wl_labels()
            self.update_views()
            return

        super().keyPressEvent(ev)


def main():
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    win = MedicalVisualizer()

    if args.image:
        try:
            win.load_nifti_from_path(args.image)
        except Exception as e:
            print(f"Failed to load image {args.image}: {e}")

    if args.segmentation:
        try:
            win.load_segmentation_from_path(args.segmentation)
        except Exception as e:
            print(f"Failed to load segmentation {args.segmentation}: {e}")

    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    
    # python test.py
    # python test.py -img /path/to/image.nii.gz
    # python test.py -img /path/to/image.nii.gz -seg /path/to/segmentation.nii.gz
    # python test.py --help