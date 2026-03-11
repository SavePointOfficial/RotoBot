#!/usr/bin/env python
"""
Rotobot GUI — Drag-and-Drop Automatic Rotoscoping

A PyQt6 interface for the Rotobot engine. Drag images onto the window,
set your prompt and parameters, click Process, get RGBA PNGs.
"""

import os
import sys
import time
import threading
import numpy as np

from rotobot_logging import get_logger
log = get_logger("GUI")

# Resolve imports
ROTOBOT_DIR = os.path.dirname(os.path.abspath(__file__))
SANDBOX_ROOT = os.path.dirname(ROTOBOT_DIR)
if ROTOBOT_DIR not in sys.path:
    sys.path.insert(0, ROTOBOT_DIR)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSlider, QProgressBar, QFileDialog,
    QTextEdit, QSplitter, QFrame, QComboBox, QCheckBox, QGroupBox,
    QScrollArea, QSizePolicy, QColorDialog, QSpinBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData, QSettings
from PyQt6.QtGui import (
    QPixmap, QImage, QFont, QColor, QPalette, QDragEnterEvent,
    QDropEvent, QPainter,
)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS


# ============================================================================
# Worker thread
# ============================================================================

class ProcessWorker(QThread):
    """Runs Rotobot engine in a background thread."""
    progress = pyqtSignal(int, int, str)      # current, total, filename
    file_done = pyqtSignal(str, object)        # filename, alpha_or_None
    finished = pyqtSignal(int, int, float)     # success, failed, elapsed

    def __init__(self, files, prompt, box_thresh, text_thresh, refine, invert,
                 output_dir, color_key=False, key_color=(0, 0, 0),
                 key_tolerance=30.0, feather=0, crop=False, upscale_to=0):
        super().__init__()
        self.files = files
        self.prompt = prompt
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh
        self.refine = refine
        self.invert = invert
        self.output_dir = output_dir
        self.color_key = color_key
        self.key_color = key_color
        self.key_tolerance = key_tolerance
        self.feather = feather
        self.crop = crop
        self.upscale_to = upscale_to
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        from rotobot_engine import RotobotEngine
        engine = RotobotEngine.get_instance()

        # Only load AI models if not in color-key mode
        if not self.color_key:
            engine.ensure_models()

        success = 0
        failed = 0
        t0 = time.perf_counter()

        for i, fpath in enumerate(self.files):
            if self._cancel:
                break

            basename = os.path.basename(fpath)
            self.progress.emit(i + 1, len(self.files), basename)

            if self.color_key:
                alpha = engine.extract_alpha_from_color(
                    image_path=fpath,
                    key_color=self.key_color,
                    tolerance=self.key_tolerance,
                    refine=self.refine,
                    invert=self.invert,
                )
            else:
                alpha = engine.extract_alpha(
                    image_path=fpath,
                    prompt=self.prompt,
                    box_threshold=self.box_thresh,
                    text_threshold=self.text_thresh,
                    refine=self.refine,
                    invert=self.invert,
                )

            if alpha is not None:
                import random
                rid = "%08d" % random.randint(0, 99999999)
                out_name = "%s_%s_alpha.png" % (os.path.splitext(basename)[0], rid)
                out_path = os.path.join(self.output_dir, out_name)
                engine.save_rgba(fpath, alpha, out_path,
                    feather=self.feather, crop=self.crop,
                    upscale_to=self.upscale_to)
                success += 1
            else:
                failed += 1

            self.file_done.emit(basename, alpha)

        elapsed = time.perf_counter() - t0
        self.finished.emit(success, failed, elapsed)


class VideoWorker(QThread):
    """Processes a video file frame-by-frame, producing alpha-channel output."""
    progress = pyqtSignal(int, int, str)      # current_frame, total_frames, status
    finished = pyqtSignal(int, int, float)     # frames_ok, frames_failed, elapsed

    def __init__(self, video_path, prompt, box_thresh, text_thresh, refine, invert,
                 output_path, output_format='webm', color_key=False,
                 key_color=(0, 0, 0), key_tolerance=30.0, feather=0):
        super().__init__()
        self.video_path = video_path
        self.prompt = prompt
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh
        self.refine = refine
        self.invert = invert
        self.output_path = output_path
        self.output_format = output_format
        self.color_key = color_key
        self.key_color = key_color
        self.key_tolerance = key_tolerance
        self.feather = feather
        self._cancel = False
        self._cancel_event = None

    def cancel(self):
        self._cancel = True
        if self._cancel_event:
            self._cancel_event.set()

    def run(self):
        import threading as _threading
        from rotobot_engine import RotobotEngine

        engine = RotobotEngine.get_instance()
        t0 = time.perf_counter()
        self._cancel_event = _threading.Event()

        info = engine.get_video_info(self.video_path)
        total = info['frames']
        fps = info['fps']
        self.progress.emit(0, total, "Loading models...")

        # Collect all processed frames into a list for encoding
        frames_data = []  # list of (frame_rgb, alpha)
        frames_ok = 0

        for idx, total_f, frame_rgb, alpha in engine.process_video_frames(
            video_path=self.video_path,
            prompt=self.prompt,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh,
            refine=self.refine,
            invert=self.invert,
            color_key=self.color_key,
            key_color=self.key_color,
            key_tolerance=self.key_tolerance,
            feather=self.feather,
            cancel_event=self._cancel_event,
        ):
            if self._cancel:
                break
            frames_data.append((frame_rgb, alpha))
            frames_ok += 1
            self.progress.emit(idx + 1, total_f,
                "Frame %d/%d" % (idx + 1, total_f))

        if self._cancel or not frames_data:
            elapsed = time.perf_counter() - t0
            self.finished.emit(0, 0, elapsed)
            return

        # Encode output
        self.progress.emit(frames_ok, total, "Encoding %s..." % self.output_format.upper())
        ok = engine.save_video_rgba(
            video_path=self.video_path,
            frame_generator=iter(frames_data),
            output_path=self.output_path,
            fps=fps,
            output_format=self.output_format,
        )

        elapsed = time.perf_counter() - t0
        self.finished.emit(frames_ok if ok else 0, 0 if ok else frames_ok, elapsed)


class InventoryWorker(QThread):
    """Runs SAM2 automatic segmentation to extract all visual elements."""
    progress = pyqtSignal(int, int, str)       # current, total, status_text
    file_done = pyqtSignal(str, object)         # filename, alpha_or_None
    finished = pyqtSignal(int, int, float)      # success, failed, elapsed

    def __init__(self, files, output_dir, box_thresh, text_thresh, refine,
                 feather=0, crop=False, upscale_to=0, detail=3,
                 group_by_source=False):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.refine = refine
        self.feather = feather
        self.crop = crop
        self.upscale_to = upscale_to
        self.detail = detail
        self.group_by_source = group_by_source
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        from rotobot_engine import RotobotEngine

        t0 = time.perf_counter()
        total_success = 0
        total_failed = 0

        # Load SAM2 model
        self.progress.emit(0, len(self.files), "Loading SAM2 model...")
        engine = RotobotEngine.get_instance()
        engine._ensure_sam2()

        for img_i, fpath in enumerate(self.files):
            if self._cancel:
                break

            basename = os.path.basename(fpath)
            stem = os.path.splitext(basename)[0]

            self.progress.emit(img_i, len(self.files),
                               "Segmenting: %s" % basename)

            # Run automatic segmentation
            segments = engine.segment_all(
                image_path=fpath,
                min_area=0.005,
                max_area=0.65,
                refine=self.refine,
                detail=self.detail,
            )

            if not segments:
                self.progress.emit(img_i + 1, len(self.files),
                                   "No elements found in %s" % basename)
                total_failed += 1
                self.file_done.emit(basename, None)
                continue

            self.progress.emit(img_i, len(self.files),
                               "Found %d elements in %s" % (len(segments), basename))

            # Determine output path per segment
            for seg_i, (alpha, area_frac, bbox) in enumerate(segments):
                if self._cancel:
                    break

                self.progress.emit(img_i, len(self.files),
                    "[%d/%d] %s: element_%03d (%.1f%%)" % (
                        seg_i + 1, len(segments), basename, seg_i + 1, area_frac * 100))

                import random
                rid = "%08d" % random.randint(0, 99999999)
                if self.group_by_source:
                    seg_dir = os.path.join(self.output_dir, stem)
                    os.makedirs(seg_dir, exist_ok=True)
                    out_path = os.path.join(seg_dir, "element_%s_alpha.png" % rid)
                else:
                    out_path = os.path.join(
                        self.output_dir, "%s_%s_alpha.png" % (stem, rid))
                engine.save_rgba(fpath, alpha, out_path,
                    feather=self.feather, crop=self.crop,
                    upscale_to=self.upscale_to)
                total_success += 1
                self.file_done.emit(basename, alpha)

        elapsed = time.perf_counter() - t0
        self.finished.emit(total_success, total_failed, elapsed)


# ============================================================================
# Drop Zone widget
# ============================================================================

class DropZone(QLabel):
    """Drag-and-drop area for images."""
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(180)
        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #666;
                border-radius: 12px;
                background: #1a1a2e;
                color: #888;
                font-size: 15px;
                padding: 20px;
            }
            DropZone:hover {
                border-color: #00d4ff;
                color: #aaa;
            }
        """)
        self.setText("🖼️  Drag & Drop Images, Videos, or Folders Here\n\nor click Browse below")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self.styleSheet().replace(
                "border-color: #666", "border-color: #00d4ff"))

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.styleSheet().replace(
            "border-color: #00d4ff", "border-color: #666"))

    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in MEDIA_EXTS:
                    files.append(path)
            elif os.path.isdir(path):
                for f in sorted(os.listdir(path)):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in MEDIA_EXTS:
                        files.append(os.path.join(path, f))

        if files:
            self.files_dropped.emit(files)
            # Determine label
            n_vid = sum(1 for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTS)
            n_img = len(files) - n_vid
            parts = []
            if n_img:
                parts.append("%d image%s" % (n_img, "s" if n_img != 1 else ""))
            if n_vid:
                parts.append("%d video%s" % (n_vid, "s" if n_vid != 1 else ""))
            self.setText("📁  %s loaded" % ", ".join(parts))

        self.setStyleSheet(self.styleSheet().replace(
            "border-color: #00d4ff", "border-color: #666"))


# ============================================================================
# Preview widget
# ============================================================================

class PreviewPanel(QLabel):
    """Shows original → alpha side-by-side."""
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(200)
        self.setStyleSheet("""
            background: #0d0d1a;
            border: 1px solid #333;
            border-radius: 8px;
        """)
        self.setText("Preview will appear here")
        self.setScaledContents(False)

    def show_result(self, image_path: str, alpha: np.ndarray):
        """Display original and alpha side by side."""
        from PIL import Image

        try:
            img = Image.open(image_path).convert("RGB")
            img_array = np.array(img)
            h, w = img_array.shape[:2]

            # Resize alpha to match if needed
            ah, aw = alpha.shape[:2]
            if (ah, aw) != (h, w):
                import cv2
                alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

            # Scale down for preview
            max_preview_w = (self.width() - 20) // 2
            max_preview_h = self.height() - 20
            scale = min(max_preview_w / w, max_preview_h / h, 1.0)
            pw = max(int(w * scale), 1)
            ph = max(int(h * scale), 1)

            # Original thumbnail
            orig_thumb = Image.fromarray(img_array).resize((pw, ph), Image.LANCZOS)

            # Alpha visualization: checkerboard + masked image
            checker = self._make_checker(pw, ph, 8)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img_array
            rgba[:, :, 3] = alpha
            rgba_img = Image.fromarray(rgba, 'RGBA').resize((pw, ph), Image.LANCZOS)

            # Composite over checkerboard
            alpha_viz = Image.fromarray(checker).convert('RGBA')
            alpha_viz.paste(rgba_img, (0, 0), rgba_img)
            alpha_viz = alpha_viz.convert('RGB')

            # Side by side
            canvas_w = pw * 2 + 10
            canvas = Image.new('RGB', (canvas_w, ph), (13, 13, 26))
            canvas.paste(orig_thumb, (0, 0))
            canvas.paste(alpha_viz, (pw + 10, 0))

            # Convert to QPixmap
            canvas_array = np.array(canvas)
            qimg = QImage(canvas_array.data, canvas_w, ph,
                          canvas_w * 3, QImage.Format.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qimg))
        except Exception as e:
            self.setText("Preview error: %s" % e)

    @staticmethod
    def _make_checker(w: int, h: int, size: int = 8) -> np.ndarray:
        """Create a checkerboard pattern for alpha visualization."""
        checker = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                if (x // size + y // size) % 2 == 0:
                    checker[y, x] = [180, 180, 180]
                else:
                    checker[y, x] = [220, 220, 220]
        return checker


# ============================================================================
# Main Window
# ============================================================================

class RotobotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 Rotobot — Automatic Rotoscoping")
        self.setMinimumSize(900, 700)
        self.resize(1000, 750)

        self._files = []
        self._worker = None
        self._output_dir = ""
        self._is_video_mode = False
        self._settings = QSettings("Rotobot", "RotobotGUI")

        self._setup_palette()
        self._build_ui()
        self._restore_geometry()
        log.info("Rotobot GUI started")

    # ------------------------------------------------------------------
    # Persistent geometry
    # ------------------------------------------------------------------

    def _restore_geometry(self):
        """Restore saved window geometry and all settings."""
        geom = self._settings.value("window/geometry")
        if geom is not None:
            self.restoreGeometry(geom)

        # Restore all UI settings
        s = self._settings
        v = s.value  # shorthand
        if v("ui/prompt") is not None:
            self._prompt_edit.setText(str(v("ui/prompt")))
        if v("ui/box_thresh") is not None:
            self._thresh_slider.setValue(int(v("ui/box_thresh")))
        if v("ui/refine") is not None:
            self._refine_combo.setCurrentIndex(int(v("ui/refine")))
        if v("ui/invert") is not None:
            self._invert_check.setChecked(str(v("ui/invert")).lower() == "true")
        if v("ui/feather") is not None:
            self._feather_slider.setValue(int(v("ui/feather")))
        if v("ui/colorkey") is not None:
            self._colorkey_check.setChecked(str(v("ui/colorkey")).lower() == "true")
        if v("ui/tolerance") is not None:
            self._tolerance_slider.setValue(int(v("ui/tolerance")))
        if v("ui/key_color") is not None:
            c = QColor(str(v("ui/key_color")))
            if c.isValid():
                self._set_key_color(c)
        if v("ui/inventory") is not None:
            self._inventory_check.setChecked(str(v("ui/inventory")).lower() == "true")
        if v("ui/crop") is not None:
            self._crop_check.setChecked(str(v("ui/crop")).lower() != "false")
        if v("ui/upscale") is not None:
            self._upscale_check.setChecked(str(v("ui/upscale")).lower() == "true")
        if v("ui/upscale_target") is not None:
            self._upscale_target.setText(str(v("ui/upscale_target")))
        if v("ui/output_dir") is not None:
            self._output_edit.setText(str(v("ui/output_dir")))
        if v("ui/detail") is not None:
            self._detail_slider.setValue(int(v("ui/detail")))
        if v("ui/group_by_source") is not None:
            self._group_check.setChecked(str(v("ui/group_by_source")).lower() != "false")
        if v("ui/video_format") is not None:
            self._video_format_combo.setCurrentIndex(int(v("ui/video_format")))
        if v("ui/vram_opt") is not None:
            self._vram_opt_check.setChecked(str(v("ui/vram_opt")).lower() == "true")
        if v("ui/vram_limit") is not None:
            self._vram_spin.setValue(int(v("ui/vram_limit")))

    def closeEvent(self, event):
        """Save window geometry and all settings on close."""
        self._settings.setValue("window/geometry", self.saveGeometry())

        # Save all UI settings
        s = self._settings
        s.setValue("ui/prompt", self._prompt_edit.text())
        s.setValue("ui/box_thresh", self._thresh_slider.value())
        s.setValue("ui/refine", self._refine_combo.currentIndex())
        s.setValue("ui/invert", self._invert_check.isChecked())
        s.setValue("ui/feather", self._feather_slider.value())
        s.setValue("ui/colorkey", self._colorkey_check.isChecked())
        s.setValue("ui/tolerance", self._tolerance_slider.value())
        s.setValue("ui/key_color", self._key_color.name())
        s.setValue("ui/inventory", self._inventory_check.isChecked())
        s.setValue("ui/crop", self._crop_check.isChecked())
        s.setValue("ui/upscale", self._upscale_check.isChecked())
        s.setValue("ui/upscale_target", self._upscale_target.text())
        s.setValue("ui/output_dir", self._output_edit.text())
        s.setValue("ui/detail", self._detail_slider.value())
        s.setValue("ui/group_by_source", self._group_check.isChecked())
        s.setValue("ui/video_format", self._video_format_combo.currentIndex())
        s.setValue("ui/vram_opt", self._vram_opt_check.isChecked())
        s.setValue("ui/vram_limit", self._vram_spin.value())

        log.info("Rotobot GUI closed")
        super().closeEvent(event)

    def _setup_palette(self):
        """Dark theme palette."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(22, 22, 38))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 235))
        palette.setColor(QPalette.ColorRole.Base, QColor(16, 16, 30))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 50))
        palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 235))
        palette.setColor(QPalette.ColorRole.Button, QColor(35, 35, 60))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 235))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 212, 255))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # --- Title ---
        title = QLabel("🤖 ROTOBOT")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #00d4ff; margin-bottom: 5px;")
        layout.addWidget(title)

        subtitle = QLabel("Automatic Rotoscoping — Powered by Grounded SAM2")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(subtitle)

        # --- Drop Zone ---
        self._drop_zone = DropZone()
        self._drop_zone.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self._drop_zone)

        # --- Browse button ---
        browse_layout = QHBoxLayout()
        browse_btn = QPushButton("📂 Browse Files...")
        browse_btn.setStyleSheet(self._button_style("#2a2a4a", "#3a3a6a"))
        browse_btn.clicked.connect(self._browse_files)
        browse_layout.addWidget(browse_btn)

        browse_folder_btn = QPushButton("📁 Browse Folder...")
        browse_folder_btn.setStyleSheet(self._button_style("#2a2a4a", "#3a3a6a"))
        browse_folder_btn.clicked.connect(self._browse_folder)
        browse_layout.addWidget(browse_folder_btn)
        layout.addLayout(browse_layout)

        # --- Settings ---
        settings_group = QGroupBox("Settings")
        settings_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #333;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #bbb;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        settings_layout = QVBoxLayout(settings_group)

        # Prompt
        prompt_row = QHBoxLayout()
        prompt_row.addWidget(QLabel("Prompt:"))
        self._prompt_edit = QLineEdit("foreground subject . main object")
        self._prompt_edit.setStyleSheet(
            "background: #1a1a2e; border: 1px solid #444; border-radius: 4px; "
            "padding: 5px; color: #ddd; font-size: 13px;")
        self._prompt_edit.setPlaceholderText(
            'e.g. "person . dog" or "car" or "foreground subject"')
        prompt_row.addWidget(self._prompt_edit, stretch=1)
        settings_layout.addLayout(prompt_row)

        # Threshold + Refine row
        controls_row = QHBoxLayout()

        controls_row.addWidget(QLabel("Box Threshold:"))
        self._thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._thresh_slider.setRange(5, 80)
        self._thresh_slider.setValue(25)
        self._thresh_slider.setStyleSheet(self._slider_style())
        self._thresh_label = QLabel("0.25")
        self._thresh_slider.valueChanged.connect(
            lambda v: self._thresh_label.setText("%.2f" % (v / 100.0)))
        controls_row.addWidget(self._thresh_slider)
        controls_row.addWidget(self._thresh_label)

        controls_row.addSpacing(20)

        controls_row.addWidget(QLabel("Refine:"))
        self._refine_combo = QComboBox()
        self._refine_combo.addItems(["Off (0)", "Light (1)", "Medium (2)", "Heavy (3)"])
        self._refine_combo.setCurrentIndex(1)
        self._refine_combo.setStyleSheet(
            "background: #1a1a2e; border: 1px solid #444; border-radius: 4px; "
            "padding: 3px; color: #ddd;")
        controls_row.addWidget(self._refine_combo)

        controls_row.addSpacing(20)

        self._invert_check = QCheckBox("Invert Alpha")
        controls_row.addWidget(self._invert_check)

        controls_row.addSpacing(20)

        controls_row.addWidget(QLabel("Feather:"))
        self._feather_slider = QSlider(Qt.Orientation.Horizontal)
        self._feather_slider.setRange(0, 20)
        self._feather_slider.setValue(0)
        self._feather_slider.setFixedWidth(100)
        self._feather_slider.setStyleSheet(self._slider_style())
        self._feather_label = QLabel("0 px")
        self._feather_slider.valueChanged.connect(
            lambda v: self._feather_label.setText("%d px" % v))
        controls_row.addWidget(self._feather_slider)
        controls_row.addWidget(self._feather_label)

        settings_layout.addLayout(controls_row)

        # Color key row
        colorkey_row = QHBoxLayout()

        self._colorkey_check = QCheckBox("Roto from Color")
        self._colorkey_check.setToolTip(
            "Key out a specific background color instead of using AI detection.\n"
            "Great for black backgrounds, green screens, etc.")
        self._colorkey_check.toggled.connect(self._on_colorkey_toggled)
        colorkey_row.addWidget(self._colorkey_check)

        self._color_btn = QPushButton("")
        self._color_btn.setFixedSize(32, 24)
        self._key_color = QColor(0, 0, 0)
        self._color_btn.setStyleSheet(
            "background: #000000; border: 2px solid #666; border-radius: 4px;")
        self._color_btn.setToolTip("Click to pick the background color to key out")
        self._color_btn.clicked.connect(self._pick_key_color)
        self._color_btn.setEnabled(False)
        colorkey_row.addWidget(self._color_btn)

        # Presets
        colorkey_row.addSpacing(8)
        black_btn = QPushButton("Black")
        black_btn.setFixedWidth(50)
        black_btn.setStyleSheet(self._button_style("#1a1a2e", "#2a2a4e"))
        black_btn.clicked.connect(lambda: self._set_key_color(QColor(0, 0, 0)))
        colorkey_row.addWidget(black_btn)

        green_btn = QPushButton("Green")
        green_btn.setFixedWidth(50)
        green_btn.setStyleSheet(self._button_style("#1a2e1a", "#2a4e2a"))
        green_btn.clicked.connect(lambda: self._set_key_color(QColor(0, 177, 64)))
        colorkey_row.addWidget(green_btn)

        blue_btn = QPushButton("Blue")
        blue_btn.setFixedWidth(50)
        blue_btn.setStyleSheet(self._button_style("#1a1a3e", "#2a2a5e"))
        blue_btn.clicked.connect(lambda: self._set_key_color(QColor(0, 71, 187)))
        colorkey_row.addWidget(blue_btn)

        white_btn = QPushButton("White")
        white_btn.setFixedWidth(50)
        white_btn.setStyleSheet(self._button_style("#3a3a3a", "#5a5a5a"))
        white_btn.clicked.connect(lambda: self._set_key_color(QColor(255, 255, 255)))
        colorkey_row.addWidget(white_btn)

        settings_layout.addLayout(colorkey_row)

        # Tolerance row (separate so it has room to breathe)
        tolerance_row = QHBoxLayout()
        tolerance_row.addSpacing(25)  # indent to align with color key controls
        tolerance_row.addWidget(QLabel("Tolerance:"))
        self._tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self._tolerance_slider.setRange(1, 100)
        self._tolerance_slider.setValue(30)
        self._tolerance_slider.setFixedWidth(200)
        self._tolerance_slider.setMinimumHeight(24)
        self._tolerance_slider.setStyleSheet(self._slider_style())
        self._tolerance_label = QLabel("30")
        self._tolerance_slider.valueChanged.connect(
            lambda v: self._tolerance_label.setText(str(v)))
        tolerance_row.addWidget(self._tolerance_slider)
        tolerance_row.addWidget(self._tolerance_label)
        tolerance_row.addStretch()
        settings_layout.addLayout(tolerance_row)

        # Auto-inventory row
        inventory_row = QHBoxLayout()

        self._inventory_check = QCheckBox("Auto-Inventory (SAM2)")
        self._inventory_check.setToolTip(
            "Automatically segment the image into all visual elements\n"
            "using SAM2, and extract each one individually.\n"
            "Great for asset extraction from complex artwork.\n\n"
            "NOTE: This requires an external LLaMA Vision API connected on port 5001.\n"
            "This is an upcoming feature for end-users in a later version.")
        self._inventory_check.toggled.connect(self._on_inventory_toggled)
        inventory_row.addWidget(self._inventory_check)

        inventory_row.addSpacing(15)
        inventory_row.addWidget(QLabel("Detail:"))
        self._detail_slider = QSlider(Qt.Orientation.Horizontal)
        self._detail_slider.setRange(1, 5)
        self._detail_slider.setValue(3)
        self._detail_slider.setFixedWidth(80)
        self._detail_slider.setStyleSheet(self._slider_style())
        self._detail_label = QLabel("3")
        self._detail_slider.valueChanged.connect(
            lambda v: self._detail_label.setText(str(v)))
        inventory_row.addWidget(self._detail_slider)
        inventory_row.addWidget(self._detail_label)

        self._vision_status = QLabel("")
        self._vision_status.setStyleSheet("color: #888; font-size: 11px;")
        inventory_row.addWidget(self._vision_status)
        inventory_row.addStretch()

        settings_layout.addLayout(inventory_row)

        # Output options row
        output_opts_row = QHBoxLayout()

        self._crop_check = QCheckBox("Crop to Content")
        self._crop_check.setChecked(True)
        self._crop_check.setToolTip("Crop output to bounding box of visible pixels")
        output_opts_row.addWidget(self._crop_check)

        output_opts_row.addSpacing(20)

        self._upscale_check = QCheckBox("Upscale Small")
        self._upscale_check.setToolTip(
            "Upscale extracted elements so the longest side\n"
            "matches the target resolution (never downscales)")
        output_opts_row.addWidget(self._upscale_check)

        self._upscale_target = QLineEdit("1024")
        self._upscale_target.setFixedWidth(55)
        self._upscale_target.setStyleSheet(
            "background: #1a1a2e; border: 1px solid #444; border-radius: 4px; "
            "padding: 3px; color: #ddd; font-size: 12px;")
        self._upscale_target.setEnabled(False)
        self._upscale_check.toggled.connect(self._upscale_target.setEnabled)
        output_opts_row.addWidget(self._upscale_target)
        output_opts_row.addWidget(QLabel("px"))

        output_opts_row.addSpacing(20)

        self._group_check = QCheckBox("Group by Source")
        self._group_check.setChecked(True)
        self._group_check.setToolTip(
            "Create a subfolder per source image and put\n"
            "all extracted elements into it")
        output_opts_row.addWidget(self._group_check)

        output_opts_row.addStretch()
        settings_layout.addLayout(output_opts_row)

        # Video output format row
        video_row = QHBoxLayout()

        self._video_mode_label = QLabel("🎬 Video Mode")
        self._video_mode_label.setStyleSheet(
            "color: #ff9500; font-weight: bold; font-size: 12px;")
        video_row.addWidget(self._video_mode_label)

        video_row.addSpacing(15)
        video_row.addWidget(QLabel("Output Format:"))
        self._video_format_combo = QComboBox()
        self._video_format_combo.addItems(["WebM (Alpha)", "PNG Sequence"])
        self._video_format_combo.setStyleSheet(
            "background: #1a1a2e; border: 1px solid #444; border-radius: 4px; "
            "padding: 3px; color: #ddd;")
        video_row.addWidget(self._video_format_combo)
        video_row.addStretch()

        # Initially hidden
        self._video_mode_label.setVisible(False)
        self._video_format_combo.setVisible(False)

        settings_layout.addLayout(video_row)

        # VRAM Optimization row
        vram_row = QHBoxLayout()

        self._vram_opt_check = QCheckBox("Optimize for VRAM (Run in Serial)")
        self._vram_opt_check.setToolTip(
            "Slows down processing by offloading inactive models to system RAM\n"
            "to prevent Out of Memory errors on lower-end GPUs."
        )
        vram_row.addWidget(self._vram_opt_check)

        vram_row.addSpacing(15)
        vram_row.addWidget(QLabel("VRAM Limit (GB):"))
        self._vram_spin = QSpinBox()
        self._vram_spin.setRange(6, 32)
        self._vram_spin.setValue(11)
        self._vram_spin.setStyleSheet(
            "QSpinBox { background: #1a1a2e; border: 1px solid #444; border-radius: 4px; padding: 3px; color: #ddd; }"
        )
        self._vram_spin.setEnabled(False)
        self._vram_opt_check.toggled.connect(self._vram_spin.setEnabled)
        self._vram_opt_check.toggled.connect(self._on_vram_limit_changed)
        self._vram_spin.valueChanged.connect(self._on_vram_limit_changed)
        vram_row.addWidget(self._vram_spin)
        
        vram_row.addStretch()
        settings_layout.addLayout(vram_row)

        # Output directory
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._output_edit = QLineEdit()
        self._output_edit.setStyleSheet(
            "background: #1a1a2e; border: 1px solid #444; border-radius: 4px; "
            "padding: 5px; color: #ddd;")
        self._output_edit.setPlaceholderText("Same as input (or choose a folder)")
        out_row.addWidget(self._output_edit, stretch=1)
        out_browse = QPushButton("...")
        out_browse.setFixedWidth(40)
        out_browse.setStyleSheet(self._button_style("#2a2a4a", "#3a3a6a"))
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(out_browse)
        settings_layout.addLayout(out_row)

        layout.addWidget(settings_group)

        # --- Process button ---
        self._process_btn = QPushButton("⚡ PROCESS")
        self._process_btn.setFixedHeight(45)
        self._process_btn.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self._process_btn.setStyleSheet(self._button_style("#003d5c", "#005f8a",
                                                            text_color="#00d4ff"))
        self._process_btn.clicked.connect(self._start_processing)
        self._process_btn.setEnabled(False)
        layout.addWidget(self._process_btn)

        # --- Progress bar ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333;
                border-radius: 6px;
                background: #1a1a2e;
                text-align: center;
                color: #ddd;
                height: 22px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #006688, stop:1 #00d4ff);
                border-radius: 5px;
            }
        """)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # --- Preview ---
        self._preview = PreviewPanel()
        layout.addWidget(self._preview, stretch=1)

        # --- Log ---
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        self._log.setStyleSheet(
            "background: #0d0d1a; border: 1px solid #333; border-radius: 6px; "
            "color: #999; font-family: Consolas; font-size: 11px;")
        layout.addWidget(self._log)

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    @staticmethod
    def _button_style(bg: str, hover: str, text_color: str = "#ddd") -> str:
        return """
            QPushButton {
                background: %s;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 8px 16px;
                color: %s;
                font-size: 13px;
            }
            QPushButton:hover {
                background: %s;
                border-color: #00d4ff;
            }
            QPushButton:disabled {
                background: #1a1a2e;
                color: #555;
                border-color: #333;
            }
        """ % (bg, text_color, hover)

    @staticmethod
    def _slider_style() -> str:
        return """
            QSlider {
                min-height: 24px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #333;
                height: 8px;
                background: #1a1a2e;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00d4ff;
                border: 1px solid #006688;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #33e0ff;
            }
            QSlider::handle:horizontal:disabled {
                background: #555;
                border-color: #444;
            }
        """

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _log_msg(self, msg: str):
        self._log.append(msg)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum())

    def _browse_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images or Videos", "",
            "Media (*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.m4v);;"
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;"
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.m4v);;"
            "All Files (*)")
        if paths:
            self._on_files_dropped(paths)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Media Folder")
        if folder:
            files = []
            for f in sorted(os.listdir(folder)):
                ext = os.path.splitext(f)[1].lower()
                if ext in MEDIA_EXTS:
                    files.append(os.path.join(folder, f))
            if files:
                self._on_files_dropped(files)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._output_edit.setText(folder)

    def _on_files_dropped(self, files: list):
        self._files = files
        self._process_btn.setEnabled(True)

        # Detect video mode
        n_vid = sum(1 for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTS)
        n_img = len(files) - n_vid
        self._is_video_mode = n_vid > 0

        # Update drop zone text
        parts = []
        if n_img:
            parts.append("%d image%s" % (n_img, "s" if n_img != 1 else ""))
        if n_vid:
            parts.append("%d video%s" % (n_vid, "s" if n_vid != 1 else ""))
        self._drop_zone.setText("📁  %s loaded" % ", ".join(parts))

        # Toggle video controls
        self._video_mode_label.setVisible(self._is_video_mode)
        self._video_format_combo.setVisible(self._is_video_mode)
        # Disable image-only controls in video mode
        self._crop_check.setEnabled(not self._is_video_mode)
        self._upscale_check.setEnabled(not self._is_video_mode)
        self._group_check.setEnabled(not self._is_video_mode)

        self._log_msg("Loaded %d files" % len(files))
        log.info("Loaded %d files (%d video, %d image)", len(files), n_vid, n_img)
        for f in files[:5]:
            self._log_msg("  • %s" % os.path.basename(f))
        if len(files) > 5:
            self._log_msg("  ... and %d more" % (len(files) - 5))

        # Show video info for first video file
        if self._is_video_mode:
            from rotobot_engine import RotobotEngine
            vid_file = [f for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTS][0]
            info = RotobotEngine.get_video_info(vid_file)
            self._log_msg("  🎬 %dx%d, %.1f fps, %d frames (%.1fs)" % (
                info['width'], info['height'], info['fps'],
                info['frames'], info['duration_s']))

    def _start_processing(self):
        if not self._files:
            return

        # Determine output dir
        out_dir = self._output_edit.text().strip()
        if not out_dir:
            out_dir = os.path.dirname(self._files[0])
        os.makedirs(out_dir, exist_ok=True)

        self._process_btn.setEnabled(False)
        self._progress_bar.setVisible(True)

        from rotobot_engine import RotobotEngine
        engine = RotobotEngine.get_instance()
        engine.set_vram_optimization(
            enabled=self._vram_opt_check.isChecked(),
            vram_gb=self._vram_spin.value()
        )

        # Get common options
        feather = self._feather_slider.value()
        use_color_key = self._colorkey_check.isChecked()
        kc = self._key_color
        key_color_rgb = (kc.red(), kc.green(), kc.blue())

        # --- Video mode ---
        if self._is_video_mode:
            video_files = [f for f in self._files
                           if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
            if not video_files:
                self._log_msg("No video files found")
                self._process_btn.setEnabled(True)
                self._progress_bar.setVisible(False)
                return

            # Process first video file
            vid_path = video_files[0]
            from rotobot_engine import RotobotEngine
            info = RotobotEngine.get_video_info(vid_path)

            self._progress_bar.setRange(0, info['frames'])
            self._progress_bar.setValue(0)
            self._log_msg("Processing video: %s (%d frames)..." % (
                os.path.basename(vid_path), info['frames']))
            log.info("Video processing started: %s, %d frames", vid_path, info['frames'])

            # Determine output format and path
            fmt_idx = self._video_format_combo.currentIndex()
            if fmt_idx == 1:  # PNG Sequence
                vid_format = 'png_seq'
                stem = os.path.splitext(os.path.basename(vid_path))[0]
                out_path = os.path.join(out_dir, stem + "_alpha_frames")
            else:  # WebM
                vid_format = 'webm'
                stem = os.path.splitext(os.path.basename(vid_path))[0]
                out_path = os.path.join(out_dir, stem + "_alpha.webm")

            self._worker = VideoWorker(
                video_path=vid_path,
                prompt=self._prompt_edit.text(),
                box_thresh=self._thresh_slider.value() / 100.0,
                text_thresh=0.20,
                refine=self._refine_combo.currentIndex(),
                invert=self._invert_check.isChecked(),
                output_path=out_path,
                output_format=vid_format,
                color_key=use_color_key,
                key_color=key_color_rgb,
                key_tolerance=float(self._tolerance_slider.value()),
                feather=feather,
            )
            self._worker.progress.connect(self._on_progress)
            self._worker.finished.connect(self._on_finished)
            self._worker.start()
            return

        # --- Image mode ---
        self._progress_bar.setRange(0, len(self._files))
        self._progress_bar.setValue(0)
        self._log_msg("Processing %d images..." % len(self._files))
        log.info("Processing started: %d images, output=%s", len(self._files), out_dir)

        # Get output options
        crop = self._crop_check.isChecked()
        upscale_to = 0
        if self._upscale_check.isChecked():
            try:
                upscale_to = int(self._upscale_target.text())
            except ValueError:
                upscale_to = 1024

        # --- Auto-Inventory mode ---
        if self._inventory_check.isChecked():
            self._log_msg("Starting Auto-Inventory (SAM2)...")
            self._progress_bar.setRange(0, len(self._files))
            self._worker = InventoryWorker(
                files=self._files,
                output_dir=out_dir,
                box_thresh=self._thresh_slider.value() / 100.0,
                text_thresh=0.20,
                refine=self._refine_combo.currentIndex(),
                feather=feather,
                crop=crop,
                upscale_to=upscale_to,
                detail=self._detail_slider.value(),
                group_by_source=self._group_check.isChecked(),
            )
            self._worker.progress.connect(self._on_progress)
            self._worker.file_done.connect(self._on_file_done)
            self._worker.finished.connect(self._on_finished)
            self._worker.start()
            return

        # --- Standard / Color Key mode ---
        self._worker = ProcessWorker(
            files=self._files,
            prompt=self._prompt_edit.text(),
            box_thresh=self._thresh_slider.value() / 100.0,
            text_thresh=0.20,
            refine=self._refine_combo.currentIndex(),
            invert=self._invert_check.isChecked(),
            output_dir=out_dir,
            color_key=use_color_key,
            key_color=key_color_rgb,
            key_tolerance=float(self._tolerance_slider.value()),
            feather=feather,
            crop=crop,
            upscale_to=upscale_to,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int, filename: str):
        self._progress_bar.setValue(current)
        self._log_msg("[%d/%d] %s" % (current, total, filename))

    def _on_file_done(self, filename: str, alpha):
        if alpha is not None:
            # Show the last completed file in preview
            for f in self._files:
                if os.path.basename(f) == filename:
                    self._preview.show_result(f, alpha)
                    break

    def _on_finished(self, success: int, failed: int, elapsed: float):
        self._progress_bar.setVisible(False)
        self._process_btn.setEnabled(True)
        self._log_msg("Done! %d/%d succeeded in %.1fs" % (
            success, success + failed, elapsed))
        log.info("Processing finished: %d/%d succeeded in %.1fs",
            success, success + failed, elapsed)
        if failed:
            self._log_msg("%d files had no detections" % failed)

    def _on_colorkey_toggled(self, checked: bool):
        """Enable/disable color key controls."""
        self._color_btn.setEnabled(checked)
        # Dim the AI prompt controls when color keying is active
        self._prompt_edit.setEnabled(not checked)
        self._thresh_slider.setEnabled(not checked)

    def _pick_key_color(self):
        """Open color picker dialog."""
        color = QColorDialog.getColor(
            self._key_color, self, "Pick Background Color to Remove")
        if color.isValid():
            self._set_key_color(color)

    def _set_key_color(self, color: QColor):
        """Set the key color and update the swatch button."""
        self._key_color = color
        self._color_btn.setStyleSheet(
            "background: %s; border: 2px solid #666; border-radius: 4px;" % color.name())
        self._colorkey_check.setChecked(True)
        self._log_msg("Key color set to %s" % color.name())

    def _on_vram_limit_changed(self, *args):
        """Update UI availability based on VRAM limits."""
        if self._vram_opt_check.isChecked():
            vram = self._vram_spin.value()
            if vram < 16:
                # Disable Auto-Inventory (needs large Vision model)
                if self._inventory_check.isChecked():
                    self._inventory_check.setChecked(False)
                self._inventory_check.setEnabled(False)
                self._inventory_check.setToolTip(
                    "Auto-Inventory requires LLaMA Vision,\n"
                    "which needs at least 16GB of VRAM."
                )
                self._vision_status.setText("Requires 16GB+ VRAM")
                self._vision_status.setStyleSheet("color: #ff5555; font-size: 11px;")
            else:
                self._inventory_check.setEnabled(True)
                self._inventory_check.setToolTip(
                    "Automatically segment the image into all visual elements\n"
                    "using SAM2, and extract each one individually.\n\n"
                    "NOTE: This requires an external LLaMA Vision API connected on port 5001.\n"
                    "This is an upcoming feature for end-users in a later version."
                )
                if not self._inventory_check.isChecked():
                    self._vision_status.setText("")
        else:
            self._inventory_check.setEnabled(True)
            if not self._inventory_check.isChecked():
                self._vision_status.setText("")

    def _on_inventory_toggled(self, checked: bool):
        """Toggle auto-inventory mode."""
        if checked:
            # Mutually exclusive with color key
            self._colorkey_check.setChecked(False)
            # Dim prompt (Vision API provides prompts)
            self._prompt_edit.setEnabled(False)
            self._thresh_slider.setEnabled(True)
            self._vision_status.setText("Will analyze images with Vision AI")
            self._vision_status.setStyleSheet("color: #00d4ff; font-size: 11px;")
        else:
            if not self._colorkey_check.isChecked():
                self._prompt_edit.setEnabled(True)
            self._vision_status.setText("")


# ============================================================================
# Main
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyle("Fusion")

    window = RotobotWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
