"""
Plot panel: a plot-type dropdown, a plane dropdown when the type needs one, the plot, and a save button.

A figure is drawn at its designed size (inches) and scaled as a whole (fonts included) to fit the panel, so the screen
looks like the saved PNG. Without this the canvas squeezes the figure to the widget and titles run into each other.

Typical workflow:

1. The presenter lists the plot types and planes with :meth:`PlotPanel.set_choices`.
2. On ``selected(type, plane)`` it builds the plot and calls :meth:`PlotPanel.show_figure`,
   :meth:`PlotPanel.show_gif`, :meth:`PlotPanel.show_image` or :meth:`PlotPanel.show_message`.
3. The save button writes the shown plot through :meth:`PlotPanel.save_to`.

Example:
    >>> panel = PlotPanel()  # doctest: +SKIP
    >>> panel.set_choices(["Correction", "Path"], ["111", "200"], per_plane={"Correction"})  # doctest: +SKIP
    >>> panel.selected.connect(lambda kind, plane: panel.show_figure(build(kind, plane)))  # doctest: +SKIP
"""

import re
import shutil
from pathlib import Path
from typing import Literal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from qtpy.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QHideEvent, QMovie, QPixmap, QShowEvent
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyrs.pole_figures.plots.save import save_figure

#: what the panel shows
PanelKind = Literal["figure", "gif", "image", "none"]
ZOOM_FIT = "Fit"  # scale the plot to the panel
ZOOMS = [ZOOM_FIT, "50%", "75%", "100%", "125%", "150%"]
BASE_DPI = 100.0  # 100% = 100 pixels per figure inch
MIN_FIT_DPI = 40.0
NO_PLOTS = "Press Predict corrections to make the plots."


def default_filename(name: str, suffix: str) -> str:
    """
    File name from a plot name: runs of other characters than letters and digits become "_", in lower case.

    Args:
        name: Plot name.
        suffix: File suffix, with the dot.

    Returns:
        The file name, "plot" + suffix when nothing is left of the name.

    Example:
        >>> default_filename("{200} correction vs phi", ".png")
        '200_correction_vs_phi.png'
    """
    stem = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "plot"
    return stem + suffix


class PlotPanel(QWidget):
    """
    Shows one plot at a time, chosen by type and (for per-plane types) plane.

    Args:
        parent: Parent widget, or None (default: None).

    Attributes:
        selected: Signal with the plot type and plane label ("" when the type has no plane). The owner builds the
            plot and calls ``show_figure``, ``show_gif``, ``show_image`` or ``show_message``.
        saved: Signal with the path the active plot was saved to.
        canvas: Canvas of the shown figure, or None.
        movie: The playing GIF, or None.
        gif_path: File of the shown GIF, or None.
        image_path: File of the shown picture, or None.
        kind: What is shown: "figure", "gif", "image" or "none".
    """

    selected = Signal(str, str)
    saved = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.type_combo = QComboBox()
        self.type_combo.setMinimumContentsLength(22)
        self.plane_label = QLabel("Plane:")
        self.plane_combo = QComboBox()
        self.plane_combo.setMinimumContentsLength(8)
        self.type_combo.currentTextChanged.connect(self._on_type)
        self.plane_combo.currentTextChanged.connect(lambda _t: self._emit())
        self.zoom = QComboBox()
        self.zoom.addItems(ZOOMS)
        self.zoom.currentTextChanged.connect(lambda _t: self._rescale())
        self.save_button = QPushButton("Save plot...")
        self.save_button.clicked.connect(self._on_save)
        self.placeholder = QLabel(NO_PLOTS)
        self._per_plane: set[str] = set()
        # while the window is being resized, redraw once it settles (a figure redraw can take a few 100 ms)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(120)
        self._resize_timer.timeout.connect(self._rescale)

        self.canvas: FigureCanvasQTAgg | None = None
        self.toolbar: NavigationToolbar2QT | None = None
        self.movie: QMovie | None = None
        self.gif_path: Path | None = None
        self.image_path: Path | None = None
        self._pixmap: QPixmap | None = None
        self._natural = (1.0, 1.0)  # figure inches, or GIF pixels
        self._kind: PanelKind = "none"

        top = QHBoxLayout()
        top.addWidget(QLabel("Plot type:"))
        top.addWidget(self.type_combo, stretch=2)
        top.addWidget(self.plane_label)
        top.addWidget(self.plane_combo, stretch=1)
        top.addStretch(1)
        self._corner = QHBoxLayout()  # extra controls (e.g. a cells/interpolated toggle), right of the dropdowns
        top.addLayout(self._corner)
        top.addWidget(QLabel("Size:"))
        top.addWidget(self.zoom)
        top.addWidget(self.save_button)
        self._toolbar_row = QHBoxLayout()
        self.area = QScrollArea()
        self.area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.area.setWidgetResizable(False)
        self.area.setStyleSheet("QScrollArea { background: white; border: none }")
        self.area.viewport().installEventFilter(self)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(self._toolbar_row)
        lay.addWidget(self.area, stretch=1)
        self.set_choices([], [], set())

    def add_corner_widget(self, w: QWidget) -> None:
        """Put a control at the top right, before the size and save controls."""
        self._corner.addWidget(w)

    def set_choices(self, types: list[str], planes: list[str], per_plane: set[str]) -> None:
        """
        Set the plot types, the planes, and which types need a plane. Emits nothing. Keeps current entries.

        Shows the no-plots message when ``types`` is empty.

        Args:
            types: Plot types, in dropdown order.
            planes: Plane labels (hkl), in dropdown order.
            per_plane: The types that need a plane.
        """
        self._per_plane = set(per_plane)
        for combo, items in ((self.type_combo, types), (self.plane_combo, planes)):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if keep in items:
                combo.setCurrentText(keep)
            combo.blockSignals(False)
            combo.setEnabled(bool(items))
        self._update_plane_visible()
        if not types:
            self.show_message(NO_PLOTS)

    def types(self) -> list[str]:
        """The plot types in the dropdown."""
        return [self.type_combo.itemText(i) for i in range(self.type_combo.count())]

    def planes(self) -> list[str]:
        """The planes in the dropdown."""
        return [self.plane_combo.itemText(i) for i in range(self.plane_combo.count())]

    def current(self) -> tuple[str, str]:
        """Current plot type and plane ("" when the type has no plane)."""
        t = str(self.type_combo.currentText())
        return t, (str(self.plane_combo.currentText()) if t in self._per_plane else "")

    def current_name(self) -> str:
        """Plane and plot type as one name, for example "200 Correction"."""
        t, plane = self.current()
        return f"{plane} {t}".strip()

    def select(self, plot_type: str, plane: str = "") -> None:
        """
        Choose a plot type (and plane) and emit ``selected`` once.

        Args:
            plot_type: Plot type to choose.
            plane: Plane to choose, or "" to keep the current plane (default: "").
        """
        for combo in (self.type_combo, self.plane_combo):
            combo.blockSignals(True)
        self.type_combo.setCurrentText(plot_type)
        if plane:
            self.plane_combo.setCurrentText(plane)
        for combo in (self.type_combo, self.plane_combo):
            combo.blockSignals(False)
        self._update_plane_visible()
        self._emit()

    def _update_plane_visible(self) -> None:
        show = self.type_combo.currentText() in self._per_plane
        self.plane_label.setVisible(show)
        self.plane_combo.setVisible(show)

    def _on_type(self, _t: str) -> None:
        self._update_plane_visible()
        self._emit()

    def _emit(self) -> None:
        t, plane = self.current()
        if t:
            self.selected.emit(t, plane)

    def _clear(self) -> None:
        if self.movie is not None:
            self.movie.stop()
            self.movie.deleteLater()
            self.movie = None
        if self.toolbar is not None:
            # a cached figure keeps its canvas callbacks, so drop the old toolbar's before deleting it
            for name in ("_id_press", "_id_release", "_id_drag"):
                cid = getattr(self.toolbar, name, None)
                if cid is not None and self.canvas is not None:
                    self.canvas.mpl_disconnect(cid)
            self._toolbar_row.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None
        if self.canvas is not None:
            self.canvas._draw_pending = False  # type: ignore[attr-defined]  # no queued redraw of a deleted canvas
        old = self.area.takeWidget()
        if old is not None:
            old.deleteLater()
        self.canvas = None
        self.gif_path = None
        self.image_path = None
        self._pixmap = None

    def show_message(self, text: str) -> None:
        """Show a note in place of a plot (nothing to save)."""
        self._clear()
        self.placeholder = QLabel(text)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setStyleSheet("color: #555; font-size: 11pt")
        self.area.setWidget(self.placeholder)
        self._kind = "none"
        self.save_button.setEnabled(False)
        self._rescale()

    def message(self) -> str:
        """The shown note, or "" while a plot is shown."""
        return str(self.placeholder.text()) if self._kind == "none" else ""

    def show_figure(self, fig: Figure) -> None:
        """Show a figure at its designed size, scaled as a whole by the zoom setting, with the navigation toolbar."""
        self._clear()
        w, h = fig.get_size_inches()
        self._natural = (float(w), float(h))
        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._toolbar_row.addWidget(self.toolbar)
        self.area.setWidget(self.canvas)
        self._kind = "figure"
        self.save_button.setEnabled(True)
        self._rescale()

    def show_gif(self, path: str | Path) -> None:
        """Play a GIF (scaled like a figure). It is paused while the panel is hidden (another tab is shown)."""
        self._clear()
        self.gif_path = Path(path)
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.movie = QMovie(str(path))
        self.movie.jumpToFrame(0)
        size = self.movie.currentImage().size()
        self._natural = (float(max(size.width(), 1)), float(max(size.height(), 1)))
        label.setMovie(self.movie)
        self.area.setWidget(label)
        self._kind = "gif"
        self.save_button.setEnabled(True)
        self._rescale()
        self.movie.start()
        if not self.isVisible():
            self.movie.setPaused(True)  # resumed by showEvent

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802 (Qt override)
        """Pause the GIF while the panel is hidden, so a hidden tab does not decode frames."""
        if self.movie is not None and self.movie.state() == QMovie.MovieState.Running:
            self.movie.setPaused(True)
        super().hideEvent(event)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 (Qt override)
        """Resume a GIF paused by ``hideEvent``."""
        if self.movie is not None and self.movie.state() == QMovie.MovieState.Paused:
            self.movie.setPaused(False)
        super().showEvent(event)

    def show_image(self, path: str | Path) -> None:
        """Show a picture file (e.g. an MTEX PNG), scaled like a figure. A file Qt cannot read shows a note."""
        self._clear()
        pix = QPixmap(str(path))
        if pix.isNull():
            self.show_message(f"Cannot show {Path(path).name}.")
            return
        self.image_path, self._pixmap = Path(path), pix
        self._natural = (float(pix.width()), float(pix.height()))
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.area.setWidget(label)
        self._kind = "image"
        self.save_button.setEnabled(True)
        self._rescale()

    @property
    def kind(self) -> PanelKind:
        """'figure', 'gif', 'image' or 'none'."""
        return self._kind

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt override)
        """
        Rescale the plot when the viewport is resized and the zoom is Fit. A figure redraws once resizing settles.

        Args:
            obj: The object the event is for.
            event: The event.

        Returns:
            The base class result: the event is never filtered here.
        """
        if obj is self.area.viewport() and event.type() == QEvent.Type.Resize and self.zoom.currentText() == ZOOM_FIT:
            if self._kind == "figure":
                self._resize_timer.start()  # restarts on every resize event: one redraw at the end
            else:
                self._rescale()  # GIFs, images and messages scale cheaply
        return super().eventFilter(obj, event)

    def _scale(self, per_unit: float) -> float:
        """Pixels per natural unit (figure inch or GIF pixel) for the chosen zoom."""
        z = self.zoom.currentText()
        if z != ZOOM_FIT:
            return per_unit * float(z.rstrip("%")) / 100
        vp = self.area.viewport().size()
        w, h = self._natural
        fit = min((vp.width() - 6) / w, (vp.height() - 6) / h)
        return max(fit, per_unit * MIN_FIT_DPI / BASE_DPI)

    def _rescale(self) -> None:
        # Fit: no scroll bars (they would shrink the viewport and the fit would chase them)
        policy = (Qt.ScrollBarPolicy.ScrollBarAlwaysOff if self.zoom.currentText() == ZOOM_FIT
                  else Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # fmt: skip
        self.area.setHorizontalScrollBarPolicy(policy)
        self.area.setVerticalScrollBarPolicy(policy)
        if self._kind == "figure" and self.canvas is not None:
            d = self._scale(BASE_DPI)
            fig = self.canvas.figure
            w, h = self._natural
            fig.set_size_inches(w, h, forward=False)
            fig._original_dpi = d  # type: ignore[attr-defined]  # matplotlib rescales by the screen pixel ratio
            fig.set_dpi(d * self.canvas.device_pixel_ratio)
            self.canvas.setFixedSize(QSize(int(round(w * d)), int(round(h * d))))
            self.canvas.draw_idle()
        elif self._kind == "gif" and self.movie is not None:
            s = self._scale(1.0)
            w, h = self._natural
            size = QSize(int(round(w * s)), int(round(h * s)))
            self.movie.setScaledSize(size)
            widget = self.area.widget()
            if widget is not None:
                widget.setFixedSize(size)
        elif self._kind == "image" and self._pixmap is not None:
            s = self._scale(1.0)
            w, h = self._natural
            size = QSize(max(1, int(round(w * s))), max(1, int(round(h * s))))
            widget = self.area.widget()
            if isinstance(widget, QLabel):
                widget.setPixmap(self._pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation))  # fmt: skip
                widget.setFixedSize(size)
        elif self._kind == "none":
            vp = self.area.viewport().size()
            self.placeholder.setFixedSize(QSize(max(vp.width() - 6, 100), max(vp.height() - 6, 100)))

    def save_to(self, path: str | Path) -> None:
        """
        Save the active plot: a figure as PNG/PDF/SVG (by suffix, 130 dpi), a GIF or picture as a copy.

        Args:
            path: Destination file.

        Raises:
            ValueError: If no plot is shown.
        """
        path = Path(path)
        if self._kind == "figure" and self.canvas is not None:
            save_figure(self.canvas.figure, path, dpi=130, bbox_inches="tight")
            self._rescale()  # savefig can reset the figure dpi
        elif self._kind == "gif" and self.gif_path is not None:
            shutil.copyfile(self.gif_path, path)
        elif self._kind == "image" and self.image_path is not None:
            shutil.copyfile(self.image_path, path)
        else:
            raise ValueError("no plot to save")
        self.saved.emit(str(path))

    def _on_save(self) -> None:
        name = self.current_name()
        if self._kind == "gif":
            path, _ = QFileDialog.getSaveFileName(self, "Save GIF", default_filename(name, ".gif"), "GIF (*.gif)")
        elif self._kind == "image" and self.image_path is not None:
            suffix = self.image_path.suffix
            path, _ = QFileDialog.getSaveFileName(self, "Save plot", self.image_path.name, f"Image (*{suffix})")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Save plot", default_filename(name, ".png"),
                                                  "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")  # fmt: skip
        if path:
            self.save_to(path)
