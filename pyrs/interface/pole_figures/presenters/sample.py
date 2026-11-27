"""
Sample presenter: load the STL, keep the model in step with the forms, redraw the mount figure.

Typical workflow:

1. :meth:`SamplePresenter.load_stl` reads the STL and puts the gauge at its bounding-box center.
2. Edits to the forms or the sample-to-stage matrix update the model and redraw the mount figure.
3. The predict and data presenters read ``model``, ``mesh`` and :meth:`SamplePresenter.mount`, and listen to
   ``changed``.

Example:
    >>> presenter = SamplePresenter(SampleTab(), model=SampleModel())  # doctest: +SKIP
    >>> presenter.load_stl("slab.stl")  # doctest: +SKIP
    True
"""

from pathlib import Path

from pydantic import ValidationError
from qtpy.QtCore import QObject, QTimer, Signal  # type: ignore[attr-defined]

from pyrs.pole_figures.core.geometry import Mount, MountConventions
from pyrs.pole_figures.core.rotation import check_rotation, nearest_rotation
from pyrs.pole_figures.core.stl_mesh import SampleMesh
from pyrs.pole_figures.plots.mount import draw_mount
from ..models.sample import SampleModel
from ..mount_matrix import IDENTITY9, rotate_mount
from ..tabs.sample_tab import SampleTab


class SamplePresenter(QObject):
    """
    Logic of the Sample tab.

    Args:
        view: The Sample tab.
        model: Starting inputs (``SampleModel()`` for the defaults).

    Attributes:
        mesh_loaded: Signal with the new SampleMesh after an STL was loaded.
        changed: Signal: the STL, gauge center, mount or view changed.
        status: Signal with a short message for the status bar.
        view: The Sample tab.
        model: The current inputs.
        mesh: The loaded sample, or None.
    """

    mesh_loaded = Signal(object)
    changed = Signal()
    status = Signal(str)

    def __init__(self, view: SampleTab, *, model: SampleModel) -> None:
        super().__init__(view)
        self.view = view
        self.model = model
        self.mesh: SampleMesh | None = None
        self._last_dir = str(Path.cwd())

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(120)  # debounce: redraw once when spin boxes settle
        self._redraw_timer.timeout.connect(self.redraw)

        for form in view.forms:
            form.set_values(self.model.model_dump())
            form.changed.connect(self._on_form_changed)
        view.open_button.clicked.connect(self._on_open)
        view.center_button.clicked.connect(self.center_gauge)
        view.center_button.setEnabled(False)
        for row in view.matrix_spins:
            for sp in row:
                sp.valueChanged.connect(lambda _v: self._on_matrix_edited())
        for i, btn in enumerate(view.rotate_buttons):
            btn.clicked.connect(lambda _c=False, i=i: self.rotate(i, view.rotate_angles[i].value()))
        view.reset_mount_button.clicked.connect(lambda: self.set_mount_matrix(IDENTITY9))
        view.set_matrix(self.model.mount_matrix)
        self._show_mount_note()

    def _on_open(self) -> None:
        path = self.view.ask_stl_path(self._last_dir)
        if path:
            self.load_stl(path)

    def load_stl(self, path: str | Path, *, center_gauge: bool = True) -> bool:
        """
        Load an STL (mm, sample frame) and show its size, volume and warnings.

        Args:
            path: The STL file.
            center_gauge: Put the gauge at the bounding-box center (default: True).

        Returns:
            True when loaded, False (after showing the error) when it cannot be read.
        """
        path = Path(path)
        try:
            mesh = SampleMesh.from_stl(path)
        except (OSError, ValueError) as exc:
            self.view.show_error("Could not read the STL", f"{path}\n\n{exc}")
            return False
        self.mesh = mesh
        self._last_dir = str(path.parent)
        self.model = self.model.model_copy(update={"stl_path": str(path)})
        self.view.path_label.setText(f"<b>{path.name}</b> in {path.parent.name}")  # a long path would not wrap
        self.view.path_label.setToolTip(str(path))
        warn = "".join(f"<br><span style='color:#b00020'>Warning: {w}</span>" for w in mesh.warnings())
        self.view.info_label.setText(
            f"{mesh.ntri} triangles, <b>{'closed' if mesh.closed else 'NOT closed'}</b>, volume {mesh.volume:.2f} mm3"
            f"<br>vx {mesh.lo[0]:g} to {mesh.hi[0]:g}, vy {mesh.lo[1]:g} to {mesh.hi[1]:g}, "
            f"vz {mesh.lo[2]:g} to {mesh.hi[2]:g} mm" + warn
        )
        self.view.center_button.setEnabled(True)
        if center_gauge:
            self.center_gauge()
        else:
            self.redraw()
        self.mesh_loaded.emit(mesh)
        self.changed.emit()
        self.status.emit(f"Loaded {mesh.describe()}")
        return True

    def center_gauge(self) -> None:
        """Put the gauge at the center of the bounding box (rounded to 0.001 mm). Does nothing without an STL."""
        if self.mesh is None:
            return
        c = (self.mesh.lo + self.mesh.hi) / 2
        vals = {
            "gauge_vx": round(float(c[0]), 3),
            "gauge_vy": round(float(c[1]), 3),
            "gauge_vz": round(float(c[2]), 3),
        }
        self.model = self.model.model_copy(update=vals)
        self.view.gauge_form.set_values(vals)
        self.redraw()
        self.changed.emit()

    def set_mount_matrix(self, m: tuple[float, ...]) -> None:
        """Use this sample-to-stage matrix (9 values, a rotation) everywhere."""
        self.model = self.model.model_copy(update={"mount_matrix": tuple(float(x) for x in m)})
        self.view.set_matrix(self.model.mount_matrix)
        self._show_mount_note()
        self._redraw_timer.start()
        self.changed.emit()

    def rotate(self, stage_axis: int, deg: float) -> None:
        """
        Turn the sample on the stage about a stage axis. The turn is added to the current matrix.

        Args:
            stage_axis: 0 for the phi axis, 1 for the chi axis, 2 for Q.
            deg: Angle (deg), right-hand rule about the stage axis. 0 does nothing.
        """
        if deg:
            self.set_mount_matrix(rotate_mount(self.model.mount_matrix, stage_axis, deg))

    def _on_matrix_edited(self) -> None:
        vals = self.view.matrix_values()
        try:
            check_rotation([vals[0:3], vals[3:6], vals[6:9]])
        except ValueError as exc:
            self.view.matrix_note.setText(f"<span style='color:#b00020'>{exc}. The last valid matrix is still in "
                                          "use.</span>")  # fmt: skip
            return
        m = nearest_rotation([vals[0:3], vals[3:6], vals[6:9]])
        self.model = self.model.model_copy(update={"mount_matrix": tuple(float(x) for x in m.ravel())})
        self._show_mount_note()
        self._redraw_timer.start()
        self.changed.emit()

    def _show_mount_note(self) -> None:
        ax = MountConventions(sample_to_instrument=self.model.mount_matrix).axes_at_zero()
        self.view.matrix_note.setText(f"At chi 0, phi 0: {ax.up} up along the phi axis, {ax.across} along the "
                                      f"chi axis, {ax.q} along Q ('~' = tilted, nearest axis).")  # fmt: skip

    def _on_form_changed(self) -> None:
        vals: dict[str, object] = {}
        for form in self.view.forms:
            vals.update(form.values())
        try:
            self.model = SampleModel.model_validate({**self.model.model_dump(), **vals})
        except ValidationError as exc:
            self.status.emit(f"Invalid input: {exc.errors()[0]['msg']}")
            return
        self._redraw_timer.start()
        self.changed.emit()

    def mount(self) -> Mount:
        """The mount at the preview omega and 2theta."""
        m = self.model
        return Mount(m.conventions(), omega=m.preview_omega, tth=m.preview_tth)

    def redraw(self) -> None:
        """Draw the mount figure at the preview orientation. Warns when the gauge center is outside the sample."""
        if self.mesh is None:
            return
        m = self.model
        fig = self.view.canvas.figure
        draw_mount(fig, self.mesh, center=m.gauge_center(), mount=self.mount(), chi_logged=m.preview_chi,
                   phi=m.preview_phi, view=m.view_from, vz_name=m.vz_name)  # fmt: skip
        self.view.canvas.draw_idle()
        if not self.gauge_inside():
            self.status.emit("The gauge center is outside the sample")

    def gauge_inside(self) -> bool:
        """True when an STL is loaded and the gauge center is inside it."""
        return self.mesh is not None and bool(self.mesh.contains(self.model.gauge_center()[None])[0])
