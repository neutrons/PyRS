# a collection of helper methdos for GUI
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from pyrs.utilities import checkdatatypes
from pyrs.utilities.config import Config
from qtpy.QtWidgets import QLineEdit, QFileDialog, QMessageBox, QComboBox  # type:ignore

if TYPE_CHECKING:
    from pyrs.core.workspaces import HidraWorkspace

_logger = logging.getLogger(__name__)


def impose_extension(filename: str, extension: str) -> str:
    """Append ``extension`` (e.g. '.nxs') to ``filename`` if it has no extension.

    A name that already has an extension is returned unchanged: replacing it
    would cut a dotted name short (``HB2B_1327.v2`` would become
    ``HB2B_1327.nxs``). Whether an existing extension is acceptable is
    `resolve_save_target`'s decision, not this function's.
    """
    path = Path(filename)
    return filename if path.suffix else str(path.with_name(path.name + extension))


def confirm_replace(parent, filename: str) -> bool:
    """Ask whether to replace an existing file.

    Args:
        parent: Parent widget for the question box.
        filename: The existing file that would be replaced.

    Returns:
        True if the user chose to replace it.
    """
    answer = QMessageBox.question(
        parent,
        "Replace existing file?",
        f"'{Path(filename).name}' already exists. Replace it?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    return answer == QMessageBox.Yes


def resolve_save_target(
    parent, filename: str, *, extension: str, format_name: str, also_accepted: Tuple[str, ...] = ()
) -> Optional[str]:
    """The file a save dialog's choice should actually be written to, or None if
    nothing should be written.

    A save dialog asks about overwriting the name as typed, so a target that
    differs from it must be checked here:

    - a name ending in ``extension`` (or one of ``also_accepted``) is used as typed;
    - a name with no extension gets ``extension`` appended, and if that file
      already exists the user is asked before it is replaced;
    - a name with any other extension is refused: writing one format under
      another format's extension would mislabel the file.

    Refusals and declined replacements are reported to the user here.

    Args:
        parent: Parent widget for any message or question box.
        filename: The name chosen in the save dialog.
        extension: The format's configured extension, e.g. '.nxs'.
        format_name: The format's name as the user knows it, e.g. 'NXstress'.
        also_accepted: Further extensions accepted as typed, e.g. ('.hdf5',).

    Returns:
        The path to write, or None if the save should not go ahead.
    """
    suffix = Path(filename).suffix
    if suffix == extension or suffix in also_accepted:
        return filename
    if suffix:
        pop_message(
            parent,
            f"Unable to save {format_name} format as '*{suffix}'",
            f"Use a file name ending in '{extension}'; nothing was saved to '{Path(filename).name}'.",
            "error",
        )
        return None
    target = impose_extension(filename, extension)
    if Path(target).exists() and not confirm_replace(parent, target):
        return None
    return target


def report_save_failure(parent, target: str, save_err: Exception) -> None:
    """Log a failed save with its traceback and report it in an error box.

    Every save action reports a failure through this, whatever the format, so
    that a failed save looks the same to the user wherever it happens.

    Args:
        parent: Parent widget for the error message box.
        target: The file that was being written.
        save_err: The exception the write raised.
    """
    _logger.exception("Failed to save '%s'", target, exc_info=save_err)
    pop_message(parent, f"Failed to save '{Path(target).name}'", str(save_err), "error")


def nxstress_save_available(ws: Optional["HidraWorkspace"]) -> bool:
    """Whether NXstress output can be offered for a workspace: NXstress must be
    enabled in the config, and the workspace must have the instrument geometry
    that `NXstress.write` requires.

    Args:
        ws: The loaded workspace, or None if nothing is loaded.

    Returns:
        True if a *Save as NXstress...* action should be enabled.
    """
    return bool(Config["nxstress.enable"]) and ws is not None and ws.get_instrument_setup() is not None


def write_nxstress_or_report(parent, filename: str, write: Callable[[str], None]) -> bool:
    """Write an NXstress file, reporting any failure in a message box instead of
    raising.

    The destination is resolved by `resolve_save_target` (configured NXstress
    extension, overwrite confirmation, refusal of another format's extension).
    The file is then written to a sibling ``<stem>.partial<ext>`` that replaces
    the destination only on success: an NXstress write truncates its target on
    open, so writing in place would destroy an existing file on any failure. If
    that sibling already exists -- for example, left by an interrupted save --
    the save is refused rather than overwriting it, and the destination is left
    untouched. Raising is avoided because an uncaught exception in a Qt slot
    aborts the application under PyQt6.

    Args:
        parent: Parent widget for the error message box.
        filename: Destination chosen by the user.
        write: Writes the NXstress file to the path it is given.

    Returns:
        True if the file was written; False if nothing was written, the reason
        having been reported (or the user having declined to replace a file).
    """
    resolved = resolve_save_target(parent, filename, extension=Config["nxstress.extension"], format_name="NXstress")
    if resolved is None:
        return False
    target = Path(resolved)
    partial = target.with_name(f"{target.stem}.partial{target.suffix}")
    if partial.exists():
        pop_message(
            parent,
            f"Not saved: '{partial.name}' already exists",
            f"Saving writes '{partial.name}' first, and a file of that name is already there -- possibly left "
            "by an interrupted save. Remove or rename it, then save again.",
            "error",
        )
        return False
    try:
        write(str(partial))
        os.replace(partial, target)
        return True
    except Exception as save_err:  # noqa: BLE001 -- see docstring
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            _logger.exception("Failed to remove partial NXstress file '%s'", partial)
        report_save_failure(parent, str(target), save_err)
        return False


def browse_dir(parent, caption, default_dir):
    """Browse a directory
    :param parent:
    :param caption:
    :param default_dir:
    :return: non-empty string for selected directory; empty string for canceled operation
    """
    # check inputs
    assert isinstance(parent, object), "Parent {} must be of some object.".format(parent)
    checkdatatypes.check_string_variable("File browsing title/caption", caption)
    checkdatatypes.check_file_name(default_dir, check_exist=False, is_dir=True)

    # get directory
    chosen_dir = QFileDialog.getExistingDirectory(
        parent, caption, default_dir, options=QFileDialog.DontUseNativeDialog
    )
    print("[DB...BAT] Chosen dir: {} of type {}".format(chosen_dir, type(chosen_dir)))
    chosen_dir = str(chosen_dir).strip()

    return chosen_dir


def browse_file(parent, caption, default_dir, file_filter, file_list=False, save_file=False):
    """Browse a file or files

    Parameters
    ----------
    parent : QWindow
        parent window
    caption : str
        Dialog box's caption
    default_dir : str
        Default directory to browse the file
    file_filter : str
        File type filter
    file_list : bool
        Flag for browse a list of files; otherwise, single file
    save_file : bool

    Returns
    -------
    str or ~list or None
        File name or A list of file names or None for user to cancel the operation

    """
    # check inputs
    # assert isinstance(parent, object), 'Parent {} must be of some object.'.format(parent)
    # checkdatatypes.check_string_variable('File browsing title/caption', caption)
    # checkdatatypes.check_file_name(default_dir, check_exist=False, is_dir=True)
    # checkdatatypes.check_bool_variable('Flag for browse a list of files to load', file_list)
    # checkdatatypes.check_bool_variable('Flag to select loading or saving file', save_file)
    if file_filter is None:
        file_filter = "All Files (*.*)"
    else:
        checkdatatypes.check_string_variable("File filter", file_filter)
        file_filter = "{};;All Files (*.*)".format(file_filter)

    if save_file:
        # browse file name to save to
        save_set = QFileDialog.getSaveFileName(
            parent, caption=caption, directory=default_dir, filter=file_filter, options=QFileDialog.DontUseNativeDialog
        )
        if isinstance(save_set, tuple):
            # returned include both file name and filter
            file_name = str(save_set[0])
        else:
            file_name = str(save_set)

    elif file_list:
        # browse file names to load
        open_set = QFileDialog.getOpenFileNames(
            parent, caption, default_dir, file_filter, options=QFileDialog.DontUseNativeDialog
        )

        if isinstance(open_set, tuple):
            file_name_list = open_set[0]
        else:
            file_name_list = open_set

        if len(file_name_list) == 0:  # user cancel browser
            return None
        else:
            return file_name_list

    else:
        # browse single file name
        open_set = QFileDialog.getOpenFileName(
            parent, caption, default_dir, file_filter, options=QFileDialog.DontUseNativeDialog
        )

        if isinstance(open_set, tuple):
            file_name = open_set[0]
        else:
            file_name = open_set

    # check result for single file whether user cancels operation
    if len(file_name) == 0:
        return None

    return str(file_name)


def parse_combo_box(combo_box, data_type):
    """

    Parameters
    ----------
    combo_box
    data_type

    Returns
    -------

    """
    assert isinstance(combo_box, QComboBox), (
        "Method parse_combo_box expects 0-th input {} to be a QComboBox instance but not a {}".format(
            combo_box, type(combo_box)
        )
    )
    assert isinstance(data_type, type), (
        "Method parse_line_edit expects 1-st input {} to be a type but not a {}".format(data_type, type(data_type))
    )

    # parse
    input_str = str(combo_box.currentText()).strip()
    # cast to specified type
    try:
        return data_type(input_str)
    except ValueError:
        return None


def parse_integer(int_str):
    """
    parse integer from a string or a LineEdit
    :param int_str:
    :return:
    """
    if isinstance(int_str, QLineEdit):
        # QLineEdit: get the string out of it
        int_str = str(int_str.text())
    else:
        # Then it has to be a string
        checkdatatypes.check_string_variable("Integer string", int_str)

    try:
        int_value = int(int_str)
    except ValueError as value_error:
        raise RuntimeError("Unable to parse {0} to integer due to {1}".format(int_str, value_error))

    return int_value


def parse_rigorous_int_string(int_str):
    """
    parse a string which must be an integer but  not anything can be converted to integer
    :param int_str:
    :return:
    """
    checkdatatypes.check_string_variable("Integer in string", int_str)

    # negative?
    if int_str.startswith("-"):
        sign = -1
        int_str = int_str.split("-")[1]
    else:
        sign = 1

    # must be an integer
    if int_str.isdigit() is False:
        raise ValueError("{} cannot be recognized as an integer rigorously".format(int_str))

    # convert
    try:
        int_number = sign * int(int_str)
    except ValueError as val_err:
        raise ValueError("Unable to convert string {} to an integer: {}".format(int_str, val_err))

    return int_number


def parse_integers(int_list_string):
    """parse a list of integers.  A range includes both its start and its end, as in combine runs
    and manual reduction (#931): "1:4" is 1, 2, 3, 4.
    example 1:4, 6:12, 8:12
    :param int_list_string:
    :return: sorted list of unique ints
    """
    checkdatatypes.check_string_variable("Integer list (string)", int_list_string)

    # remove unnecessary spaces
    int_list_string = int_list_string.replace(" ", "")

    # split by ,
    int_range_list = int_list_string.split(",")

    # parse to integers
    int_list = list()
    try:
        for int_range in int_range_list:
            column_counts = int_range.count(":") + int_range.count("-")

            if column_counts == 0:
                # single value
                int_list.append(parse_rigorous_int_string(int_range))

            elif column_counts == 1:
                # given a range
                if "-" in int_range:
                    int_str_list = int_range.split("-")
                else:
                    int_str_list = int_range.split(":")

                start_int = parse_rigorous_int_string(int_str_list[0])
                end_int = parse_rigorous_int_string(int_str_list[1])
                int_list.extend(range(start_int, end_int + 1))

            else:
                # bad inputs
                raise ValueError("{0} has too many : to recognize".format(int_range))
    except ValueError as val_err:
        raise RuntimeError('Unable to parse integer list "{}" due to {}'.format(int_list_string, val_err))

    # remove additional integers
    int_list = list(set(int_list))

    int_list.sort()

    return int_list


def pop_message(parent, message, detailed_message=None, message_type="error"):
    """pop up a message with specified message type such as error, warning, info...
    :param parent:
    :param message:
    :param detailed_message: detailed message optionally shown to user
    :param message_type: str as ['error', 'warning', 'info'] but NOT case sensitive
    :return:
    """
    message_type = message_type.lower()
    if message_type not in ["error", "warning", "info"]:
        raise TypeError("Message type {0} is not supported.".format(message_type))

    # check types
    checkdatatypes.check_string_variable("(Main) message to show", message)
    if detailed_message is not None:
        checkdatatypes.check_string_variable("(Detailed) message to show", detailed_message)

    # create a QMessageBox
    msg_box = QMessageBox()

    # set information type
    if message_type == "info":
        msg_box.setIcon(QMessageBox.Information)
    elif message_type == "error":
        msg_box.setIcon(QMessageBox.Critical)
    elif message_type == "warning":
        msg_box.setIcon(QMessageBox.Warning)

    # set text
    msg_box.setText(message)
    if detailed_message is not None:
        msg_box.setDetailedText(detailed_message)  # another button
    msg_box.setWindowTitle("PyRS Message")

    # box
    msg_box.setStandardButtons(QMessageBox.Ok)

    ret_val = msg_box.exec_()
    print("Message box return value: {}".format(ret_val))
