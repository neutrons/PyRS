"""Worker (pyrs.interface.threading.worker_pool): a failing task is reported, and finished always fires"""

from pyrs.interface.threading.worker_pool import Worker


def _run(target):
    """Run a Worker in this thread and record what it emits."""
    emitted = {"result": [], "success": [], "error": [], "finished": 0}
    worker = Worker(target, args=None)
    worker.result.connect(emitted["result"].append)
    worker.success.connect(emitted["success"].append)
    worker.error.connect(emitted["error"].append)
    worker.finished.connect(lambda: emitted.__setitem__("finished", emitted["finished"] + 1))
    worker.run()
    return emitted


def test_worker_success_emits_result_and_finished():
    emitted = _run(lambda args: 42)
    assert emitted == {"result": [42], "success": [True], "error": [], "finished": 1}


def test_worker_failure_emits_error_and_finished(capsys):
    """It used to call traceback.print_exc(e), which raised TypeError and emitted nothing"""

    # Arrange
    def fails(args):
        raise ValueError("starting value outside bounds")

    # Act
    emitted = _run(fails)

    # Assert - the message reaches the GUI, and the traceback is still printed
    assert emitted["error"] == ["ValueError: starting value outside bounds"]
    assert emitted["result"] == [None] and emitted["success"] == [False] and emitted["finished"] == 1
    assert "starting value outside bounds" in capsys.readouterr().err
