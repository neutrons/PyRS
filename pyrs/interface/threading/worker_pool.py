from typing import Dict, List
from qtpy.QtCore import QObject, QThread, Signal  # type:ignore
from pyrs.meta.decorators.Singleton import Singleton
import time


class Worker(QObject):
    finished = Signal()
    success = Signal(bool)
    result = Signal(object)

    target = None
    args = None

    def __init__(self, target, args=None):
        super().__init__()
        self.target = target
        self.args = args

    def run(self):
        """Long-running task."""
        try:
            results = self.target(self.args)
            # results.code = 200 # set to 200 for testing
            self.result.emit(results)
            self.success.emit(True)
        except Exception as e:  # noqa: BLE001
            # print stacktrace
            import traceback

            traceback.print_exc(e)
            self.result.emit(None)
            self.success.emit(False)
        self.finished.emit()


class TimmerWorker(QObject):
    finished = Signal()
    progress = Signal(int)

    _kill = False

    def __init__(self):
        super().__init__()

    def stop(self):
        self._kill = True

    def run(self):
        """inf running task."""
        segment = -1
        while not self._kill:
            time.sleep(10)
            segment = segment + 1
            self.progress.emit(segment)


@Singleton
class WorkerPool:
    max_threads = 8
    threads: Dict[Worker, QThread] = {}
    worker_queue: List[Worker] = []

    def createWorker(self, target, args):
        return Worker(target=target, args=args)

    def createTimmerWorker(self):
        return TimmerWorker()

    def _dequeueWorker(self, worker):
        self.threads.pop(worker)
        if len(self.worker_queue) > 0:
            self.submitWorker(self.worker_queue.pop())

    def submitWorker(self, worker):
        if len(self.threads) >= self.max_threads:
            # add to queue
            self.worker_queue.append(worker)
        else:
            # spawn thread and deligate
            thread = QThread()
            # WARN: maybe the worker shouldnt be a key, not sure how equivelence is solved
            self.threads[worker] = thread
            worker.moveToThread(thread)
            # Step 5: Connect signals and slots
            thread.started.connect(worker.run)

            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)

            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(lambda: self._dequeueWorker(worker))

            # Step 6: Start the thread
            thread.start()
