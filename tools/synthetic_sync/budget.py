"""Single-process, resumable numerical-attempt budgets for POSIX experiments."""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import signal
import threading
import time

try:
    import fcntl
except ImportError:
    fcntl = None


class BudgetExceeded(BaseException):
    """Stop an experiment without solver exception handlers swallowing the cap."""


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


class ExperimentBudget:
    """Reserve every call before execution; retain exact inputs and completed records.

    One writer, main thread, POSIX only. Limits cover calls made through attempt(),
    not arbitrary uninstrumented solver calls or model tokens. Wall budget means
    cumulative active attempt time. Interrupted calls consume their reserved time.
    Python signal deadlines can be delayed by native calls; use an outer process
    timeout when a hard wall limit is required.
    """

    def __init__(self, path, *, metadata, max_calls, wall_seconds, per_call_seconds):
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        if any(not math.isfinite(x) or x <= 0 for x in (wall_seconds, per_call_seconds)):
            raise ValueError("Time limits must be finite and positive")
        self.path = Path(path)
        self.header = dict(kind="budget", version=1, metadata=metadata,
                           max_calls=max_calls, wall_seconds=wall_seconds,
                           per_call_seconds=per_call_seconds)
        _json(self.header)
        self.file = None
        self.rows = []
        self.active = False

    def __enter__(self):
        if fcntl is None:
            raise OSError("Experiment budgets currently require POSIX file locks and signals")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.file.seek(0)
            self.rows = [json.loads(line) for line in self.file if line.strip()]
            if self.rows:
                if self.rows[0] != self.header:
                    raise ValueError("Ledger metadata or limits changed; use a new experiment ledger")
            else:
                self._append(self.header)
        except BaseException:
            self.file.close()
            self.file = None
            raise
        return self

    def __exit__(self, *args):
        self.file.close()
        self.file = None

    def _append(self, row):
        text = _json(row)
        self.file.write(text + "\n")
        self.file.flush()
        import os
        os.fsync(self.file.fileno())
        self.rows.append(json.loads(text))

    def _key(self, label, request):
        return hashlib.sha256(_json(dict(label=label, request=request,
                                        metadata=self.header["metadata"])).encode()).hexdigest()

    def cached(self, label, request):
        """Return a matching completed JSON record, never a live solver object."""
        key = self._key(label, request)
        return next((row["result"] for row in reversed(self.rows)
                     if row.get("kind") == "completed" and row["key"] == key), None)

    @contextmanager
    def attempt(self, label, request):
        if self.file is None or self.active:
            raise RuntimeError("Open the budget first; nested attempts are not supported")
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Attempt deadlines require the main thread")
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise RuntimeError("An existing alarm owns the process deadline")
        starts = [row for row in self.rows if row["kind"] == "started"]
        finished = {row["attempt"]: row for row in self.rows
                    if row["kind"] in ("completed", "failed")}
        spent = sum(finished[row["attempt"]]["elapsed_s"] if row["attempt"] in finished
                    else row["time_limit_s"] for row in starts)
        remaining = self.header["wall_seconds"] - spent
        if len(starts) >= self.header["max_calls"] or remaining <= 0:
            raise BudgetExceeded("Experiment call/time budget exhausted")
        limit = min(remaining, self.header["per_call_seconds"])
        key = self._key(label, request)
        index = len(starts) + 1
        self._append(dict(kind="started", attempt=index, key=key, label=label,
                          request=request, time_limit_s=limit))
        result = []

        class Attempt:
            def complete(self, record):
                if result:
                    raise RuntimeError("Attempt already completed")
                result.append(json.loads(_json(record)))

        def expired(*args):
            raise BudgetExceeded("Numerical attempt deadline exceeded")

        previous = signal.signal(signal.SIGALRM, expired)
        start = time.monotonic()
        self.active = True
        try:
            signal.setitimer(signal.ITIMER_REAL, limit)
            yield Attempt()
            if time.monotonic() - start >= limit:
                raise BudgetExceeded("Numerical attempt deadline exceeded")
            if not result:
                raise RuntimeError("Attempt exited without a result record")
            signal.setitimer(signal.ITIMER_REAL, 0)
            self._append(dict(kind="completed", attempt=index, key=key,
                              elapsed_s=time.monotonic()-start, result=result[0]))
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self._append(dict(kind="failed", attempt=index, key=key,
                              elapsed_s=time.monotonic()-start,
                              error_type=type(error).__name__, error=str(error)))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            self.active = False
