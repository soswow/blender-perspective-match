"""Budget caps must survive failure, restart and attempted cache reuse."""

import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest

from tools.synthetic_sync.budget import BudgetExceeded, ExperimentBudget


@unittest.skipUnless(os.name == "posix", "Experiment ledger requires POSIX locks/signals")
class ExperimentBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "attempts.jsonl"

    def budget(self, **overrides):
        options = dict(metadata={"source": "frozen"}, max_calls=2,
                       wall_seconds=10, per_call_seconds=1)
        options.update(overrides)
        return ExperimentBudget(self.path, **options)

    def test_reserved_before_execution_and_restart_retains_cap(self):
        with self.budget() as budget:
            with budget.attempt("one", {"focal": 700}) as attempt:
                self.assertEqual(json.loads(self.path.read_text().splitlines()[-1])["kind"], "started")
                attempt.complete({"success": True})
            with self.assertRaisesRegex(ValueError, "failure"):
                with budget.attempt("two", {}):
                    raise ValueError("failure")
        with self.budget() as budget:
            self.assertEqual(budget.cached("one", {"focal": 700}), {"success": True})
            self.assertIsNone(budget.cached("one", {"focal": 701}))
            self.assertIsNone(budget.cached("two", {}))
            with self.assertRaises(BudgetExceeded):
                with budget.attempt("third", {}):
                    self.fail("exhausted budget executed")

    def test_changed_source_or_limits_cannot_reuse_ledger(self):
        with self.budget():
            pass
        for changes in ({"metadata": {"source": "changed"}}, {"max_calls": 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                with self.budget(**changes):
                    pass

    def test_deadline_is_recorded_and_restores_handler(self):
        handler = signal.getsignal(signal.SIGALRM)
        with self.budget(per_call_seconds=0.02) as budget:
            with self.assertRaises(BudgetExceeded):
                with budget.attempt("slow", {}):
                    time.sleep(0.2)
        last = json.loads(self.path.read_text().splitlines()[-1])
        self.assertEqual(last["kind"], "failed")
        self.assertEqual(last["error_type"], "BudgetExceeded")
        self.assertEqual(signal.getsignal(signal.SIGALRM), handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_interrupted_reservation_consumes_time_on_resume(self):
        with self.budget(wall_seconds=1) as budget:
            budget._append(dict(kind="started", attempt=1, key="interrupted",
                                label="killed", request={}, time_limit_s=1))
        with self.budget(wall_seconds=1) as budget:
            with self.assertRaises(BudgetExceeded):
                with budget.attempt("after-kill", {}):
                    self.fail("interrupted reservation was free")

    def test_missing_result_is_a_failed_attempt(self):
        with self.budget() as budget:
            with self.assertRaisesRegex(RuntimeError, "without a result"):
                with budget.attempt("empty", {}):
                    pass
            self.assertIsNone(budget.cached("empty", {}))

    def test_second_writer_cannot_reserve_same_budget(self):
        with self.budget():
            with self.assertRaises(BlockingIOError):
                with self.budget():
                    pass
