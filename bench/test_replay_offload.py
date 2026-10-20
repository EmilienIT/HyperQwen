#!/usr/bin/env python3
"""CPU-only negative controls for the offload replay oracle."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock

import harness

SCRIPT = Path(__file__).resolve().parent / "replay_offload_serve.py"


class Output(io.StringIO):
    def reconfigure(self, **kwargs):
        pass


class ReplayTests(unittest.TestCase):
    def run_replay(self, growth, missing=False, legacy=False):
        requests = 0

        def stream(*args, **kwargs):
            nonlocal requests
            requests += 1
            yield {"choices": [{"delta": {"content": "ok"}}]}
            yield {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}

        def samples():
            values = [("vllm:kv_offload_store_bytes_total", "", 0)]
            if not missing:
                name, labels = ("vllm:kv_offload_total_bytes_total", '{transfer_type="CPU_to_GPU"}') if legacy else ("vllm:kv_offload_load_bytes_total", "")
                values.append((name, labels, 500 + (growth if requests == 3 else 0)))
            return values

        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.txt"
            corpus.write_text("real source text", encoding="utf-8")
            out = Output()
            with mock.patch.object(sys, "argv", [str(SCRIPT), "test", "18020", "10", "1"]), \
                    mock.patch.dict(os.environ, {"DEPTH_CORPUS_GLOB": str(corpus), "TIER_GIB": "0"}), \
                    mock.patch.object(harness, "stream", side_effect=stream), \
                    mock.patch.object(harness, "samples", side_effect=samples), \
                    mock.patch("time.perf_counter", side_effect=[1, 11, 12, 20, 21, 22, 30, 31, 32]), \
                    contextlib.redirect_stdout(out):
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(SCRIPT), run_name="__main__")
            return result.exception.code, json.loads(out.getvalue().splitlines()[-1])

    def test_fast_gpu_hit_is_not_evidence_of_cpu_offload(self):
        code, result = self.run_replay(0)
        self.assertEqual((code, result["verdict"]), (1, "NOT-SERVED"))

    def test_positive_load_delta_and_fast_replay_pass(self):
        code, result = self.run_replay(1000)
        self.assertEqual((code, result["verdict"]), (0, "SERVED"))

    def test_missing_load_metrics_are_an_invalid_run(self):
        code, result = self.run_replay(0, missing=True)
        self.assertEqual((code, result["verdict"]), (2, "INVALID-NO-LOAD-METRICS"))

    def test_counter_reset_is_an_invalid_run(self):
        code, result = self.run_replay(-100)
        self.assertEqual((code, result["verdict"]), (2, "INVALID-COUNTER-RESET"))

    def test_deprecated_direction_counter_still_works(self):
        code, result = self.run_replay(1000, legacy=True)
        self.assertEqual((code, result["loaded_bytes"]), (0, 1000))

    def test_missing_corpus_never_reaches_the_server(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(sys, "argv", [str(SCRIPT), "test", "18020", "10", "1"]), \
                mock.patch.dict(os.environ, {"DEPTH_CORPUS_GLOB": str(Path(tmp) / "missing.txt")}), \
                mock.patch.object(harness, "samples") as samples, \
                contextlib.redirect_stdout(Output()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(SCRIPT), run_name="__main__")
            self.assertEqual(result.exception.code, 2)
            samples.assert_not_called()


if __name__ == "__main__":
    unittest.main()
