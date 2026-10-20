#!/usr/bin/env python3
"""CPU-only checks for the long corpus on an active, non-hardcoded Python install."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parent / "make_long_corpus.py"


class LongCorpusTests(unittest.TestCase):
    def test_active_package_and_frozen_utf8_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bench").mkdir()
            head = "frozen head\r\n\u00e6\u00f8\u00e5\n".encode("utf-8")
            (root / "bench/labd_corpus.txt").write_bytes(head)
            package = root / "active-libs/vllm"
            (package / "v1").mkdir(parents=True)
            (package / "__init__.py").write_text('raise RuntimeError("do not import the runtime")', encoding="utf-8")
            source = "# actual active vllm source\n" * 100
            (package / "v1/source.py").write_text(source, encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True,
                                    env=dict(os.environ, HOME=tmp, USERPROFILE=tmp, PYTHONPATH=str(package.parent)))
            self.assertEqual(result.returncode, 0, result.stderr)
            data = (root / "bench/labd_corpus_long.txt").read_bytes()
            self.assertTrue(data.startswith(head))
            self.assertIn(source.encode(), data)


if __name__ == "__main__":
    unittest.main()
