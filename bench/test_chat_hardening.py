#!/usr/bin/env python3
"""CPU-only checks for explicit and default chat-template hardening targets."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prepare"))
import harden_chat_template as hardening


class HardeningTests(unittest.TestCase):
    def setUp(self):
        # Target selection is under test; the atomic publisher has a separate crash suite.
        writer = mock.patch.object(hardening, "write_text",
                                   side_effect=lambda p, text: Path(p).write_text(text, encoding="utf-8"))
        writer.start()
        self.addCleanup(writer.stop)

    def test_target_scope_and_deduplication(self):
        for mode in ("file", "directory", "both", "default"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                explicit = root / "external/chat_template.jinja"
                other = root / "models/other/chat_template.jinja"
                for path in (explicit, other):
                    path.parent.mkdir(parents=True)
                    path.write_text(hardening.OLD, encoding="utf-8")
                args = {"file": [str(explicit)], "directory": [str(explicit.parent)],
                        "both": [str(explicit.parent), str(explicit)], "default": []}[mode]
                out = io.StringIO()
                with mock.patch.object(hardening, "__file__", str(root / "prepare/script.py")), \
                        mock.patch.object(sys, "argv", ["script.py", *args]), contextlib.redirect_stdout(out):
                    self.assertEqual(hardening.main(), 0)
                self.assertEqual(out.getvalue().count("chat_template.jinja"), 1)
                self.assertEqual(hardening.MARKER in explicit.read_text(encoding="utf-8"), mode != "default")
                self.assertEqual(hardening.MARKER in other.read_text(encoding="utf-8"), mode == "default")

    def test_help_never_rewrites_an_earlier_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat_template.jinja"
            path.write_text(hardening.OLD, encoding="utf-8")
            with mock.patch.object(sys, "argv", ["script.py", str(path), "--help"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    hardening.main()
            self.assertEqual(result.exception.code, 0)
            self.assertEqual(path.read_text(encoding="utf-8"), hardening.OLD)


if __name__ == "__main__":
    unittest.main()
