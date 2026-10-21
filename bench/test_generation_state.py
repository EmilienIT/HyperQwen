#!/usr/bin/env python3
"""CPU-only interruption/resume checks for self-distillation JSONL output."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drafter"))
from generation_state import load_completed


class GenerationTests(unittest.TestCase):
    def test_torn_tail_and_unterminated_complete_record(self):
        for tail, want in ((b'{"id":', {1}), (b'{"id":2,"text":"\xc3', {1}),
                           (b'{"id":2}', {1, 2}), (b'{"id":2}\n', {1, 2})):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "gen.jsonl"
                prefix = b'{"id":1}\n'
                path.write_bytes(prefix + tail)
                self.assertEqual(load_completed(path), want)
                self.assertTrue(path.read_bytes().startswith(prefix))
                with path.open("ab") as f:
                    f.write(b'{"id":3}\n')
                records = [json.loads(line) for line in path.read_bytes().splitlines()]
                self.assertEqual({r['id'] for r in records}, want | {3})
                before = path.read_bytes()
                self.assertEqual(load_completed(path), want | {3})
                self.assertEqual(path.read_bytes(), before)

    def test_internal_corruption_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen.jsonl"
            original = b'{"id":1}\nbroken\n{"id":3}\n'
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "invalid JSONL"):
                load_completed(path)
            self.assertEqual(path.read_bytes(), original)

    def test_missing_output_is_a_fresh_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen.jsonl"
            self.assertEqual(load_completed(path), set())
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
