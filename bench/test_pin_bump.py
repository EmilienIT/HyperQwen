#!/usr/bin/env python3
"""CPU-only checks that rejected pin bumps leave every destination file untouched."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
FILES = ("docker/requirements.txt", "verify.sh", "patches/apply.sh", "Dockerfile", "kvarn/install.sh",
         "README.md", "docs/docker.md", "docs/install.md")


class PinTests(unittest.TestCase):
    def run_bump(self, broken=False, bad_copin=False, dry_run=False):
        with tempfile.TemporaryDirectory() as tmp:
            root, fork = Path(tmp) / "project", Path(tmp) / "vllm"
            for name in FILES:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / name, target)
            if broken:
                path = root / "docs/install.md"
                text = path.read_text(encoding="utf-8").replace("pip install vllm==0.31.0", "pip install vllm==UNKNOWN")
                path.write_text(text, encoding="utf-8")
            (fork / "requirements").mkdir(parents=True)
            def git(folder, *args):
                return subprocess.run(["git", "-C", str(folder), *args], check=True, capture_output=True)
            git(root, "init", "-q")
            git(root, "add", ".")
            git(fork, "init", "-q")
            git(fork, "config", "user.name", "test")
            git(fork, "config", "user.email", "test@example.invalid")
            old_cubin = re.search(r"flashinfer-cubin==([^\s]+)", (root / "docs/install.md").read_text(encoding="utf-8")).group(1)
            (fork / "requirements/cuda.txt").write_text(f"flashinfer-cubin=={old_cubin}\n", encoding="utf-8")
            (fork / "requirements/common.txt").write_text("", encoding="utf-8")
            git(fork, "add", ".")
            git(fork, "commit", "-qm", "old")
            git(fork, "tag", "v0.31.0")
            (fork / "requirements/cuda.txt").write_text("flashinfer-cubin==99.0.0\n", encoding="utf-8")
            (fork / "requirements/common.txt").write_text("huggingface_hub>=99.0.0\n" if bad_copin else "", encoding="utf-8")
            git(fork, "add", ".")
            git(fork, "commit", "-qm", "new")
            git(fork, "tag", "v0.32.0")
            before = {name: (root / name).read_bytes() for name in FILES}
            args = [sys.executable, str(REPO / "scripts/pin-bump.py"), "--from", "0.31.0", "--to", "0.32.0",
                    "--fork", str(fork), "--repo", str(root)] + (["--dry-run"] if dry_run else [])
            result = subprocess.run(args, capture_output=True, text=True)
            after = {name: (root / name).read_bytes() for name in FILES}
            return result, before, after

    def test_missing_pattern_does_not_leave_partial_edits(self):
        result, before, after = self.run_bump(broken=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(before, after)

    def test_copin_failure_does_not_leave_partial_edits(self):
        result, before, after = self.run_bump(bad_copin=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(before, after)

    def test_valid_bump_writes_and_dry_run_does_not(self):
        for dry in (False, True):
            with self.subTest(dry=dry):
                result, before, after = self.run_bump(dry_run=dry)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(before == after, dry)
                if not dry:
                    self.assertIn(b"vllm==0.32.0", after["docker/requirements.txt"])


if __name__ == "__main__":
    unittest.main()
