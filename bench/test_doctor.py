#!/usr/bin/env python3
"""CPU-only checks for doctor health failures and redacted diagnostics."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class DoctorTests(unittest.TestCase):
    def run_doctor(self, code, status, config_status=0):
        with tempfile.TemporaryDirectory(prefix="hq-doctor-") as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            shutil.copyfile(REPO / "scripts/hq-doctor.sh", root / "scripts/hq-doctor.sh")
            (root / "resolve_config.sh").write_text(
                f'resolve_effective_config() {{ return {config_status}; }}\n', encoding="utf-8")
            binary = root / "bin"
            binary.mkdir()
            for name, body in {"curl": f'printf %s {code}; exit {status}',
                               "docker": "exit 1", "ss": "exit 1", "nvidia-smi": "exit 1"}.items():
                path = binary / name
                path.write_bytes(("#!/bin/sh\n" + body + "\n").encode())
                path.chmod(0o755)
            # Git Bash needs its own POSIX PATH for a Windows temporary directory.
            script = 'export PATH="$PWD/bin:$PATH"; bash scripts/hq-doctor.sh'
            return subprocess.run(["bash", "-c", script], cwd=root, capture_output=True, text=True,
                                  env=dict(os.environ, VLLM_API_KEY="test-value"), timeout=15)

    def test_failed_curl_is_no_server(self):
        result = self.run_doctor("000", 7)
        self.assertEqual(result.returncode, 1)
        self.assertIn("doctor: no server on", result.stdout)
        self.assertNotIn("HTTP 000000", result.stdout)

    def test_http_failure_is_not_ready(self):
        result = self.run_doctor("503", 0)
        self.assertEqual(result.returncode, 1)
        self.assertIn("answers HTTP 503", result.stdout)

    def test_health_200_passes(self):
        result = self.run_doctor("200", 0)
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("test-value", result.stdout + result.stderr)

    def test_bad_config_is_reported_through_the_pipeline(self):
        result = self.run_doctor("200", 0, config_status=1)
        self.assertIn("unvalidated config", result.stdout)


if __name__ == "__main__":
    unittest.main()
