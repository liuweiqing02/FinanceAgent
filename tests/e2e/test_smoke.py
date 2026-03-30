import subprocess
import sys


def test_e2e_smoke() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "app.main", "--ticker", "AAPL"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "研报已生成" in proc.stdout
