import gzip
import os
import subprocess
import tempfile
from pathlib import Path

# Set env vars BEFORE importing server module (it runs os.makedirs at import time)
_test_results_dir = tempfile.mkdtemp(prefix='shaketune_test_')
os.environ['RESULTS_DIR'] = _test_results_dir
os.environ['KLIPPER_DIR'] = '/tmp/fake-klipper'
os.environ['SHAKETUNE_DIR'] = '/tmp/fake-shaketune'

import pytest
from starlette.testclient import TestClient

from service import server
from service.server import app


@pytest.fixture()
def results_dir(tmp_path):
    """Per-test isolated results directory."""
    d = tmp_path / 'results'
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _patch_results_dir(monkeypatch, results_dir):
    """Redirect RESULTS_DIR to per-test tmp dir for all tests."""
    monkeypatch.setattr(server, 'RESULTS_DIR', str(results_dir))


@pytest.fixture()
def client():
    """FastAPI test client."""
    return TestClient(app)


def make_csv_bytes(num_rows=100):
    """Generate synthetic Klipper accelerometer CSV data."""
    lines = ['#time,accel_x,accel_y,accel_z']
    for i in range(num_rows):
        t = i * 0.001
        lines.append(f'{t:.6f},45.2,-12.8,9810.5')
    return '\n'.join(lines).encode()


def make_gzip_csv(num_rows=100):
    """Generate gzipped synthetic CSV data."""
    return gzip.compress(make_csv_bytes(num_rows))


@pytest.fixture()
def mock_cli(monkeypatch):
    """Mock subprocess.run to simulate successful CLI graph generation.

    Creates a fake PNG file at the -o output path so shutil.move succeeds.
    """
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        # Find the -o argument to create the expected output file
        for i, arg in enumerate(cmd):
            if arg == '-o' and i + 1 < len(cmd):
                output_path = Path(cmd[i + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b'fake-png-data')
                break
        return subprocess.CompletedProcess(cmd, 0, stdout='...done!', stderr='')

    monkeypatch.setattr('subprocess.run', fake_run)
    return calls


@pytest.fixture()
def mock_cli_failure(monkeypatch):
    """Mock subprocess.run to simulate CLI failure."""

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout='', stderr='Analysis error: bad data')

    monkeypatch.setattr('subprocess.run', fake_run)


def create_result_file(results_dir, printer, filename):
    """Helper to pre-create a result file for GET endpoint tests."""
    printer_dir = Path(results_dir) / printer
    printer_dir.mkdir(parents=True, exist_ok=True)
    path = printer_dir / filename
    path.write_bytes(b'fake-png-data')
    return path
