import gzip
import importlib
import subprocess
from pathlib import Path

import pytest
from starlette.testclient import TestClient

TEST_API_TOKEN = 'test-token'
AUTH_HEADERS = {'X-ShakeTune-Token': TEST_API_TOKEN}


@pytest.fixture(scope='session')
def server_module(tmp_path_factory):
    """Import the service after installing a safe, isolated test configuration."""
    environment = pytest.MonkeyPatch()
    environment.setenv('RESULTS_DIR', str(tmp_path_factory.mktemp('shaketune_results')))
    environment.setenv('KLIPPER_DIR', '/tmp/fake-klipper')
    environment.setenv('APP_DIR', '/app')
    environment.setenv('SHAKETUNE_API_TOKEN', TEST_API_TOKEN)
    environment.setenv('SHAKETUNE_ANALYSIS_TIMEOUT_SECONDS', '300')
    environment.setenv('SHAKETUNE_ANALYSIS_CONCURRENCY', '1')
    environment.setenv('SHAKETUNE_MAX_UPLOAD_FILES', '16')
    environment.setenv('SHAKETUNE_MAX_UPLOAD_BYTES', str(32 * 1024 * 1024))
    environment.setenv('SHAKETUNE_MAX_DECOMPRESSED_BYTES', str(64 * 1024 * 1024))
    environment.setenv('SHAKETUNE_MAX_TOTAL_DECOMPRESSED_BYTES', str(128 * 1024 * 1024))
    environment.setenv('SHAKETUNE_MAX_REQUEST_BODY_BYTES', str(64 * 1024 * 1024))
    environment.delenv('SHAKETUNE_API_TOKEN_FILE', raising=False)
    module = importlib.import_module('service.server')
    yield module
    environment.undo()


@pytest.fixture()
def results_dir(tmp_path):
    """Per-test isolated results directory."""
    d = tmp_path / 'results'
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _patch_results_dir(monkeypatch, results_dir, server_module):
    """Redirect RESULTS_DIR to per-test tmp dir for all tests."""
    monkeypatch.setattr(server_module, 'RESULTS_DIR', str(results_dir))


@pytest.fixture()
def client(server_module):
    """FastAPI test client."""
    with TestClient(server_module.app, headers=AUTH_HEADERS) as test_client:
        yield test_client


@pytest.fixture()
def unauthenticated_client(server_module):
    """FastAPI test client without the analysis API token."""
    with TestClient(server_module.app) as test_client:
        yield test_client


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
def mock_cli(monkeypatch, server_module):
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

    monkeypatch.setattr(server_module.subprocess, 'run', fake_run)
    return calls


@pytest.fixture()
def mock_cli_failure(monkeypatch, server_module):
    """Mock subprocess.run to simulate CLI failure."""

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout='', stderr='Analysis error: bad data')

    monkeypatch.setattr(server_module.subprocess, 'run', fake_run)


def create_result_file(results_dir, printer, filename):
    """Helper to pre-create a result file for GET endpoint tests."""
    printer_dir = Path(results_dir) / printer
    printer_dir.mkdir(parents=True, exist_ok=True)
    path = printer_dir / filename
    path.write_bytes(b'fake-png-data')
    return path
