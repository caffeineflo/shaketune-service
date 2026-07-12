import os
import subprocess
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).parents[1] / 'macros' / 'scripts'


def write_executable(path, contents):
    path.write_text(contents)
    path.chmod(0o755)


@pytest.mark.parametrize('script_name', ['run_shaper.sh', 'upload_shaper.sh', 'upload_belts.sh'])
def test_official_upload_timeout_exceeds_service_analysis_timeout(script_name):
    script = (SCRIPTS_DIR / script_name).read_text()
    assert 'curl --timeout 360 ' in script


def restart_environment(tmp_path, succeeds):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()

    write_executable(
        bin_dir / 'curl',
        '#!/bin/sh\n'
        'if [ "${FAKE_RESTART_SUCCEEDS:-0}" = "1" ]; then\n'
        '  : > "$FAKE_RESTART_MARKER"\n'
        '  exit 0\n'
        'fi\n'
        'exit 7\n',
    )
    write_executable(
        bin_dir / 'wget',
        '#!/bin/sh\n'
        'case "$*" in\n'
        '  *printer/info*) printf \'{"state":"ready"}\' ;;\n'
        '  *toolhead*)\n'
        '    if [ -f "$FAKE_RESTART_MARKER" ]; then\n'
        '      printf \'{"homed_axes":""}\'\n'
        '    else\n'
        '      printf \'{"homed_axes":"xyz"}\'\n'
        '    fi\n'
        '    ;;\n'
        'esac\n',
    )
    write_executable(bin_dir / 'sleep', '#!/bin/sh\nexit 0\n')

    environment = os.environ.copy()
    environment.update(
        {
            'FAKE_RESTART_MARKER': str(tmp_path / 'restarted'),
            'FAKE_RESTART_SUCCEEDS': '1' if succeeds else '0',
            'PATH': f'{bin_dir}:{environment["PATH"]}',
            'SHAKETUNE_RESTART_POLL_SECONDS': '1',
            'SHAKETUNE_RESTART_TIMEOUT_SECONDS': '2',
        },
    )
    return environment


def upload_environment(tmp_path, graph_retrievable):
    bin_dir = tmp_path / 'upload-bin'
    bin_dir.mkdir()

    write_executable(
        bin_dir / 'curl',
        '#!/bin/sh\n'
        'printf \'{"url":"/results/k1test/20260712_150000_belts.png"}\'\n',
    )
    write_executable(bin_dir / 'date', '#!/bin/sh\nprintf \'20260712_150000\\n\'\n')
    write_executable(bin_dir / 'wget', '#!/bin/sh\nexit "${FAKE_GRAPH_EXIT_STATUS:-0}"\n')

    file_a = tmp_path / 'raw_data_a.csv'
    file_b = tmp_path / 'raw_data_b.csv'
    file_a.write_bytes(b'a' * 1_000_001)
    file_b.write_bytes(b'b' * 1_000_001)
    token_file = tmp_path / 'token'
    token_file.write_text('test-token\n')

    environment = os.environ.copy()
    environment.update(
        {
            'FAKE_GRAPH_EXIT_STATUS': '0' if graph_retrievable else '1',
            'PATH': f'{bin_dir}:{environment["PATH"]}',
            'SHAKETUNE_BELT_FILE_A': str(file_a),
            'SHAKETUNE_BELT_FILE_B': str(file_b),
            'SHAKETUNE_TOKEN_FILE': str(token_file),
        },
    )
    return environment, file_a, file_b


@pytest.mark.parametrize('script_name', ['run_shaper.sh', 'run_belts.sh'])
def test_calibration_script_refuses_shared_lock(tmp_path, script_name):
    lock_dir = tmp_path / 'shaketune_calibration.lock'
    lock_dir.mkdir()
    environment = os.environ.copy()
    environment['SHAKETUNE_LOCK_DIR'] = str(lock_dir)

    result = subprocess.run(
        ['sh', str(SCRIPTS_DIR / script_name)],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=5,
    )

    assert result.returncode == 1


def test_belt_runner_checks_token_before_contacting_moonraker(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    moonraker_marker = tmp_path / 'moonraker-called'
    write_executable(bin_dir / 'wget', '#!/bin/sh\n: > "$FAKE_MOONRAKER_MARKER"\nexit 1\n')

    environment = os.environ.copy()
    environment.update(
        {
            'FAKE_MOONRAKER_MARKER': str(moonraker_marker),
            'PATH': f'{bin_dir}:{environment["PATH"]}',
            'SHAKETUNE_LOCK_DIR': str(tmp_path / 'lock'),
            'SHAKETUNE_TOKEN_FILE': str(tmp_path / 'missing-token'),
        },
    )
    result = subprocess.run(
        ['sh', str(SCRIPTS_DIR / 'run_belts.sh')],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=5,
    )

    assert (result.returncode, moonraker_marker.exists()) == (1, False)


def test_belt_runner_bounds_failed_gcode_call_and_reports_context(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    wget_args = tmp_path / 'wget-args'
    write_executable(
        bin_dir / 'wget',
        '#!/bin/sh\n'
        'printf "%s\\n" "$*" >> "$FAKE_WGET_ARGS"\n'
        'case "$*" in\n'
        '  *print_stats*) printf \'{"state":"standby"}\' ;;\n'
        '  *extruder*|*heater_bed*) printf \'{"target": 0}\' ;;\n'
        '  *gcode/script*) exit 1 ;;\n'
        'esac\n',
    )
    write_executable(bin_dir / 'sleep', '#!/bin/sh\nexit 0\n')
    token_file = tmp_path / 'token'
    token_file.write_text('test-token\n')

    environment = os.environ.copy()
    environment.update(
        {
            'FAKE_WGET_ARGS': str(wget_args),
            'PATH': f'{bin_dir}:{environment["PATH"]}',
            'SHAKETUNE_LOCK_DIR': str(tmp_path / 'lock'),
            'SHAKETUNE_TOKEN_FILE': str(token_file),
        },
    )
    result = subprocess.run(
        ['sh', str(SCRIPTS_DIR / 'run_belts.sh')],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=5,
    )

    assert (result.returncode, '-T 360' in wget_args.read_text(), 'G28' in result.stdout) == (1, True, True)


def test_belt_runner_reports_csv_timeout_under_errexit(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    write_executable(
        bin_dir / 'wget',
        '#!/bin/sh\n'
        'case "$*" in\n'
        '  *print_stats*) printf \'{"state":"standby"}\' ;;\n'
        '  *extruder*|*heater_bed*) printf \'{"target": 0}\' ;;\n'
        '  *gcode/script*) exit 0 ;;\n'
        'esac\n',
    )
    write_executable(bin_dir / 'sleep', '#!/bin/sh\nexit 0\n')
    token_file = tmp_path / 'token'
    token_file.write_text('test-token\n')

    environment = os.environ.copy()
    environment.update(
        {
            'PATH': f'{bin_dir}:{environment["PATH"]}',
            'SHAKETUNE_BELT_FILE_A': str(tmp_path / 'missing-a.csv'),
            'SHAKETUNE_BELT_FILE_B': str(tmp_path / 'missing-b.csv'),
            'SHAKETUNE_LOCK_DIR': str(tmp_path / 'lock'),
            'SHAKETUNE_TOKEN_FILE': str(token_file),
        },
    )
    result = subprocess.run(
        ['sh', str(SCRIPTS_DIR / 'run_belts.sh')],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=5,
    )

    assert (result.returncode, 'ERROR: Belt A CSV not ready' in result.stdout) == (1, True)


def test_firmware_restart_accepts_ready_unhomed_transition(tmp_path):
    result = subprocess.run(
        ['sh', str(SCRIPTS_DIR / 'firmware_restart.sh')],
        capture_output=True,
        check=False,
        env=restart_environment(tmp_path, succeeds=True),
        text=True,
        timeout=5,
    )

    assert result.returncode == 0


def test_firmware_restart_rejects_failed_post_without_transition(tmp_path):
    result = subprocess.run(
        ['sh', str(SCRIPTS_DIR / 'firmware_restart.sh')],
        capture_output=True,
        check=False,
        env=restart_environment(tmp_path, succeeds=False),
        text=True,
        timeout=5,
    )

    assert result.returncode == 1


def test_belt_upload_removes_csvs_after_verified_graph(tmp_path):
    environment, file_a, file_b = upload_environment(tmp_path, graph_retrievable=True)

    subprocess.run(
        ['sh', str(SCRIPTS_DIR / 'upload_belts.sh'), 'service', '8080', 'k1test'],
        capture_output=True,
        check=True,
        env=environment,
        text=True,
        timeout=10,
    )

    assert not file_a.exists() and not file_b.exists()


def test_belt_upload_preserves_csvs_when_graph_is_not_retrievable(tmp_path):
    environment, file_a, file_b = upload_environment(tmp_path, graph_retrievable=False)

    result = subprocess.run(
        ['sh', str(SCRIPTS_DIR / 'upload_belts.sh'), 'service', '8080', 'k1test'],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=10,
    )

    assert (result.returncode, file_a.exists(), file_b.exists()) == (1, True, True)
