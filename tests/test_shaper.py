import subprocess

import pytest

from tests.conftest import make_csv_bytes, make_gzip_csv


class TestShaperEndpoint:
    """POST /shaper endpoint tests."""

    def test_standard_upload_two_axes(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/shaper', files=[
            ('files', ('raw_data_x_x.csv', csv, 'text/csv')),
            ('files', ('raw_data_y_y.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 200
        data = resp.json()
        assert data['type'] == 'shaper'
        assert data['printer'] == 'default'
        assert len(data['urls']) == 2
        axes = {r['axis'] for r in data['urls']}
        assert axes == {'x', 'y'}
        for r in data['urls']:
            assert f'_shaper_{r["axis"]}.png' in r['url']

    def test_busybox_separate_fields(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/shaper', files=[
            ('file_x', ('raw_data_x_x.csv', csv, 'text/csv')),
            ('file_y', ('raw_data_y_y.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 200
        data = resp.json()
        assert len(data['urls']) == 2

    def test_single_file(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/shaper', files=[
            ('files', ('raw_data_x_x.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 200
        data = resp.json()
        assert len(data['urls']) == 1
        assert data['urls'][0]['axis'] == 'x'

    def test_no_files_returns_400(self, client, mock_cli):
        resp = client.post('/shaper')
        assert resp.status_code == 400
        assert 'requires at least 1 file' in resp.json()['detail']

    def test_custom_printer(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/shaper', files=[
            ('files', ('raw_data_x_x.csv', csv, 'text/csv')),
        ], data={'printer': 'k1v3'})
        data = resp.json()
        assert data['printer'] == 'k1v3'
        assert '/results/k1v3/' in data['urls'][0]['url']

    def test_custom_timestamp(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/shaper', files=[
            ('files', ('raw_data_x_x.csv', csv, 'text/csv')),
        ], data={'timestamp': '20260325_120000'})
        assert '20260325_120000_shaper_x.png' in resp.json()['urls'][0]['url']

    def test_gzip_upload(self, client, mock_cli):
        gz = make_gzip_csv()
        resp = client.post('/shaper', files=[
            ('files', ('raw_data_x_x.csv.gz', gz, 'application/gzip')),
            ('files', ('raw_data_y_y.csv.gz', gz, 'application/gzip')),
        ])
        assert resp.status_code == 200
        assert len(resp.json()['urls']) == 2

    def test_extra_params_passed_to_cli(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/shaper', files=[
            ('files', ('raw_data_x_x.csv', csv, 'text/csv')),
        ], data={'max_freq': '150.0', 'scv': '3.0'})
        assert resp.status_code == 200
        cmd = mock_cli[0][0]
        assert '--max_freq' in cmd
        assert '150.0' in cmd
        assert '--scv' in cmd
        assert '3.0' in cmd

    def test_cli_called_per_axis(self, client, mock_cli):
        """Each CSV file gets its own CLI invocation."""
        csv = make_csv_bytes()
        resp = client.post('/shaper', files=[
            ('files', ('raw_data_x_x.csv', csv, 'text/csv')),
            ('files', ('raw_data_y_y.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 200
        assert len(mock_cli) == 2
        # Each call should have exactly one CSV path
        for call_cmd, _ in mock_cli:
            csv_args = [a for a in call_cmd if a.endswith('.csv')]
            assert len(csv_args) == 1

        assert mock_cli[0][1]['timeout'] == 300
        assert 'SHAKETUNE_API_TOKEN' not in mock_cli[0][1]['env']

    def test_cli_failure_returns_500(self, client, mock_cli_failure):
        csv = make_csv_bytes()
        resp = client.post('/shaper', files=[
            ('files', ('raw_data_x_x.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 500
        assert 'Analysis failed' in resp.json()['detail']

    @pytest.mark.parametrize('printer', ['!', '../outside', 'printer name', 'a' * 65])
    def test_rejects_invalid_printer_names(self, client, mock_cli, printer):
        resp = client.post(
            '/shaper',
            files=[('files', ('raw_data_x_x.csv', make_csv_bytes(), 'text/csv'))],
            data={'printer': printer},
        )
        assert resp.status_code == 400

    @pytest.mark.parametrize('timestamp', ['2026-03-25', '20260230_120000', '../20260325_120000'])
    def test_rejects_invalid_timestamps(self, client, mock_cli, timestamp):
        resp = client.post(
            '/shaper',
            files=[('files', ('raw_data_x_x.csv', make_csv_bytes(), 'text/csv'))],
            data={'timestamp': timestamp},
        )
        assert resp.status_code == 400

    def test_rejects_unknown_axis(self, client, mock_cli):
        resp = client.post(
            '/shaper',
            files=[('files', ('raw_data_z_z.csv', make_csv_bytes(), 'text/csv'))],
        )
        assert resp.status_code == 400

    def test_rejects_duplicate_axis(self, client, mock_cli):
        resp = client.post('/shaper', files=[
            ('files', ('first_x.csv', make_csv_bytes(), 'text/csv')),
            ('files', ('second_x.csv', make_csv_bytes(), 'text/csv')),
        ])
        assert resp.status_code == 400

    @pytest.mark.parametrize(
        'filename',
        ['../raw_data_x_x.csv', 'dir\\raw_data_x_x.csv', 'raw_data_x_x.txt', 'raw_data_x_x.CSV'],
    )
    def test_rejects_unsafe_filenames(self, client, mock_cli, filename):
        resp = client.post(
            '/shaper',
            files=[('files', (filename, make_csv_bytes(), 'text/csv'))],
        )
        assert resp.status_code == 400

    def test_rejects_too_many_files(self, client, mock_cli, monkeypatch, server_module):
        monkeypatch.setattr(server_module, 'MAX_UPLOAD_FILES', 1)
        resp = client.post('/shaper', files=[
            ('files', ('raw_data_x_x.csv', make_csv_bytes(), 'text/csv')),
            ('files', ('raw_data_y_y.csv', make_csv_bytes(), 'text/csv')),
        ])
        assert resp.status_code == 413

    def test_rejects_upload_over_transfer_limit(self, client, mock_cli, monkeypatch, server_module):
        monkeypatch.setattr(server_module, 'MAX_UPLOAD_BYTES_PER_FILE', 32)
        resp = client.post(
            '/shaper',
            files=[('files', ('raw_data_x_x.csv', make_csv_bytes(), 'text/csv'))],
        )
        assert resp.status_code == 413

    def test_rejects_gzip_bomb(self, client, mock_cli, monkeypatch, server_module):
        monkeypatch.setattr(server_module, 'MAX_DECOMPRESSED_BYTES_PER_FILE', 64)
        resp = client.post(
            '/shaper',
            files=[('files', ('raw_data_x_x.csv.gz', make_gzip_csv(), 'application/gzip'))],
        )
        assert resp.status_code == 413

    def test_rejects_invalid_gzip_data(self, client, mock_cli):
        resp = client.post(
            '/shaper',
            files=[('files', ('raw_data_x_x.csv.gz', b'not-gzip', 'application/gzip'))],
        )
        assert resp.status_code == 400

    def test_rejects_combined_decompressed_limit(self, client, mock_cli, monkeypatch, server_module):
        csv = make_csv_bytes(num_rows=10)
        monkeypatch.setattr(server_module, 'MAX_TOTAL_DECOMPRESSED_BYTES', len(csv) + 1)
        resp = client.post('/shaper', files=[
            ('files', ('raw_data_x_x.csv', csv, 'text/csv')),
            ('files', ('raw_data_y_y.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 413

    def test_cli_timeout_returns_504(self, client, monkeypatch, server_module):
        def time_out(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs['timeout'])

        monkeypatch.setattr(server_module.subprocess, 'run', time_out)
        resp = client.post(
            '/shaper',
            files=[('files', ('raw_data_x_x.csv', make_csv_bytes(), 'text/csv'))],
        )
        assert resp.status_code == 504

    def test_success_without_graph_output_returns_500(self, client, monkeypatch, server_module):
        def succeed_without_output(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout='done', stderr='')

        monkeypatch.setattr(server_module.subprocess, 'run', succeed_without_output)
        resp = client.post(
            '/shaper',
            files=[('files', ('raw_data_x_x.csv', make_csv_bytes(), 'text/csv'))],
        )
        assert resp.status_code == 500

    def test_cli_error_detail_is_bounded(self, client, monkeypatch, server_module):
        def fail_with_large_error(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout='', stderr='x' * 5_000)

        monkeypatch.setattr(server_module.subprocess, 'run', fail_with_large_error)
        resp = client.post(
            '/shaper',
            files=[('files', ('raw_data_x_x.csv', make_csv_bytes(), 'text/csv'))],
        )
        assert len(resp.json()['detail']) <= 520
