from tests.conftest import make_csv_bytes, make_gzip_csv


class TestBeltsEndpoint:
    """POST /belts endpoint tests."""

    def test_standard_upload(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv', csv, 'text/csv')),
            ('files', ('raw_data_b.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 200
        data = resp.json()
        assert data['type'] == 'belts'
        assert data['printer'] == 'default'
        assert data['url'].endswith('_belts.png')

    def test_busybox_separate_fields(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('file_a', ('raw_data_a.csv', csv, 'text/csv')),
            ('file_b', ('raw_data_b.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 200
        assert resp.json()['type'] == 'belts'

    def test_one_file_returns_400(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 400
        assert 'requires 2 files' in resp.json()['detail']

    def test_no_files_returns_400(self, client, mock_cli):
        resp = client.post('/belts')
        assert resp.status_code == 400

    def test_custom_printer_and_timestamp(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv', csv, 'text/csv')),
            ('files', ('raw_data_b.csv', csv, 'text/csv')),
        ], data={'printer': 'k1v4', 'timestamp': '20260325_140000'})
        data = resp.json()
        assert data['printer'] == 'k1v4'
        assert '20260325_140000_belts.png' in data['url']
        assert '/results/k1v4/' in data['url']

    def test_gzip_upload(self, client, mock_cli):
        gz = make_gzip_csv()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv.gz', gz, 'application/gzip')),
            ('files', ('raw_data_b.csv.gz', gz, 'application/gzip')),
        ])
        assert resp.status_code == 200

    def test_cli_failure_returns_500(self, client, mock_cli_failure):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv', csv, 'text/csv')),
            ('files', ('raw_data_b.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 500
        assert 'Analysis failed' in resp.json()['detail']

    def test_three_files_returns_400(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv', csv, 'text/csv')),
            ('files', ('raw_data_b.csv', csv, 'text/csv')),
            ('files', ('raw_data_c.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 400

    def test_duplicate_output_filename_returns_400(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv', csv, 'text/csv')),
            ('files', ('raw_data_a.csv.gz', make_gzip_csv(), 'application/gzip')),
        ])
        assert resp.status_code == 400
