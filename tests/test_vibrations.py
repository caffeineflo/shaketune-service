from tests.conftest import make_csv_bytes, make_gzip_csv


class TestVibrationsEndpoint:
    """POST /vibrations endpoint tests."""

    def test_standard_upload(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/vibrations', files=[
            ('files', ('vib_data_1.csv', csv, 'text/csv')),
            ('files', ('vib_data_2.csv', csv, 'text/csv')),
        ], data={'kinematics': 'corexy'})
        assert resp.status_code == 200
        data = resp.json()
        assert data['type'] == 'vibrations'
        assert data['printer'] == 'default'
        assert data['url'].endswith('_vibrations.png')

    def test_custom_kinematics_passed_to_cli(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/vibrations', files=[
            ('files', ('vib_data.csv', csv, 'text/csv')),
        ], data={'kinematics': 'cartesian'})
        assert resp.status_code == 200
        cmd = mock_cli[0][0]
        assert '--kinematics' in cmd
        assert 'cartesian' in cmd

    def test_custom_printer_and_timestamp(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/vibrations', files=[
            ('files', ('vib_data.csv', csv, 'text/csv')),
        ], data={'printer': 'k1v3', 'timestamp': '20260325_150000'})
        data = resp.json()
        assert data['printer'] == 'k1v3'
        assert '20260325_150000_vibrations.png' in data['url']

    def test_no_files_returns_422(self, client, mock_cli):
        # vibrations uses File(...) (required), FastAPI returns 422 for missing required fields
        resp = client.post('/vibrations')
        assert resp.status_code == 422

    def test_gzip_upload(self, client, mock_cli):
        gz = make_gzip_csv()
        resp = client.post('/vibrations', files=[
            ('files', ('vib_data.csv.gz', gz, 'application/gzip')),
        ])
        assert resp.status_code == 200

    def test_cli_failure_returns_500(self, client, mock_cli_failure):
        csv = make_csv_bytes()
        resp = client.post('/vibrations', files=[
            ('files', ('vib_data.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 500
        assert 'Analysis failed' in resp.json()['detail']
