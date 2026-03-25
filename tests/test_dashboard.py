from tests.conftest import create_result_file


class TestDashboard:
    """GET / and GET /printer/{name} tests."""

    def test_home_empty(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        assert 'No results yet.' in resp.text

    def test_home_with_printers(self, client, results_dir):
        create_result_file(results_dir, 'k1v3', '20260325_100000_shaper.png')
        create_result_file(results_dir, 'k1v4', '20260325_110000_belts.png')
        resp = client.get('/')
        assert resp.status_code == 200
        assert 'k1v3' in resp.text
        assert 'k1v4' in resp.text

    def test_printer_detail(self, client, results_dir):
        create_result_file(results_dir, 'k1v3', '20260325_100000_shaper.png')
        create_result_file(results_dir, 'k1v3', '20260325_110000_belts.png')
        resp = client.get('/printer/k1v3')
        assert resp.status_code == 200
        assert 'k1v3' in resp.text

    def test_printer_not_found(self, client):
        resp = client.get('/printer/nonexistent')
        assert resp.status_code == 404

    def test_printer_exists_but_empty(self, client, results_dir):
        # Create printer dir with no matching result files
        (results_dir / 'emptyprinter').mkdir()
        resp = client.get('/printer/emptyprinter')
        assert resp.status_code == 200
