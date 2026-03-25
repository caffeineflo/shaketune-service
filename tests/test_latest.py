from tests.conftest import create_result_file


class TestLatestEndpoints:
    """GET /latest/{type} and /latest/{printer}/{type} tests."""

    def test_no_results_returns_404(self, client):
        resp = client.get('/latest/shaper', follow_redirects=False)
        assert resp.status_code == 404

    def test_invalid_graph_type_returns_400(self, client):
        resp = client.get('/latest/invalid', follow_redirects=False)
        assert resp.status_code == 400

    def test_redirect_to_latest(self, client, results_dir):
        create_result_file(results_dir, 'default', '20260325_100000_shaper.png')
        create_result_file(results_dir, 'default', '20260325_120000_shaper.png')
        resp = client.get('/latest/shaper', follow_redirects=False)
        assert resp.status_code == 302
        assert '20260325_120000_shaper.png' in resp.headers['location']

    def test_printer_specific_latest(self, client, results_dir):
        create_result_file(results_dir, 'k1v3', '20260325_100000_belts.png')
        resp = client.get('/latest/k1v3/belts', follow_redirects=False)
        assert resp.status_code == 302
        assert '/results/k1v3/' in resp.headers['location']

    def test_returns_most_recent(self, client, results_dir):
        create_result_file(results_dir, 'default', '20260325_080000_shaper.png')
        create_result_file(results_dir, 'default', '20260325_160000_shaper.png')
        create_result_file(results_dir, 'default', '20260325_120000_shaper.png')
        resp = client.get('/latest/shaper', follow_redirects=False)
        assert resp.status_code == 302
        assert '20260325_160000_shaper.png' in resp.headers['location']

    def test_printer_invalid_type_returns_400(self, client, results_dir):
        create_result_file(results_dir, 'k1v3', '20260325_100000_shaper.png')
        resp = client.get('/latest/k1v3/invalid', follow_redirects=False)
        assert resp.status_code == 400

    def test_nonexistent_printer_returns_404(self, client):
        resp = client.get('/latest/nonexistent/shaper', follow_redirects=False)
        assert resp.status_code == 404
