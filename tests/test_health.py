class TestHealthAndApi:
    """GET /health and GET /api tests."""

    def test_health_check(self, client):
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'ok'
        assert data['service'] == 'shaketune-service'

    def test_api_docs(self, client):
        resp = client.get('/api')
        assert resp.status_code == 200
        data = resp.json()
        assert 'endpoints' in data
        assert 'POST /shaper' in data['endpoints']
        assert 'POST /belts' in data['endpoints']
        assert 'GET /health' in data['endpoints']
