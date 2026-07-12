import asyncio

import pytest
from starlette.testclient import TestClient


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
        assert 'X-ShakeTune-Token' in data['authentication']
        assert 'endpoints' in data
        assert 'POST /shaper' in data['endpoints']
        assert 'POST /belts' in data['endpoints']
        assert 'GET /health' in data['endpoints']

    def test_health_does_not_require_authentication(self, unauthenticated_client):
        resp = unauthenticated_client.get('/health')
        assert resp.status_code == 200

    @pytest.mark.parametrize('path', ['/shaper', '/belts', '/vibrations'])
    def test_analysis_endpoints_require_authentication(self, unauthenticated_client, path):
        resp = unauthenticated_client.post(path)
        assert resp.status_code == 401

    def test_analysis_endpoint_rejects_invalid_token(self, client):
        resp = client.post('/shaper', headers={'X-ShakeTune-Token': 'wrong-token'})
        assert resp.status_code == 401


class TestApiTokenConfiguration:
    """Service startup token configuration tests."""

    def test_loads_token_from_file(self, monkeypatch, server_module, tmp_path):
        token_file = tmp_path / 'token'
        token_file.write_text('file-token\n', encoding='utf-8')
        monkeypatch.delenv('SHAKETUNE_API_TOKEN', raising=False)
        monkeypatch.setenv('SHAKETUNE_API_TOKEN_FILE', str(token_file))
        assert server_module._load_api_token() == 'file-token'

    def test_fails_closed_without_token_configuration(self, monkeypatch, server_module):
        monkeypatch.delenv('SHAKETUNE_API_TOKEN', raising=False)
        monkeypatch.delenv('SHAKETUNE_API_TOKEN_FILE', raising=False)
        with pytest.raises(RuntimeError, match='Set SHAKETUNE_API_TOKEN'):
            server_module._load_api_token()

    def test_fails_closed_when_token_file_is_unreadable(self, monkeypatch, server_module, tmp_path):
        monkeypatch.delenv('SHAKETUNE_API_TOKEN', raising=False)
        monkeypatch.setenv('SHAKETUNE_API_TOKEN_FILE', str(tmp_path / 'missing-token'))
        with pytest.raises(RuntimeError, match='Unable to read'):
            server_module._load_api_token()


class TestRequestBodyLimit:
    """Ingress request limits are enforced before endpoint body parsing."""

    @staticmethod
    def _scope(headers=None):
        return {
            'type': 'http',
            'asgi': {'version': '3.0'},
            'http_version': '1.1',
            'method': 'POST',
            'scheme': 'http',
            'path': '/shaper',
            'raw_path': b'/shaper',
            'query_string': b'',
            'headers': headers or [],
            'client': ('127.0.0.1', 1234),
            'server': ('testserver', 80),
        }

    def test_rejects_declared_oversized_body_before_downstream(self, server_module):
        downstream_called = False
        sent_messages = []

        async def downstream(scope, receive, send):
            nonlocal downstream_called
            downstream_called = True

        async def receive():
            return {'type': 'http.request', 'body': b'', 'more_body': False}

        async def send(message):
            sent_messages.append(message)

        middleware = server_module.RequestBodyLimitMiddleware(downstream, max_body_bytes=10)
        asyncio.run(
            middleware(
                self._scope([(b'content-length', b'11')]),
                receive,
                send,
            ),
        )

        status = next(message['status'] for message in sent_messages if message['type'] == 'http.response.start')
        assert (status, downstream_called) == (413, False)

    def test_rejects_chunked_body_as_limit_is_crossed(self, server_module):
        messages = iter(
            [
                {'type': 'http.request', 'body': b'123456', 'more_body': True},
                {'type': 'http.request', 'body': b'789012', 'more_body': False},
            ],
        )
        downstream_completed = False
        sent_messages = []

        async def downstream(scope, receive, send):
            nonlocal downstream_completed
            try:
                while True:
                    message = await receive()
                    if not message.get('more_body', False):
                        break
                downstream_completed = True
            except server_module.RequestBodyTooLarge:
                await send({'type': 'http.response.start', 'status': 400, 'headers': []})
                await send({'type': 'http.response.body', 'body': b'parser error'})

        async def receive():
            return next(messages)

        async def send(message):
            sent_messages.append(message)

        middleware = server_module.RequestBodyLimitMiddleware(downstream, max_body_bytes=10)
        asyncio.run(
            middleware(
                self._scope([(b'transfer-encoding', b'chunked')]),
                receive,
                send,
            ),
        )

        status = next(message['status'] for message in sent_messages if message['type'] == 'http.response.start')
        assert (status, downstream_completed) == (413, False)

    def test_allows_body_at_limit(self, server_module):
        messages = iter(
            [
                {'type': 'http.request', 'body': b'12345', 'more_body': True},
                {'type': 'http.request', 'body': b'67890', 'more_body': False},
            ],
        )
        downstream_completed = False

        async def downstream(scope, receive, send):
            nonlocal downstream_completed
            while True:
                message = await receive()
                if not message.get('more_body', False):
                    break
            downstream_completed = True

        async def receive():
            return next(messages)

        async def send(message):
            return None

        middleware = server_module.RequestBodyLimitMiddleware(downstream, max_body_bytes=10)
        asyncio.run(middleware(self._scope(), receive, send))

        assert downstream_completed

    def test_chunked_multipart_request_returns_413(self, server_module):
        body = (
            b'--boundary\r\n'
            b'Content-Disposition: form-data; name="files"; filename="raw_data_x_x.csv"\r\n'
            b'Content-Type: text/csv\r\n\r\n'
            b'12345678901234567890\r\n'
            b'--boundary--\r\n'
        )
        chunks = (body[index : index + 7] for index in range(0, len(body), 7))
        limited_app = server_module.RequestBodyLimitMiddleware(server_module.app, max_body_bytes=32)

        with TestClient(limited_app) as limited_client:
            response = limited_client.post(
                '/shaper',
                content=chunks,
                headers={
                    'Content-Type': 'multipart/form-data; boundary=boundary',
                    'Transfer-Encoding': 'chunked',
                    'X-ShakeTune-Token': 'test-token',
                },
            )

        assert response.status_code == 413
