"""Minimal capacity metadata for the existing GLM admission gateway.

Reads resolved vLLM settings without enabling its development endpoints.
Authenticated routes pass straight through, including streaming responses.
When VLLM_API_KEY is set, protect all HTTP routes except health/CORS preflight;
vLLM's built-in authentication covers only its versioned API paths.
"""
import hmac
import json
import os


class CapacityMiddleware:
    def __init__(self, app):
        self.app = app
        self.key = os.environ.get('VLLM_API_KEY', '').encode()

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        scheme, _, token = headers.get(b'authorization', b'').partition(b' ')
        status, data = 200, None
        public = (scope['path'] == '/health' and scope.get('method') == 'GET') or scope.get('method') == 'OPTIONS'
        if self.key and not public and (scheme.lower() != b'bearer' or not hmac.compare_digest(token, self.key)):
            status, data = 401, {'error': 'Unauthorized'}
        elif scope['path'] != '/v1/amos/capacity':
            return await self.app(scope, receive, send)
        elif scope.get('method') != 'GET':
            status, data = 405, {'error': 'Method not allowed'}
        else:
            config = getattr(scope['app'].state, 'vllm_config', None)
            if config is None:
                status, data = 503, {'error': 'Engine configuration unavailable'}
            else:
                data = {'context_length': config.model_config.max_model_len,
                        'max_running_requests': config.scheduler_config.max_num_seqs}
        body = json.dumps(data).encode()
        await send({'type': 'http.response.start', 'status': status, 'headers': [
            (b'content-type', b'application/json'), (b'content-length', str(len(body)).encode())]})
        await send({'type': 'http.response.body', 'body': body})
