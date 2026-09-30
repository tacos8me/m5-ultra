"""CPU-only tests: the ds41-og supervisor forwards only its allow-list to the child (fake children, no MLX/GPU/box).

The child binds loopback, so omlx skips API-key checks for everything it serves; the supervisor is what LAN
clients reach through llama-swap (/v1/... and /upstream/ds41/...). Allowed: POST /v1/chat/completions,
/v1/completions, /v1/messages, /v1/responses; GET /v1/models, /og/stats; DS41_OG_EXTRA_PATHS pairs.
Everything else: 404 + an OpenAI-shaped error body, and the child never sees the request.

Run: SUP_TEST_PORT=12630 python og_serve/test_proxy_allowlist.py   (uses SUP_TEST_PORT .. +5)
"""
import http.client
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('SUP_TEST_PORT', '12630')
from test_supervisor_failover import EXPECTED, BoxPort, Sup, check, stream  # noqa: E402

BASE = int(os.environ['SUP_TEST_PORT'])

REFUSED = [
    ('GET', '/'),
    ('GET', '/admin'),
    ('GET', '/admin/api/settings'),
    ('POST', '/admin/api/settings'),
    ('POST', '/admin/api/models/ds41-og/unload'),
    ('POST', '/v1/mcp/execute'),
    ('GET', '/v1/mcp/tools'),
    ('POST', '/v1/websearch/fetch'),
    ('POST', '/v1/models/ds41-og/unload'),
    ('POST', '/v1/models/ds41-og/load'),
    ('GET', '/v1/models/status'),
    ('GET', '/v1/models/ds41'),
    ('GET', '/api/status'),
    ('POST', '/v1/embeddings'),
    ('POST', '/v1/rerank'),
    ('POST', '/v1/messages/count_tokens'),
    ('GET', '/v1/responses/resp_fake'),
    ('DELETE', '/v1/responses/resp_fake'),
    ('GET', '/v1/chat/completions'),      # right path, wrong method
    ('PUT', '/og/stats'),
    ('POST', '/og/fe'),                   # trace-mode runtime switches of the og worker
    ('POST', '/v1/chat/completions/'),    # trailing slash is another path
    ('POST', '/v1/chat/completions/../../admin/api/settings'),  # raw dot segments, not normalized away
    ('POST', '/v1%2Fmcp/execute'),        # percent-encoded separator decodes to a refused path
    ('GET', '/%61dmin/api/settings'),
    ('OPTIONS', '/v1/chat/completions'),
    ('HEAD', '/v1/models'),
]


def call(port, method, path, body=None):
    """One raw request (http.client keeps the path exactly as given); returns (status, parsed body or bytes)."""
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers={'Content-Type': 'application/json'} if data else {})
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw


def child_hits(port):
    return call(port, 'GET', '/fake/stats')[1]['hits']


def main():
    box = BoxPort(BASE)
    results = []
    sup = Sup(BASE + 1, BASE, DS41_OG_RESTART_WAIT_S='1')  # child on BASE + 2
    child = BASE + 2
    try:
        # the fake child answers every refused path itself (so a forwarded request would be visible)
        st, body = call(child, 'POST', '/v1/mcp/execute', {})
        results.append(check('fake child exposes the omlx-only surface', st == 200 and body.get('fake') == 'catch-all', body))
        before = dict(child_hits(child))

        # allowed
        r = stream(sup.base, {})
        results.append(check('POST /v1/chat/completions (stream) forwarded', r['content'] == EXPECTED and r['done']
                             and not r['errors'], r))
        st, d = call(sup.port, 'POST', '/v1/chat/completions', dict(model='ds41', messages=[]))
        results.append(check('POST /v1/chat/completions forwarded', st == 200
                             and d['choices'][0]['message']['content'] == EXPECTED, (st, d)))
        r = stream(sup.base, {'prompt': 'x'}, path='/v1/completions')
        results.append(check('POST /v1/completions forwarded', r['text'] == EXPECTED and not r['errors'], r))
        st, d = call(sup.port, 'POST', '/v1/messages', dict(model='ds41', max_tokens=8, messages=[]))
        results.append(check('POST /v1/messages forwarded', st == 200 and d['content'][0]['text'] == 'saw it', (st, d)))
        st, d = call(sup.port, 'POST', '/v1/responses', dict(model='ds41', input='hi'))
        results.append(check('POST /v1/responses forwarded', st == 200 and d.get('object') == 'response', (st, d)))
        st, d = call(sup.port, 'GET', '/v1/models')
        results.append(check('GET /v1/models forwarded', st == 200 and d['data'][0]['id'] == 'ds41-og', (st, d)))
        st, d = call(sup.port, 'GET', '/og/stats')
        results.append(check('GET /og/stats forwarded', st == 200 and 'steps' in d, (st, d)))
        st, d = call(sup.port, 'GET', '/v1/models?x=1')
        st2, _ = call(sup.port, 'POST', '/v1/mcp/execute?x=/v1/models', {})
        results.append(check('query strings do not change the decision', st == 200 and st2 == 404, (st, st2, d)))
        st, d = call(sup.port, 'GET', '/health')
        results.append(check('supervisor /health still served', st == 200 and d.get('status') == 'ok', (st, d)))

        # refused
        bad, handled = [], 1  # the query-string probe above was refused too
        for method, path in REFUSED:
            st, d = call(sup.port, method, path, {} if method in ('POST', 'PUT', 'DELETE') else None)
            if method == 'HEAD':  # no body; Starlette may answer 405 before the handler: never forwarded either way
                handled += st == 404
                if st not in (404, 405):
                    bad.append((method, path, st, d))
                continue
            handled += 1
            shaped = (isinstance(d, dict) and d.get('error', {}).get('code') == 'not_found'
                      and d['error'].get('type') == 'invalid_request_error'
                      and 'not served here' in d['error'].get('message', ''))
            if st != 404 or not shaped:
                bad.append((method, path, st, d))
        results.append(check(f'{len(REFUSED)} other method/path pairs -> 404 with an OpenAI error body', not bad, bad))
        after = child_hits(child)
        leaked = {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}
        results.append(check('no refused request reached the child', not leaked, leaked))
        h = sup.health()
        results.append(check('/health counts refused requests', h.get('refused') == handled, (handled, h)))
    finally:
        sup.stop()

    # DS41_OG_EXTRA_PATHS: exact extra pairs without a code change
    sup = Sup(BASE + 3, BASE, DS41_OG_RESTART_WAIT_S='1',
              DS41_OG_EXTRA_PATHS='post /v1/messages/count_tokens, GET /v1/models/status')
    child = BASE + 4
    try:
        st, d = call(sup.port, 'POST', '/v1/messages/count_tokens', dict(model='ds41', messages=[]))
        st2, _ = call(sup.port, 'GET', '/v1/models/status')
        st3, _ = call(sup.port, 'GET', '/v1/messages/count_tokens')
        hits = child_hits(child)
        results.append(check('DS41_OG_EXTRA_PATHS pairs forwarded (method case-insensitive), others still 404',
                             st == 200 and st2 == 200 and st3 == 404
                             and hits.get('POST /v1/messages/count_tokens') == 1
                             and hits.get('GET /v1/models/status') == 1
                             and 'GET /v1/messages/count_tokens' not in hits, (st, st2, st3, hits)))
        r = stream(sup.base, {})
        results.append(check('inference unchanged with extras', r['content'] == EXPECTED and not r['errors'], r))
    finally:
        sup.stop()
    box.down()
    ok = all(results)
    print('ALL PASS' if ok else 'SOME FAILED', f'({sum(results)}/{len(results)})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
