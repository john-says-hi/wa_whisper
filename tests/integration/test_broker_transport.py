"""Real loopback protocol checks with fake inference and no SSH or CUDA."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from wa_whisper.broker_server import make_server


def test_http_requires_authentication_and_preserves_protocol():
    class State:
        def request(self, request):
            return {"received": request["action"]}
    server = make_server(State(), "test-private-token", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/rpc"
    try:
        request = urllib.request.Request(url, data=b'{"version":1,"action":"status"}')
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(request, timeout=2)
        assert failure.value.code == 403
        request.add_header("Authorization", "Bearer test-private-token")
        with urllib.request.urlopen(request, timeout=2) as response:
            assert json.load(response) == {"version": 1, "ok": True, "result": {"received": "status"}}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
