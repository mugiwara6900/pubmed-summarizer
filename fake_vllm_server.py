#!/usr/bin/env python3
"""A stand-in for vLLM, for testing failure handling without a DGX Spark.

It returns a canned sentence, NOT real Llama output. It proves how the client
behaves on slow, failing, and dropped connections, and nothing about the model.

    python fake_vllm_server.py --delay 45                  # slow but healthy
    python fake_vllm_server.py --script drop,503,ok        # fail, fail, then succeed

Script steps run in order, one per request, then the last step repeats:
    ok        respond normally
    drop      close the socket without responding
    hang      never respond (long enough to trigger the client's read timeout)
    garbage   return 200 with a body that is not JSON
    <number>  return that HTTP status
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CANNED_REPLY = "[fake server] This is a canned reply, not real model output."


class FakeVLLM(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, script=("ok",), delay=0.0, hang_seconds=3600.0):
        super().__init__(address, _Handler)
        self.script = list(script)
        self.delay = delay
        self.hang_seconds = hang_seconds
        self.requests_seen: list[dict] = []
        self.auth_headers: list[str | None] = []
        self._lock = threading.Lock()

    def next_step(self) -> str:
        with self._lock:
            return self.script.pop(0) if len(self.script) > 1 else self.script[0]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # Keep demo and test output clean.
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests_seen.append(body)
        self.server.auth_headers.append(self.headers.get("Authorization"))

        step = self.server.next_step()
        if step == "drop":
            self.close_connection = True
            self.connection.close()
            return
        if step == "hang":
            time.sleep(self.server.hang_seconds)
            return
        time.sleep(self.server.delay)
        if step == "garbage":
            self._send(200, b"<html>not json</html>")
        elif step == "ok":
            reply = {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": CANNED_REPLY},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 90, "completion_tokens": 12, "total_tokens": 102},
            }
            self._send(200, json.dumps(reply).encode())
        else:
            self._send(int(step), json.dumps({"error": f"simulated {step}"}).encode())

    def _send(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake vLLM server for failure testing")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to wait before responding")
    parser.add_argument("--script", default="ok", help="comma-separated steps, e.g. drop,503,ok")
    args = parser.parse_args()
    server = FakeVLLM(("127.0.0.1", args.port), args.script.split(","), args.delay)
    print(f"Fake vLLM on http://127.0.0.1:{args.port}/v1/chat/completions "
          f"(delay={args.delay}s, script={args.script})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
