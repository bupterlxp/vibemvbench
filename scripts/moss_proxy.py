"""Tiny forwarding proxy for the MOSS (Nex) responses API: adds max_output_tokens to every request, because Codex sends
none and the server default (8192) cuts long tool calls off ("Incomplete response returned, reason: max_output_tokens").

  python scripts/moss_proxy.py [port=18765] [max_output_tokens=65536]
Point the Codex provider base_url at http://127.0.0.1:<port>/v1 (containers use --network host).
"""
import http.server, json, os, socketserver, sys, urllib.request

UP = os.environ["MOSS_API_ORIGIN"]
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18765
MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 65536


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self, body):
        req = urllib.request.Request(UP + self.path, data=body, method=self.command,
                                     headers={k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "accept-encoding")})
        try:
            r = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            r = e
        self.send_response(r.status if hasattr(r, "status") else r.code)
        for k, v in r.headers.items():
            if k.lower() not in ("transfer-encoding", "content-length", "connection", "content-encoding"):
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.endswith("/responses"):
            try:
                d = json.loads(body); d.setdefault("max_output_tokens", MAX); body = json.dumps(d).encode()
            except Exception:
                pass
        self._forward(body)

    def do_GET(self):
        self._forward(None)

    def log_message(self, *a):
        pass


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    S(("127.0.0.1", PORT), H).serve_forever()
