"""Serve both ModelImage pages through stock Python HTTPServer; assert module MIME."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1]

class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass

server = ThreadingHTTPServer(
    ("127.0.0.1", 0), partial(QuietHandler, directory=str(root))
)
thread = Thread(target=server.serve_forever, daemon=True)
thread.start()
base = "http://127.0.0.1:" + str(server.server_port)

try:
    for module in ("core.js", "world-core.js", "v3-core.js"):
        with urlopen(f"{base}/{module}", timeout=5) as response:
            mime = response.headers.get_content_type()
            body = response.read().decode("utf-8")
            assert response.status == 200, (module, response.status)
            assert mime in ("text/javascript", "application/javascript"), (module, mime)
            assert "export " in body, f"{module}: missing ES module exports"
            print(f"{module}: 200, {mime}, {len(body)} characters")
    for page, module in (
        ("index.html", "core.js"),
        ("world-model.html", "world-core.js"),
        ("v3.html", "v3-core.js"),
    ):
        with urlopen(f"{base}/{page}", timeout=5) as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
            assert f"from './{module}'" in html, (page, module)
            assert ".mjs" not in html, page
            print(f"{page}: 200, imports {module}")
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)

print("Both browser pages and JavaScript MIME types passed.")
