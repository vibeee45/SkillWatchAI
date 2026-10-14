import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

OUTPUT = Path("data/phase11_receiver_received.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/events":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        event = json.loads(self.rfile.read(length))
        event_id = event.get("event_id")

        if not event_id:
            self.send_error(400, "event_id required")
            return

        with OUTPUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": event,
                "idempotency_key": self.headers.get("Idempotency-Key")
            }) + "\n")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "accepted": True,
            "event_id": event_id
        }).encode())

    def log_message(self, format, *args):
        print(format % args)

server = HTTPServer(("127.0.0.1", 8765), Handler)
print("Test receiver listening on http://127.0.0.1:8765/events")
print("Press Ctrl+C to stop.")
server.serve_forever()
