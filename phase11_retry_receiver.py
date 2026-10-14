import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OUTPUT = Path("data/phase11_retry_receiver_received.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

state = {"calls": 0}
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/events":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        event = json.loads(self.rfile.read(length))

        with lock:
            state["calls"] += 1
            call_number = state["calls"]

        print(f"Request {call_number}: event_id={event.get('event_id')}")

        if call_number == 1:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"accepted": False, "reason": "simulated failure"}).encode()
            )
            print("SIMULATED FAILURE: HTTP 503")
            return

        with OUTPUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": event,
                "idempotency_key": self.headers.get("Idempotency-Key")
            }) + "\n")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({
                "accepted": True,
                "event_id": event.get("event_id")
            }).encode()
        )
        print("SUCCESS: HTTP 200")

    def log_message(self, format, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 8766), Handler)
print("Retry test receiver: http://127.0.0.1:8766/events")
print("First POST returns HTTP 503; subsequent POSTs return HTTP 200.")
print("Press Ctrl+C to stop.")
server.serve_forever()
