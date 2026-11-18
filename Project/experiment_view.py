"""Serve the experiment dashboard from the local, read-only tracker."""

import argparse
import json
import sqlite3
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


PROJECT = Path(__file__).resolve().parent
DATABASE = PROJECT / "experiments" / "results.sqlite3"
ASSETS = PROJECT / "experiment_web"
ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
    "/predict": ("predict.html", "text/html; charset=utf-8"),
    "/predict.js": ("predict.js", "text/javascript; charset=utf-8"),
}


def read_runs(run_id=None):
    if not DATABASE.is_file():
        return []
    with closing(sqlite3.connect(f"{DATABASE.as_uri()}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        runs = []
        rows = (
            connection.execute("SELECT * FROM runs ORDER BY id DESC") if run_id is None
            else connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        )
        for row in rows:
            run = dict(row)
            run["params"] = json.loads(run["params"], parse_constant=str)
            run["metrics"] = json.loads(run["metrics"])
            runs.append(run)
        return runs


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/runs":
            try:
                body = json.dumps(read_runs(), allow_nan=False).encode("utf-8")
            except (sqlite3.Error, ValueError):
                self.log_error("Could not read the experiment tracker")
                self.send_error(500, "Could not read the experiment tracker")
                return
            self.respond(body, "application/json; charset=utf-8")
        elif path == "/api/predict/models":
            from predictor import model_option
            self.send_json([option for run in read_runs() if (option := model_option(run))])
        elif path in ROUTES:
            filename, content_type = ROUTES[path]
            self.respond((ASSETS / filename).read_bytes(), content_type)
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in {"/api/predict/load", "/api/predict"}:
            self.send_error(404)
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json({"error": "Send a JSON request."}, 415)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 32768:
                raise ValueError("Invalid request size.")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict) or type(payload.get("run_id")) is not int or payload["run_id"] < 1:
                raise ValueError("Choose a valid experiment run.")
            if payload["run_id"] >= 2**63:
                raise ValueError("Choose a valid experiment run.")

            from predictor import load_model

            model = load_model(payload["run_id"])
            data = model.schema() if path.endswith("/load") else model.predict(payload.get("features"))
            self.send_json(data)
        except (ValueError, OverflowError) as error:
            self.send_json({"error": str(error)}, 400)
        except (OSError, sqlite3.Error):
            self.log_error("Could not load the model or training data")
            self.send_json({"error": "Could not load the model or training data."}, 500)

    def send_json(self, data, status=200):
        self.respond(json.dumps(data, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def respond(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler) as server:
            address = f"http://127.0.0.1:{server.server_port}"
            print(f"Experiment dashboard at {address}/\nPredict at {address}/predict", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as error:
        parser.exit(1, f"Could not start the dashboard: {error}\n")


if __name__ == "__main__":
    main()
