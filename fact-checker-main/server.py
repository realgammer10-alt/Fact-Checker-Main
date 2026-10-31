import argparse
import importlib.util
import json
import logging
import os
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Literal, TypedDict, cast
from urllib.parse import urlsplit


LOGGER = logging.getLogger("fact_checker.server")
MAX_REQUEST_BODY_BYTES = 16 * 1024
MAX_ACTIVE_JOBS = 4
MAX_RETAINED_JOBS = 100
MAX_DISPLAYED_JOBS = 10
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DASHBOARD_PATH = Path(__file__).with_name("dashboard.html")


class Job(TypedDict):
    job_id: str
    question: str
    status: Literal["running", "succeeded", "failed"]
    created_at: str
    finished_at: str | None
    duration_seconds: float | None
    result: str | None
    error: str | None
    _started_monotonic: float


class RequestError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class ServiceBusy(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def public_job(job: Job) -> dict[str, object]:
    return {key: value for key, value in job.items() if not key.startswith("_")}


class MonitoringState:
    def __init__(self) -> None:
        self.started_at = utc_now()
        self._started_monotonic = time.monotonic()
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._endpoint_requests: Counter[str] = Counter()
        self._submitted = 0
        self._succeeded = 0
        self._failed = 0

    def record_endpoint_request(self, path: str) -> None:
        if path.startswith("/api/jobs/"):
            path = "/api/jobs/{job_id}"
        elif path not in {"/api/health", "/api/status", "/api/fact-check"}:
            path = "/api/other"
        with self._lock:
            self._endpoint_requests[path] += 1

    def create_job(self, question: str) -> Job:
        with self._lock:
            active = sum(job["status"] == "running" for job in self._jobs.values())
            if active >= MAX_ACTIVE_JOBS:
                raise ServiceBusy

            now = utc_now()
            job: Job = {
                "job_id": uuid.uuid4().hex,
                "question": question,
                "status": "running",
                "created_at": now,
                "finished_at": None,
                "duration_seconds": None,
                "result": None,
                "error": None,
                "_started_monotonic": time.monotonic(),
            }
            self._jobs[job["job_id"]] = job
            self._submitted += 1
            self._prune_jobs()
            return cast(Job, job.copy())

    def finish_job(
        self,
        job_id: str,
        status: Literal["succeeded", "failed"],
        result: str | None = None,
        error: str | None = None,
    ) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown job: {job_id}")
            if job["status"] != "running":
                raise ValueError(f"Job is already {job['status']}.")

            job["status"] = status
            job["finished_at"] = utc_now()
            job["duration_seconds"] = round(
                time.monotonic() - job["_started_monotonic"], 3
            )
            job["result"] = result
            job["error"] = error
            if status == "succeeded":
                self._succeeded += 1
            else:
                self._failed += 1
            return cast(Job, job.copy())

    def get_job(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return public_job(job) if job is not None else None

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            jobs = list(self._jobs.values())
            recent_jobs = [public_job(job) for job in jobs[-MAX_DISPLAYED_JOBS:]]
            active_jobs = sum(job["status"] == "running" for job in jobs)
            endpoint_requests = dict(self._endpoint_requests)
            submitted = self._submitted
            succeeded = self._succeeded
            failed = self._failed

        return {
            "started_at": self.started_at,
            "uptime_seconds": round(time.monotonic() - self._started_monotonic, 1),
            "active_jobs": active_jobs,
            "max_active_jobs": MAX_ACTIVE_JOBS,
            "jobs": {
                "submitted": submitted,
                "succeeded": succeeded,
                "failed": failed,
            },
            "endpoint_requests": endpoint_requests,
            "recent_jobs": list(reversed(recent_jobs)),
        }

    def _prune_jobs(self) -> None:
        while len(self._jobs) > MAX_RETAINED_JOBS:
            removable_job_id = next(
                (
                    job_id
                    for job_id, job in self._jobs.items()
                    if job["status"] != "running"
                ),
                None,
            )
            if removable_job_id is None:
                return
            del self._jobs[removable_job_id]


def fact_checker_readiness() -> dict[str, object]:
    langchain_available = importlib.util.find_spec("langchain") is not None
    api_key_configured = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    return {
        "status": "ready" if langchain_available and api_key_configured else "not_ready",
        "checks": {
            "http_api": "healthy",
            "langchain": "available" if langchain_available else "missing",
            "openai_api_key": "configured" if api_key_configured else "missing",
        },
    }


def run_fact_check(question: str) -> str:
    from fact_checker import fact_check

    return fact_check(question)


class FactCheckerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(
        self,
        server_address: tuple[str, int],
        fact_check_function: Callable[[str], str] | None = None,
    ) -> None:
        self.monitor = MonitoringState()
        self.fact_check_function = fact_check_function or run_fact_check
        self.executor = ThreadPoolExecutor(
            max_workers=MAX_ACTIVE_JOBS, thread_name_prefix="fact-check"
        )
        super().__init__(server_address, FactCheckerRequestHandler)

    def server_close(self) -> None:
        super().server_close()
        self.executor.shutdown(wait=True, cancel_futures=True)


class FactCheckerRequestHandler(BaseHTTPRequestHandler):
    server_version = "FactCheckerDashboard/1.0"

    @property
    def app(self) -> FactCheckerHTTPServer:
        return cast(FactCheckerHTTPServer, self.server)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            self.app.monitor.record_endpoint_request(path)

        if path == "/":
            self._serve_dashboard()
        elif path == "/api/health":
            self._send_json(200, self._health_report())
        elif path == "/api/status":
            self._send_json(200, self.app.monitor.snapshot())
        elif path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/")
            if not job_id or "/" in job_id:
                self._send_json(404, {"error": "Job not found."})
                return
            job = self.app.monitor.get_job(job_id)
            if job is None:
                self._send_json(404, {"error": "Job not found."})
            else:
                self._send_json(200, job)
        elif path.startswith("/api/"):
            self._send_json(404, {"error": "API endpoint not found."})
        else:
            self._send_json(404, {"error": "Page not found."})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            self.app.monitor.record_endpoint_request(path)
        if path != "/api/fact-check":
            self._send_json(404, {"error": "API endpoint not found."})
            return

        try:
            payload = self._read_json()
            if not isinstance(payload, dict):
                raise RequestError(400, "Request body must be a JSON object.")

            from fact_checker import validate_question

            try:
                question = validate_question(payload.get("question"))
            except (TypeError, ValueError) as error:
                raise RequestError(400, str(error)) from error
        except RequestError as error:
            self._send_json(error.status_code, {"error": str(error)})
            return

        try:
            job = self.app.monitor.create_job(question)
        except ServiceBusy:
            self._send_json(
                503,
                {
                    "error": (
                        f"Fact-check capacity is full; retry when fewer than "
                        f"{MAX_ACTIVE_JOBS} jobs are running."
                    )
                },
            )
            return

        try:
            self.app.executor.submit(self._run_job, job["job_id"], question)
        except RuntimeError:
            LOGGER.exception("Unable to schedule fact-check job %s", job["job_id"])
            self.app.monitor.finish_job(
                job["job_id"], "failed", error="Fact-check could not be scheduled."
            )
            self._send_json(503, {"error": "Fact-check service is shutting down."})
            return

        self._send_json(
            202,
            {
                "job_id": job["job_id"],
                "status": "running",
                "status_url": f"/api/jobs/{job['job_id']}",
            },
        )

    def do_PUT(self) -> None:
        self._method_not_allowed()

    def do_PATCH(self) -> None:
        self._method_not_allowed()

    def do_DELETE(self) -> None:
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            self.app.monitor.record_endpoint_request(path)
        self.send_response(405)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_json(self) -> object:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type.lower() != "application/json":
            raise RequestError(415, "Content-Type must be application/json.")

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise RequestError(411, "Content-Length is required.")
        try:
            content_length = int(raw_length)
        except ValueError as error:
            raise RequestError(400, "Content-Length must be an integer.") from error
        if content_length < 0:
            raise RequestError(400, "Content-Length cannot be negative.")
        if content_length > MAX_REQUEST_BODY_BYTES:
            raise RequestError(413, "Request body is too large.")

        try:
            body = self.rfile.read(content_length).decode("utf-8")
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RequestError(400, "Request body must contain valid JSON.") from error

    def _health_report(self) -> dict[str, object]:
        readiness = fact_checker_readiness()
        return {
            "status": "healthy",
            "checked_at": utc_now(),
            "checks": readiness["checks"],
            "fact_checker": readiness["status"],
        }

    def _run_job(self, job_id: str, question: str) -> None:
        try:
            result = self.app.fact_check_function(question)
            if not isinstance(result, str):
                raise TypeError("Fact-check function must return a string.")
        except Exception:
            LOGGER.exception("Fact-check job %s failed", job_id)
            self.app.monitor.finish_job(
                job_id,
                "failed",
                error="Fact-check failed. Check the server logs for details.",
            )
        else:
            self.app.monitor.finish_job(job_id, "succeeded", result=result)

    def _serve_dashboard(self) -> None:
        try:
            content = DASHBOARD_PATH.read_bytes()
        except OSError:
            LOGGER.exception("Unable to read dashboard file at %s", DASHBOARD_PATH)
            self._send_json(500, {"error": "Dashboard could not be loaded."})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status_code: int, payload: object) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format_string: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), format_string % args)


def parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fact-checker monitor.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address.")
    parser.add_argument(
        "--port", type=parse_port, default=DEFAULT_PORT, help="Port to listen on."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        with FactCheckerHTTPServer((args.host, args.port)) as server:
            LOGGER.info("Dashboard available at http://%s:%s", args.host, args.port)
            server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Dashboard server stopped.")


if __name__ == "__main__":
    main()
