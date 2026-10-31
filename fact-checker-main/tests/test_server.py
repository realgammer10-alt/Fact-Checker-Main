import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import (
    FactCheckerHTTPServer,
    MAX_ACTIVE_JOBS,
    MAX_REQUEST_BODY_BYTES,
    MAX_RETAINED_JOBS,
    MonitoringState,
    ServiceBusy,
)


class MonitoringAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = FactCheckerHTTPServer(
            ("127.0.0.1", 0), fact_check_function=lambda question: f"Checked: {question}"
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(
        self, path: str, method: str = "GET", body: bytes | None = None,
        content_type: str | None = None,
    ) -> tuple[int, object]:
        headers = {}
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            response = urlopen(request, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            content = response.read()
            if response.headers.get_content_type() == "application/json":
                return response.status, json.loads(content)
            return response.status, content.decode("utf-8")

    def post_json(self, payload: object) -> tuple[int, object]:
        return self.request(
            "/api/fact-check",
            method="POST",
            body=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
        )

    def test_health_reports_api_and_fact_checker_configuration(self) -> None:
        status_code, health = self.request("/api/health")
        self.assertEqual(status_code, 200)
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["checks"]["http_api"], "healthy")
        self.assertIn(health["fact_checker"], {"ready", "not_ready"})

    def test_dashboard_is_served_with_security_headers(self) -> None:
        request = Request(self.base_url)
        with urlopen(request, timeout=3) as response:
            body = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["X-Frame-Options"], "DENY")
            self.assertIn("Content-Security-Policy", response.headers)
        self.assertIn("Fact Checker Monitor", body)

    def test_status_tracks_endpoint_requests_and_job_totals(self) -> None:
        self.request("/api/status")
        status_code, status = self.request("/api/status")
        self.assertEqual(status_code, 200)
        self.assertGreaterEqual(status["uptime_seconds"], 0)
        self.assertEqual(status["endpoint_requests"]["/api/status"], 2)
        self.assertEqual(status["jobs"]["submitted"], 0)

    def test_fact_check_job_completes_and_can_be_polled(self) -> None:
        status_code, submitted = self.post_json({"question": "Is this accurate?"})
        self.assertEqual(status_code, 202)
        self.assertEqual(submitted["status"], "running")

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status_code, job = self.request(submitted["status_url"])
            self.assertEqual(status_code, 200)
            if job["status"] != "running":
                break
            time.sleep(0.01)

        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["result"], "Checked: Is this accurate?")
        status_code, metrics = self.request("/api/status")
        self.assertEqual(status_code, 200)
        self.assertEqual(metrics["jobs"]["submitted"], 1)
        self.assertEqual(metrics["jobs"]["succeeded"], 1)
        self.assertEqual(metrics["active_jobs"], 0)

    def test_fact_check_errors_are_recorded_without_exposing_exception(self) -> None:
        self.server.fact_check_function = self.fail_fact_check
        status_code, submitted = self.post_json({"question": "Cause an error"})
        self.assertEqual(status_code, 202)

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            _, job = self.request(submitted["status_url"])
            if job["status"] != "running":
                break
            time.sleep(0.01)

        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"], "Fact-check failed. Check the server logs for details.")
        self.assertNotIn("private error", job["error"])

    @staticmethod
    def fail_fact_check(question: str) -> str:
        raise RuntimeError("private error details")

    def test_rejects_invalid_and_oversized_requests(self) -> None:
        cases = [
            (b"{", "application/json", 400),
            (b"[]", "application/json", 400),
            (b'{"question":"   "}', "application/json", 400),
            (b'{"question":"x"}', "text/plain", 415),
            (b"x" * (MAX_REQUEST_BODY_BYTES + 1), "application/json", 413),
        ]
        for body, content_type, expected_status in cases:
            with self.subTest(expected_status=expected_status, body_length=len(body)):
                status_code, response = self.request(
                    "/api/fact-check",
                    method="POST",
                    body=body,
                    content_type=content_type,
                )
                self.assertEqual(status_code, expected_status)
                self.assertIn("error", response)

    def test_rejects_questions_over_limit_and_missing_jobs(self) -> None:
        status_code, response = self.post_json({"question": "x" * 4001})
        self.assertEqual(status_code, 400)
        self.assertIn("4000 characters", response["error"])

        status_code, response = self.request("/api/jobs/missing")
        self.assertEqual(status_code, 404)
        self.assertEqual(response["error"], "Job not found.")

    def test_monitoring_state_caps_active_jobs(self) -> None:
        state = MonitoringState()
        jobs = [
            state.create_job(f"Question {index}")
            for index in range(MAX_ACTIVE_JOBS)
        ]
        with self.assertRaises(ServiceBusy):
            state.create_job("One too many")

        state.finish_job(jobs[0]["job_id"], "succeeded", result="Done")
        next_job = state.create_job("Available slot")
        self.assertEqual(next_job["status"], "running")

    def test_monitoring_state_prunes_old_completed_jobs(self) -> None:
        state = MonitoringState()
        oldest_job_id = ""
        newest_job_id = ""
        for index in range(MAX_RETAINED_JOBS + 1):
            job = state.create_job(f"Question {index}")
            if index == 0:
                oldest_job_id = job["job_id"]
            newest_job_id = job["job_id"]
            state.finish_job(job["job_id"], "succeeded", result="Done")

        snapshot = state.snapshot()
        self.assertEqual(snapshot["jobs"]["submitted"], MAX_RETAINED_JOBS + 1)
        self.assertIsNone(state.get_job(oldest_job_id))
        self.assertIsNotNone(state.get_job(newest_job_id))


if __name__ == "__main__":
    unittest.main()
