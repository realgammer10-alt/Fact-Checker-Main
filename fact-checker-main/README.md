# Fact Checker

This project uses prompt chaining to generate an initial answer, identify and check its assumptions, and produce a revised answer. It includes a command-line interface and a local monitoring dashboard for checking API health and tracking fact-check jobs.

## Requirements

The command-line fact checker uses LangChain's `OpenAI` integration and requires a compatible LangChain installation and an `OPENAI_API_KEY`. The monitoring server, dashboard, and tests use only the Python standard library.

## Run a fact check

```powershell
$env:OPENAI_API_KEY = "your-api-key"
python fact_checker.py "What type of mammal lays the biggest eggs?"
```

The question is optional; when omitted, the example question is used. Questions must contain non-whitespace text and be no longer than 4,000 characters. The provided [`fact_checker.ipynb`](./fact_checker.ipynb) notebook remains available.

## Run the monitoring dashboard

```powershell
python server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The server binds to localhost by default. Change the bind address or port with `python server.py --host 127.0.0.1 --port 8080`. The dashboard has no authentication; do not bind it to a public or shared network interface without adding appropriate access controls.

The dashboard checks API health every 5 seconds and refreshes uptime, endpoint request counts, and job activity every 3 seconds. It submits fact checks as background jobs and polls their status until they finish. At most four jobs run at once; recent job records are retained in memory, up to 100.

### Monitoring API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | API liveness and fact-checker readiness (LangChain installed and `OPENAI_API_KEY` configured). |
| `GET /api/status` | Uptime, active and completed job totals, recent jobs, and endpoint request counts. |
| `POST /api/fact-check` | Start a job with a JSON body such as `{"question":"Is this claim accurate?"}`; returns `202` and a job ID. |
| `GET /api/jobs/{job_id}` | Retrieve a job's running, succeeded, or failed status and its result. |

Health checks verify local configuration, not connectivity or credentials against the external model provider. A fact-check failure is recorded in job status and its details are written to the server log; provider error details are not returned to the dashboard.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

Tests use a stub fact-check function and do not require an API key or make external requests.

## example

Question: *"What type of mammal lays the biggest eggs?"*


**Initial answer:**
The biggest eggs laid by any mammal belong to the elephant.

**Assumptions made:**
- The elephant is a mammal
- Mammals lay eggs
- Eggs come in different sizes
- Elephants lay bigger eggs than other mammals

**Verification of assumptions:**
- The elephant is a mammal: TRUE
- Mammals lay eggs: FALSE - Most mammals give birth to live young.
- Eggs come in different sizes: TRUE
- Elephants lay bigger eggs than other mammals: FALSE - Elephants do not lay eggs.

**New answer:**
This question cannot be answered because elephants do not lay eggs and most mammals give birth to live young.

## credits
Proof of concept by [Jasper](https://twitter.com/0xjasper)
