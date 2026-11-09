# Contributing to JEV trading bot

## Before you start

1. Read `docs/architecture.md`, `docs/api.md`, and the doc for the specific subsystem you plan to change before writing any code.
2. Search open issues and pull requests to check whether the problem or feature is already being tracked.
3. Never commit `.env` files, Alpaca API keys, JEV API keys, Hugging Face tokens, or locally saved `.pt` model checkpoints; they are listed in `.gitignore` and must stay out of version control.
4. This system targets Alpaca paper trading only; do not add live-trading endpoints, live-trading credentials, or any configuration that enables real order placement.

## Local development

Follow the setup steps in `README.md` (copy `.env.example`, create a virtual environment, install from `requirements.txt`, and start the server with `uvicorn app.main:app --reload`). The directories `__pycache__/`, `.venv/`, `venv/`, and the Hugging Face Hub cache (typically `~/.cache/huggingface/`) are already listed in `.gitignore`; keep them and any locally downloaded checkpoint files out of commits.

## Making a change

- Preserve the module boundary between `app/combined/combiner.py` (pure logic, no IO, no FastAPI imports) and `app/combined/pipeline.py` (IO and stateful coordination); unit tests for `combiner.py` must not require a running server, live Alpaca credentials, or a live JEV API.
- When adding or modifying a FastAPI route in `app/main.py` or `app/combined/api.py`, preserve the existing JSON response field names and types unless the change is intentional, clearly described in the PR, and reflected in an update to `docs/api.md`.
- When modifying signal-combination rules in `app/combined/combiner.py`, update or add corresponding unit tests in `tests/test_combiner.py` that cover the new rule path without live API calls.
- Update the relevant `docs/*.md` page whenever observable behavior changes — route response shapes, configuration options, or model loading behavior.

## Verification

Run the following before submitting a pull request:

```bash
# Run the full test suite.
pytest

# Verify the server starts cleanly with a valid .env.
uvicorn app.main:app --reload

# Optional: test both model integrations on historical data without placing an order.
curl -X POST http://127.0.0.1:8000/combined/dummy \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "quantity": 1}'
```

All tests must pass and the server must start without errors before a PR is opened.

## Pull requests

A pull request description must include:

- The problem being solved and the approach taken, with enough context for a reviewer unfamiliar with the specific subsystem.
- Whether the change is user-visible (new route, changed response field, new environment variable) or internal (refactor, model architecture update, test addition).
- Which tests were run and whether they all pass.
- Whether `docs/` pages were updated to reflect the change.
- A branch name following the `feature/...` or `fix/...` convention (for example, `feature/add-stop-loss-rule` or `fix/jepa-checkpoint-loading`).

## Reporting bugs

A bug report must include:

- Environment: operating system, Python version, CUDA version, and GPU model (or CPU if applicable).
- Reproduction steps: the exact sequence of commands or API calls that trigger the bug.
- Expected behavior and actual behavior.
- Sanitized log output: paste the traceback or error message with all API keys, tokens, and account identifiers replaced by placeholder strings such as `<REDACTED>`.

