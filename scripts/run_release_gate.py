"""Run and publish exactly one Phase 8 release gate.

Examples:
    .venv/bin/python scripts/run_release_gate.py transcript
    .venv/bin/python scripts/run_release_gate.py domain
    .venv/bin/python scripts/run_release_gate.py contract
    .venv/bin/python scripts/run_release_gate.py supabase
    CARTISAN_BROWSER_QA_COMMAND='...' .venv/bin/python scripts/run_release_gate.py browser

Every invocation writes its own JSON record. Missing live credentials or a missing
browser command produce ``not_run`` and a non-zero exit, never a green skip.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "release_evidence"
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from cartisan_agent.merchant_types import Claim  # noqa: E402

COMMANDS = {
    "domain": [[sys.executable, "-m", "pytest", "-q", "-m", "domain"]],
    "contract": [[sys.executable, "-m", "pytest", "-q", "-m", "contract"]],
    "transcript": [[sys.executable, "-m", "pytest", "-q", "-m", "transcript"]],
    "supabase": [
        [sys.executable, "scripts/verify_phase5_live.py"],
        [sys.executable, "scripts/verify_phase6_live.py"],
        [sys.executable, "scripts/verify_phase7_live.py"],
    ],
}


def _precondition(gate: str) -> str | None:
    if gate == "supabase" and not os.getenv("SUPABASE_DATABASE_URL"):
        return "SUPABASE_DATABASE_URL is not configured"
    if gate == "browser" and not os.getenv("CARTISAN_BROWSER_QA_COMMAND"):
        return "CARTISAN_BROWSER_QA_COMMAND is not configured"
    return None


def _commands(gate: str) -> list[list[str]]:
    if gate == "browser":
        return [shlex.split(os.environ["CARTISAN_BROWSER_QA_COMMAND"])]
    return COMMANDS[gate]


def _redact(output: str) -> str:
    output = re.sub(r"https://rzp\.io/\S+", "[razorpay-url-redacted]", output)
    output = re.sub(r"(?i)(token|secret|password|apikey)=\S+", r"\1=[redacted]", output)
    return output[-20_000:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", choices=[*COMMANDS, "browser"])
    args = parser.parse_args()
    started = datetime.now(UTC)
    reason = _precondition(args.gate)
    outputs: list[str] = []
    completed = 0
    total = len(COMMANDS.get(args.gate, [None]))
    status = "not_run" if reason else "passed"

    if not reason:
        for command in _commands(args.gate):
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT)
            if args.gate in {"domain", "contract", "transcript"}:
                # These gates are deterministic and local by definition. Loading the
                # developer's live .env would silently turn an API import into a
                # Supabase check and collapse two independent gates into one.
                environment.pop("SUPABASE_DATABASE_URL", None)
                environment["CARTISAN_DB_PATH"] = str(
                    Path("/tmp") / f"cartisan-{args.gate}-{os.getpid()}.db"
                )
            result = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, env=environment
            )
            outputs.append(_redact(result.stdout + result.stderr))
            if result.returncode != 0:
                status = "failed"
                reason = f"command exited {result.returncode}: {' '.join(command)}"
                break
            completed += 1

    ratio = Claim(
        key="gate_pass_rate",
        value=(completed / total) if status != "not_run" else None,
        unit="ratio",
        basis="completed gate commands / required gate commands",
        inputs={"completed": completed, "required": total},
        limitations=[] if status == "passed" else [reason or "gate did not pass"],
    )
    correlations = sorted(set(re.findall(r"\b(?:corr|correlation)[_ :=-]+([A-Za-z0-9_-]+)", "\n".join(outputs), re.I)))
    report = {
        "schema_version": 1,
        "gate": args.gate,
        "status": status,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "claim": ratio.payload(),
        "correlation_ids": correlations,
        "reason": reason,
        "commands": _commands(args.gate) if not reason or args.gate != "browser" else [],
        "output": outputs,
    }
    REPORTS.mkdir(exist_ok=True)
    destination = REPORTS / f"{args.gate}-{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.gate}: {status}; evidence={destination.relative_to(ROOT)}")
    if reason:
        print(reason)
    return 0 if status == "passed" else (2 if status == "not_run" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
