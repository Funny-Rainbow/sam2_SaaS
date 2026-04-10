#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_segment(payload: dict) -> dict:
    from saas_api import run_segment as run_segment_api

    return run_segment_api(payload)


def _error_payload(exc: Exception) -> dict:
    return {
        "ok": False,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "stack": traceback.format_exc(),
        },
    }


def _dispatch(command: str, payload: dict) -> dict:
    if command == "segment":
        return run_segment(payload)
    raise RuntimeError(f"Unsupported command: {command}")


def _normalize_success_payload(payload: dict) -> dict:
    if isinstance(payload, dict):
        if "ok" not in payload:
            return {"ok": True, **payload}
        return payload
    return {"ok": True, "result": payload}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["segment"])
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--response-json", required=True)
    args = parser.parse_args()

    request_path = Path(args.request_json)
    response_path = Path(args.response_json)

    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        result = _dispatch(args.command, payload)
        write_json(response_path, _normalize_success_payload(result))
        return 0
    except Exception as exc:
        failure = _error_payload(exc)
        try:
            write_json(response_path, failure)
        finally:
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
