from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    return loop.run_until_complete(coro)


def run_segment(payload: dict[str, Any]) -> dict[str, Any]:
    from sam2_segment_api import SegmentRequest, segment_video

    request = SegmentRequest(**payload)
    response = _run_async(segment_video(request))
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif hasattr(response, "dict"):
        data = response.dict()
    elif isinstance(response, dict):
        data = response
    else:
        data = {"result": response}

    if isinstance(data, dict):
        data.setdefault("ok", data.get("success", True))
        return data
    return {"ok": True, "result": data}


def segment(payload: dict[str, Any]) -> dict[str, Any]:
    return run_segment(payload)
