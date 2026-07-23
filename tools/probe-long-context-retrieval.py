#!/usr/bin/env python3
"""Run a deterministic needle-retrieval probe at an exact chat-token length."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

FILLER = (
    "Archive entry: routine maintenance completed; no authorization codes, "
    "credentials, tickets, or project secrets are present in this entry.\n"
)
NEEDLE = (
    "\nRETRIEVAL NEEDLE: The exact authorization code for Project Cobalt is "
    "KITE-7391-ONYX. Preserve this code exactly.\n"
)
EXPECTED = "KITE-7391-ONYX"
QUESTION = (
    "The reference archive above contains one authorization code for Project "
    "Cobalt. Return that exact code and nothing else."
)
SYSTEM = "Answer the retrieval question directly. Return only the requested code."


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def repeated_filler(chars: int) -> str:
    if chars <= 0:
        return ""
    return (FILLER * (chars // len(FILLER) + 1))[:chars]


def build_messages(filler_chars: int, needle_fraction: float) -> list[dict[str, str]]:
    left_chars = int(filler_chars * needle_fraction)
    archive = repeated_filler(left_chars) + NEEDLE + repeated_filler(
        filler_chars - left_chars
    )
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "--- BEGIN REFERENCE ARCHIVE ---\n"
                f"{archive}"
                "--- END REFERENCE ARCHIVE ---\n\n"
                f"{QUESTION}"
            ),
        },
    ]


def token_count(base_url: str, model: str, messages: list[dict[str, str]]) -> int:
    data = post_json(
        f"{base_url}/tokenize",
        {"model": model, "messages": messages},
        timeout=300.0,
    )
    count = int(data.get("count", 0))
    if count <= 0:
        raise RuntimeError(f"/tokenize returned an invalid count: {data}")
    return count


def calibrate_messages(
    base_url: str,
    model: str,
    target_tokens: int,
    needle_fraction: float,
) -> tuple[list[dict[str, str]], int, int]:
    low = 0
    high = target_tokens * 8
    best: tuple[int, int, list[dict[str, str]]] | None = None

    while low <= high:
        chars = (low + high) // 2
        messages = build_messages(chars, needle_fraction)
        count = token_count(base_url, model, messages)
        distance = abs(count - target_tokens)
        if best is None or distance < best[0]:
            best = (distance, chars, messages)
        if count == target_tokens:
            return messages, count, chars
        if count < target_tokens:
            low = chars + 1
        else:
            high = chars - 1

    assert best is not None
    distance, chars, messages = best
    count = token_count(base_url, model, messages)
    if distance > max(16, target_tokens // 10_000):
        raise RuntimeError(
            f"could not calibrate near {target_tokens} tokens: got {count}"
        )
    return messages, count, chars


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--model", default="GLM-5.2")
    parser.add_argument("--target-tokens", type=int, default=300_000)
    parser.add_argument("--needle-fraction", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.target_tokens <= 0:
        parser.error("--target-tokens must be positive")
    if not 0.0 <= args.needle_fraction <= 1.0:
        parser.error("--needle-fraction must be between 0 and 1")

    base_url = f"http://{args.host}:{args.port}"
    calibration_started = time.monotonic()
    messages, actual_tokens, filler_chars = calibrate_messages(
        base_url,
        args.model,
        args.target_tokens,
        args.needle_fraction,
    )
    calibration_seconds = time.monotonic() - calibration_started
    prompt_bytes = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )

    request_started = time.monotonic()
    response = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": args.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
        },
        timeout=900.0,
    )
    request_seconds = time.monotonic() - request_started

    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    combined = f"{content}\n{reasoning}"
    passed = EXPECTED in combined
    result = {
        "target_tokens": args.target_tokens,
        "actual_tokens": actual_tokens,
        "filler_chars": filler_chars,
        "needle_fraction": args.needle_fraction,
        "expected": EXPECTED,
        "passed": passed,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "calibration_seconds": calibration_seconds,
        "request_seconds": request_seconds,
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "reasoning": reasoning,
        "usage": response.get("usage"),
        "response_id": response.get("id"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in (
        "target_tokens",
        "actual_tokens",
        "needle_fraction",
        "expected",
        "passed",
        "prompt_sha256",
        "calibration_seconds",
        "request_seconds",
        "finish_reason",
        "usage",
    )}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
