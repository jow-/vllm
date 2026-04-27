# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Plaintext response trace logger.

When VLLM_RESPONSE_TRACE_LOG is set to a writable file path, every raw model
response chunk (before any reasoning or tool-call parsing) is appended to that
file in the format:

    [timestamp_with_ms] chunk_len

    <raw chunk bytes>


This is useful for debugging reasoning and tool-call parser edge cases.
"""

import contextlib
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_ENV_VAR = "VLLM_RESPONSE_TRACE_LOG"

# Module-level singleton state
_trace_file: object = None  # either None or an open file
_trace_lock = Lock()


def init_response_trace():
    """
    Initialize the response trace logger from the environment variable.
    Call once at server startup.
    """
    global _trace_file

    path = os.environ.get(_ENV_VAR)
    if not path:
        return

    with contextlib.suppress(OSError):
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
        _trace_file = open(path, "a", encoding="utf-8")  # noqa: SIM115


def trace_chunk(text: str):
    """
    Append a single raw response chunk to the trace log.

    Args:
        text: The raw decoded text from one engine output step, before any
              reasoning or tool-call parsing.
    """
    global _trace_file
    fh = _trace_file
    if fh is None:
        return

    chunk_bytes = text.encode("utf-8")
    chunk_len = len(chunk_bytes)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    line = f"[{ts}] {chunk_len}\n{text}\n\n"

    with _trace_lock:
        try:
            fh.write(line)
            fh.flush()
        except OSError:
            _trace_file = None


def close_response_trace():
    """Close the trace file if it is open."""
    global _trace_file
    if _trace_file is not None:
        with contextlib.suppress(OSError):
            _trace_file.close()
        _trace_file = None
