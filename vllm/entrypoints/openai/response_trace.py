# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Plaintext response trace loggers.

VLLM_RESPONSE_TRACE_LOG
    When set to a writable file path, every raw model response chunk (before
    any reasoning or tool-call parsing) is appended in the format:

        [timestamp_with_ms] chunk_len

        <raw chunk bytes>

VLLM_PARSED_RESPONSE_TRACE_LOG
    When set to a writable file path, every post-parsing SSE line that is
    about to be sent to the client is appended verbatim (one line per chunk).

Both are useful for debugging reasoning and tool-call parser edge cases.
"""

import contextlib
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_ENV_VAR = "VLLM_RESPONSE_TRACE_LOG"
_PARSED_ENV_VAR = "VLLM_PARSED_RESPONSE_TRACE_LOG"

# Module-level singleton state – raw pre-parsing trace
_trace_file: object = None  # either None or an open file
_trace_lock = Lock()

# Module-level singleton state – post-parsing (OpenAI SSE) trace
_parsed_trace_file: object = None
_parsed_trace_lock = Lock()


def init_response_trace():
    """
    Initialize both response trace loggers from environment variables.
    Call once at server startup.
    """
    global _trace_file, _parsed_trace_file

    path = os.environ.get(_ENV_VAR)
    if path:
        with contextlib.suppress(OSError):
            parent = Path(path).parent
            parent.mkdir(parents=True, exist_ok=True)
            _trace_file = open(path, "a", encoding="utf-8")  # noqa: SIM115

    parsed_path = os.environ.get(_PARSED_ENV_VAR)
    if parsed_path:
        with contextlib.suppress(OSError):
            parent = Path(parsed_path).parent
            parent.mkdir(parents=True, exist_ok=True)
            _parsed_trace_file = open(  # noqa: SIM115
                parsed_path, "a", encoding="utf-8"
            )


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


def trace_parsed_chunk(sse_line: str):
    """
    Append one post-parsing SSE line to the parsed trace log.

    Args:
        sse_line: The full SSE payload string (e.g. ``data: {...}\\n\\n``)
                  that is about to be sent to the client.
    """
    global _parsed_trace_file
    fh = _parsed_trace_file
    if fh is None:
        return

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    line = f"[{ts}] {sse_line}"
    if not line.endswith("\n"):
        line += "\n"

    with _parsed_trace_lock:
        try:
            fh.write(line)
            fh.flush()
        except OSError:
            _parsed_trace_file = None


def close_response_trace():
    """Close both trace files if they are open."""
    global _trace_file, _parsed_trace_file
    if _trace_file is not None:
        with contextlib.suppress(OSError):
            _trace_file.close()
        _trace_file = None
    if _parsed_trace_file is not None:
        with contextlib.suppress(OSError):
            _parsed_trace_file.close()
        _parsed_trace_file = None
