"""Offline regression checks for openllmetry #4482 review follow-up.

Runs without the full OpenAI/OTel stack by validating source contracts and
the emitted-id helper logic in isolation.
"""
from __future__ import annotations

import ast
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # packages/opentelemetry-instrumentation-openai
SRC = (
    ROOT / "opentelemetry/instrumentation/openai/v1/responses_wrappers.py"
).read_text(encoding="utf-8")


def _mark_response_emitted(response_id, emitted: OrderedDict, max_size: int = 2048) -> bool:
    if not response_id:
        return False
    if response_id in emitted:
        emitted.move_to_end(response_id)
        return False
    emitted[response_id] = None
    while len(emitted) > max_size:
        emitted.popitem(last=False)
    return True


def test_source_has_emitted_guard() -> None:
    assert "_mark_response_emitted" in SRC
    assert "_EMITTED_RESPONSE_IDS_MAX" in SRC
    # sync + async completed paths still pop after optional emission
    assert SRC.count("responses.pop(parsed_response.id, None)") >= 2
    # stream path pops via local response_id only when completed
    assert "if status == \"completed\" and response_id:" in SRC
    assert "responses.pop(response_id, None)" in SRC
    # must not unconditionally pop on any stream exit
    assert "if getattr(self._traced_data, \"response_id\", None):\n                    responses.pop" not in SRC


def test_mark_emitted_helper() -> None:
    emitted: OrderedDict[str, None] = OrderedDict()
    assert _mark_response_emitted("resp_1", emitted) is True
    assert _mark_response_emitted("resp_1", emitted) is False
    assert _mark_response_emitted("resp_2", emitted) is True
    assert _mark_response_emitted("", emitted) is False
    assert _mark_response_emitted(None, emitted) is False
    # eviction keeps helper best-effort
    small: OrderedDict[str, None] = OrderedDict()
    for i in range(5):
        assert _mark_response_emitted(f"id_{i}", small, max_size=3) is True
    assert len(small) <= 3
    # evicted id can be marked again
    assert _mark_response_emitted("id_0", small, max_size=3) is True


def test_stream_pop_condition_is_status_gated() -> None:
    assert 'if status == "completed" and response_id:' in SRC
    assert "response_status" in SRC
    # completed path marks emission and pops; non-completed must not
    completed_block = SRC.split("if status == \"completed\" and response_id:")[1][:200]
    assert "_mark_response_emitted" in completed_block
    assert "responses.pop(response_id, None)" in completed_block


def main() -> int:
    tests = [
        test_source_has_emitted_guard,
        test_mark_emitted_helper,
        test_stream_pop_condition_is_status_gated,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
