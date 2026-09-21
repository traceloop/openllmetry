"""Offline regression checks for openllmetry #4482 review follow-up.

Runs without the full OpenAI/OTel stack by validating source contracts and
the emitted-id helper logic in isolation.
"""
from __future__ import annotations

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
    assert "_emitted_response_ids_lock" in SRC
    assert "with _emitted_response_ids_lock:" in SRC
    # sync + async completed paths still pop after optional emission
    assert SRC.count("responses.pop(parsed_response.id, None)") >= 2
    # stream path pops via local response_id only when completed
    assert "is_terminal_completed" in SRC
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


def test_stream_checks_emission_before_span_end() -> None:
    assert "should_emit = _mark_response_emitted(response_id)" in SRC
    # emission decision must appear before span.end() in the completed stream path
    stream_fn = SRC.split("def _process_complete_response")[1]
    mark_at = stream_fn.find("_mark_response_emitted")
    end_at = stream_fn.find("self._span.end()")
    assert mark_at != -1 and end_at != -1 and mark_at < end_at
    # duplicates still close the span and still pop the completed entry
    assert "if should_emit:" in stream_fn
    assert "if is_terminal_completed:" in stream_fn
    assert "responses.pop(response_id, None)" in stream_fn


def main() -> int:
    tests = [
        test_source_has_emitted_guard,
        test_mark_emitted_helper,
        test_stream_checks_emission_before_span_end,
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
