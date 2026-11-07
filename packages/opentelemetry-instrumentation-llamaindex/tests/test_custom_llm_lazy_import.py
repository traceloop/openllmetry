"""The instrumentation must import without any ``llama-index-llms-*`` package installed.

``llama_index.llms`` is a namespace package that only exists once an LLM integration is
installed. The check runs in a subprocess so hiding that package cannot leak into other tests.
"""

import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys
    from unittest.mock import MagicMock


    class HideLlamaIndexLlms(importlib.abc.MetaPathFinder):
        # Simulate an environment with only llama-index-core installed.
        def find_spec(self, name, path=None, target=None):
            if name == "llama_index.llms" or name.startswith("llama_index.llms."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None


    sys.meta_path.insert(0, HideLlamaIndexLlms())

    from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor  # noqa: E402,F401
    from opentelemetry.instrumentation.llamaindex.custom_llm_instrumentor import (  # noqa: E402
        CustomLLMInstrumentor,
    )

    # Nothing to wrap without LLM integrations, so this must be a no-op rather than an error.
    CustomLLMInstrumentor(MagicMock()).instrument()
    """
)


BROKEN_INTEGRATION_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys
    from unittest.mock import MagicMock


    class BrokenLlamaIndexLlms(importlib.abc.MetaPathFinder):
        # Simulate an installed LLM integration whose own dependency is missing.
        def find_spec(self, name, path=None, target=None):
            if name == "llama_index.llms":
                raise ModuleNotFoundError("No module named 'some_missing_dep'", name="some_missing_dep")
            return None


    sys.meta_path.insert(0, BrokenLlamaIndexLlms())

    from opentelemetry.instrumentation.llamaindex.custom_llm_instrumentor import (  # noqa: E402
        CustomLLMInstrumentor,
    )

    try:
        CustomLLMInstrumentor(MagicMock()).instrument()
    except ModuleNotFoundError as exc:
        assert exc.name == "some_missing_dep", exc.name
    else:
        raise AssertionError("expected the broken integration's import error to propagate")
    """
)


def _run(script):
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)


def test_imports_without_llama_index_llms():
    """Importing and instrumenting is a no-op when no LLM integration is installed."""
    result = _run(SCRIPT)
    assert result.returncode == 0, result.stderr


def test_broken_llm_integration_is_not_silenced():
    """A missing dependency of an installed integration still raises instead of being skipped."""
    result = _run(BROKEN_INTEGRATION_SCRIPT)
    assert result.returncode == 0, result.stderr
