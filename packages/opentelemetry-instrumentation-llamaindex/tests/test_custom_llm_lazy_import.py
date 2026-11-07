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


def test_imports_without_llama_index_llms():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
