"""FastMCP instrumentation must not load server support in client-only installs.

The session fixture already instruments this process and imports fastmcp, so
the import-boundary checks run in a fresh interpreter.
"""

import subprocess
import sys
import textwrap

import pytest
from opentelemetry.instrumentation.mcp.fastmcp_instrumentation import FastMCPInstrumentor

# Simulates fastmcp-slim[client]: the server-only dependency is missing.
_BLOCK_SERVER = """
import importlib.abc
import sys

class _BlockServer(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "uncalled_for":
            raise ImportError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, _BlockServer())
"""

_INSTRUMENT = """
from opentelemetry.instrumentation.mcp import McpInstrumentor
instrumentor = McpInstrumentor()
instrumentor.instrument()
"""

_CLIENT_IMPORTS = """
from fastmcp.client.auth import OAuth
from fastmcp import Client
assert OAuth is not None and Client is not None
"""

_SERVER_NOT_LOADED = """
assert "fastmcp.server.server" not in sys.modules
"""


def _run(*snippets):
    code = "\n".join(textwrap.dedent(snippet) for snippet in snippets)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_client_only_imports_after_instrument():
    # Control: the imports work without instrumentation under the same blocker.
    _run(_BLOCK_SERVER, _CLIENT_IMPORTS, _SERVER_NOT_LOADED)
    _run(_BLOCK_SERVER, _INSTRUMENT, _CLIENT_IMPORTS, _SERVER_NOT_LOADED)


def test_client_only_imports_before_instrument():
    _run(_BLOCK_SERVER, _CLIENT_IMPORTS, _INSTRUMENT, _SERVER_NOT_LOADED)


def test_fastmcp_init_still_captures_server_name():
    _run(
        _INSTRUMENT,
        """
        from fastmcp import FastMCP

        FastMCP("name-test")
        assert instrumentor._fastmcp_instrumentor._server_name == "name-test.mcp"
        FastMCP(name="kw-test")
        assert instrumentor._fastmcp_instrumentor._server_name == "kw-test.mcp"
        """,
    )


def test_fastmcp_init_error_propagates():
    instrumentor = FastMCPInstrumentor()
    instrumentor._server_name = "before.mcp"
    error = ValueError("bad server config")

    def failing_init(*args, **kwargs):
        raise error

    with pytest.raises(ValueError) as exc_info:
        instrumentor._fastmcp_init_wrapper()(failing_init, None, ("broken",), {})

    assert exc_info.value is error
    assert instrumentor._server_name == "before.mcp"
