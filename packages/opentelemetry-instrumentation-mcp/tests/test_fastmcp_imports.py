"""Exercise import hooks in fresh processes, including client-only installations."""

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from textwrap import dedent

import pytest


def _is_installed(package):
    try:
        version(package)
    except PackageNotFoundError:
        return False
    return True


def _run_in_subprocess(script):
    # Import hooks and cached modules must not leak between import-order cases.
    result = subprocess.run(
        [sys.executable, "-c", dedent(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not _is_installed("fastmcp-slim"), reason="requires FastMCP's split distribution")
@pytest.mark.parametrize("import_first", [False, True])
def test_fastmcp_client_imports_do_not_load_server(import_first):
    _run_in_subprocess(f"""
        import sys

        from opentelemetry.instrumentation.mcp import McpInstrumentor
        from wrapt import BoundFunctionWrapper, FunctionWrapper

        if {import_first!r}:
            from fastmcp.client.auth import OAuth
            from fastmcp import Client

        McpInstrumentor().instrument()

        from fastmcp.client.auth import OAuth
        from fastmcp import Client

        assert OAuth is not None
        assert isinstance(Client.__aenter__, (BoundFunctionWrapper, FunctionWrapper))
        assert isinstance(Client.__aexit__, (BoundFunctionWrapper, FunctionWrapper))
        assert not any(
            name == "fastmcp.server" or name.startswith("fastmcp.server.")
            for name in sys.modules
        ), "Client imports unexpectedly loaded FastMCP server support"
    """)


@pytest.mark.skipif(not _is_installed("fastmcp"), reason="requires a full FastMCP installation")
@pytest.mark.parametrize("import_first", [False, True])
@pytest.mark.parametrize("module", ["fastmcp", "fastmcp.server.server"])
def test_fastmcp_server_name_is_captured_in_either_import_order(import_first, module):
    _run_in_subprocess(f"""
        from opentelemetry.instrumentation.mcp import McpInstrumentor

        if {import_first!r}:
            from {module} import FastMCP

        instrumentor = McpInstrumentor()
        instrumentor.instrument()

        from {module} import FastMCP

        server = FastMCP("positional-server")
        assert server.name == "positional-server"
        assert instrumentor._fastmcp_instrumentor._server_name == "positional-server.mcp"

        server = FastMCP(name="keyword-server")
        assert server.name == "keyword-server"
        assert instrumentor._fastmcp_instrumentor._server_name == "keyword-server.mcp"
    """)
