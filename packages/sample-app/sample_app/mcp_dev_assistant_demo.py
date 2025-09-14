#!/usr/bin/env python3
"""
MCP Development Assistant Demo Client
Demonstrates the enhanced OpenLLMetry instrumentation with tool-like spans.
"""

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop


class MCPDevAssistantDemo:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # Initialize OpenLLMetry with console exporter to show the enhanced spans
        Traceloop.init(
            app_name="mcp-dev-assistant-demo-client",
            exporter=ConsoleSpanExporter(),
            disable_batch=True,  # For real-time tracing in demo
        )

    async def connect_to_dev_assistant(self):
        """Connect to the Development Assistant MCP server"""
        # Get the server script path
        server_script = Path(__file__).parent / "mcp_dev_assistant_server.py"

        if not server_script.exists():
            raise FileNotFoundError(f"Server script not found: {server_script}")

        # Set up server parameters
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_script)],
            env=os.environ
        )

        # Connect to the server
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()
        print("\n✅ Connected to Development Assistant MCP Server")

    async def demo_enhanced_tracing(self):
        """Demonstrate the enhanced MCP instrumentation with various tools"""
        print("\n🚀 Starting MCP Enhanced Instrumentation Demo")
        print("=" * 60)

        # Demo 1: List available tools
        print("\n1️⃣  Listing available tools...")
        tools_response = await self.session.list_tools()
        available_tools = [tool.name for tool in tools_response.tools]
        print(f"Available tools: {', '.join(available_tools)}")

        # Demo 2: Get system information
        print("\n2️⃣  Getting system information...")
        try:
            await self.session.call_tool("get_system_info", {})
            print("System info retrieved successfully")
        except Exception as e:
            print(f"System info failed: {e}")

        # Demo 3: List files in current directory
        print("\n3️⃣  Listing files in current directory...")
        try:
            await self.session.call_tool("list_files", {"directory": "."})
            print("Files listed successfully")
        except Exception as e:
            print(f"List files failed: {e}")

        # Demo 4: Search for Python imports
        print("\n4️⃣  Searching for Python imports...")
        try:
            await self.session.call_tool("search_code", {
                "pattern": "import",
                "directory": ".",
                "file_extensions": [".py"],
                "max_results": 10
            })
            print("Code search completed successfully")
        except Exception as e:
            print(f"Code search failed: {e}")

        # Demo 5: Check Git status (if in a git repo)
        print("\n5️⃣  Checking Git status...")
        try:
            await self.session.call_tool("git_status", {"repository_path": "."})
            print("Git status retrieved successfully")
        except Exception as e:
            print(f"Git status failed: {e}")

        # Demo 6: Run a simple command
        print("\n6️⃣  Running a simple command...")
        try:
            await self.session.call_tool("run_command", {
                "command": "echo 'Hello from MCP!'",
                "timeout": 5
            })
            print("Command executed successfully")
        except Exception as e:
            print(f"Command execution failed: {e}")

        # Demo 7: Create and read a test file
        print("\n7️⃣  Creating and reading a test file...")
        try:
            # Write a test file
            await self.session.call_tool("write_file", {
                "file_path": "/tmp/mcp_test.txt",
                "content": "Hello from MCP Development Assistant!\nThis is a test file created during the demo."
            })
            print("Test file created successfully")

            # Read it back
            await self.session.call_tool("read_file", {
                "file_path": "/tmp/mcp_test.txt",
                "max_lines": 5
            })
            print("Test file read successfully")

        except Exception as e:
            print(f"File operations failed: {e}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    async def run(self):
        """Run the complete demo"""
        try:
            await self.connect_to_dev_assistant()
            await self.demo_enhanced_tracing()

            print("\n" + "=" * 60)
            print("🎉 Demo completed! Check the trace output above to see:")
            print("   • Tool-like span names (e.g., 'list_files.tool', 'git_status.tool')")
            print("   • Clean JSON input/output instead of raw serialization")
            print("   • Proper Traceloop span attributes (traceloop.span.kind = 'tool')")
            print("   • Meaningful entity names for each tool call")

        except KeyboardInterrupt:
            print("\n\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
        finally:
            await self.cleanup()


async def main():
    """Main entry point"""
    print("MCP Development Assistant - Enhanced Instrumentation Demo")
    print("This demo showcases the enhanced OpenLLMetry MCP instrumentation")
    print("that makes MCP tool calls appear like @tool decorated functions.")

    demo = MCPDevAssistantDemo()
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())
