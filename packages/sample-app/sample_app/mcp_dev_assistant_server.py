#!/usr/bin/env python3
"""
MCP Development Assistant Server
A sample MCP server with useful development tools to demonstrate OpenLLMetry instrumentation.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel
from traceloop.sdk import Traceloop

# Load environment variables
load_dotenv()

# Initialize OpenTelemetry with Traceloop SDK (automatically includes MCP instrumentation)
Traceloop.init(
    app_name="dev-assistant-mcp-server",
    disable_batch=True,  # For real-time tracing in demo
)

# Initialize the MCP server
server = FastMCP("dev-assistant")


class FileInfo(BaseModel):
    """File information structure"""
    name: str
    size: int
    modified: str
    is_directory: bool


class GitStatus(BaseModel):
    """Git status structure"""
    branch: str
    staged: List[str]
    modified: List[str]
    untracked: List[str]
    ahead: int
    behind: int


class ProcessResult(BaseModel):
    """Process execution result"""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float


@server.tool()
def list_files(directory: str = ".") -> List[FileInfo]:
    """
    List files and directories in the specified path.

    Args:
        directory: The directory path to list (defaults to current directory)

    Returns:
        List of files and directories with their information
    """
    try:
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")

        files = []
        for item in path.iterdir():
            stat = item.stat()
            files.append(FileInfo(
                name=item.name,
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                is_directory=item.is_dir()
            ))

        # Sort directories first, then files alphabetically
        files.sort(key=lambda x: (not x.is_directory, x.name.lower()))
        return files

    except Exception as e:
        raise RuntimeError(f"Failed to list files: {str(e)}")


@server.tool()
def read_file(file_path: str, max_lines: Optional[int] = None) -> str:
    """
    Read the contents of a text file.

    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read (optional)

    Returns:
        File contents as a string
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")

        if not path.is_file():
            raise ValueError(f"{file_path} is not a file")

        with open(path, 'r', encoding='utf-8') as f:
            if max_lines:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip('\n'))
                return '\n'.join(lines)
            else:
                return f.read()

    except Exception as e:
        raise RuntimeError(f"Failed to read file: {str(e)}")


@server.tool()
def write_file(file_path: str, content: str, append: bool = False) -> str:
    """
    Write content to a file.

    Args:
        file_path: Path where to write the file
        content: Content to write
        append: Whether to append to existing file (default: False)

    Returns:
        Success message with file info
    """
    try:
        path = Path(file_path)

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = 'a' if append else 'w'
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)

        stat = path.stat()
        return f"Successfully {'appended to' if append else 'wrote'} {file_path} ({stat.st_size} bytes)"

    except Exception as e:
        raise RuntimeError(f"Failed to write file: {str(e)}")


@server.tool()
def run_command(command: str, working_directory: str = ".", timeout: int = 30) -> ProcessResult:
    """
    Execute a shell command and return the result.

    Args:
        command: Shell command to execute
        working_directory: Directory to run the command in (default: current)
        timeout: Command timeout in seconds (default: 30)

    Returns:
        Process execution result with stdout, stderr, and exit code
    """
    try:
        start_time = datetime.now()

        result = subprocess.run(
            command,
            shell=True,
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        return ProcessResult(
            command=command,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            execution_time=execution_time
        )

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout} seconds")
    except Exception as e:
        raise RuntimeError(f"Failed to execute command: {str(e)}")


@server.tool()
def git_status(repository_path: str = ".") -> GitStatus:
    """
    Get Git repository status information.

    Args:
        repository_path: Path to the Git repository (default: current directory)

    Returns:
        Git status information including branch, staged/modified files, etc.
    """
    try:
        repo_path = Path(repository_path)
        if not (repo_path / '.git').exists():
            raise ValueError(f"No Git repository found in {repository_path}")

        # Get current branch
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

        # Get status
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )

        staged = []
        modified = []
        untracked = []

        if status_result.returncode == 0:
            for line in status_result.stdout.strip().split('\n'):
                if not line:
                    continue
                status_code = line[:2]
                filename = line[3:]

                if status_code[0] in ['A', 'M', 'D', 'R', 'C']:
                    staged.append(filename)
                if status_code[1] in ['M', 'D']:
                    modified.append(filename)
                if status_code == '??':
                    untracked.append(filename)

        # Get ahead/behind info
        ahead, behind = 0, 0
        try:
            ahead_behind_result = subprocess.run(
                ["git", "rev-list", "--left-right", "--count", f"{branch}...origin/{branch}"],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            if ahead_behind_result.returncode == 0:
                parts = ahead_behind_result.stdout.strip().split('\t')
                ahead = int(parts[0]) if len(parts) > 0 else 0
                behind = int(parts[1]) if len(parts) > 1 else 0
        except Exception:
            pass

        return GitStatus(
            branch=branch,
            staged=staged,
            modified=modified,
            untracked=untracked,
            ahead=ahead,
            behind=behind
        )

    except Exception as e:
        raise RuntimeError(f"Failed to get Git status: {str(e)}")


@server.tool()
def search_code(
    pattern: str,
    directory: str = ".",
    file_extensions: List[str] = None,
    max_results: int = 50
) -> Dict[str, List[str]]:
    """
    Search for a pattern in code files.

    Args:
        pattern: Text pattern to search for
        directory: Directory to search in (default: current)
        file_extensions: List of file extensions to include (e.g., ['.py', '.js'])
        max_results: Maximum number of matches to return (default: 50)

    Returns:
        Dictionary mapping file paths to lists of matching lines
    """
    try:
        if file_extensions is None:
            file_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.rb', '.go', '.rs']

        search_path = Path(directory)
        if not search_path.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")

        results = {}
        total_matches = 0

        for ext in file_extensions:
            if total_matches >= max_results:
                break

            for file_path in search_path.rglob(f"*{ext}"):
                if total_matches >= max_results:
                    break

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        matches = []
                        for line_num, line in enumerate(f, 1):
                            if pattern.lower() in line.lower():
                                matches.append(f"{line_num}: {line.strip()}")
                                total_matches += 1
                                if total_matches >= max_results:
                                    break

                        if matches:
                            results[str(file_path.relative_to(search_path))] = matches

                except Exception:
                    continue  # Skip files that can't be read

        return results

    except Exception as e:
        raise RuntimeError(f"Failed to search code: {str(e)}")


@server.tool()
def get_system_info() -> Dict[str, Any]:
    """
    Get system information including Python version, OS, and environment.

    Returns:
        Dictionary containing system information
    """
    try:
        import platform

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_executable": sys.executable,
            "current_directory": str(Path.cwd()),
            "environment_variables": {
                k: v for k, v in os.environ.items()
                if not any(secret in k.lower() for secret in ['key', 'token', 'password', 'secret'])
            }
        }

    except Exception as e:
        raise RuntimeError(f"Failed to get system info: {str(e)}")


def main():
    """Main entry point for the server"""
    # Get transport from environment or default to stdio
    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    print(f"Starting Development Assistant MCP Server on {transport}", file=sys.stderr)

    # List available tools (FastMCP stores tools differently)
    tool_names = [
        "list_files", "read_file", "write_file", "run_command",
        "git_status", "search_code", "get_system_info"
    ]
    print(f"Server has {len(tool_names)} tools available:", file=sys.stderr)
    for tool_name in tool_names:
        print(f"  - {tool_name}", file=sys.stderr)

    # Run the server
    server.run(transport=transport)


if __name__ == "__main__":
    main()
