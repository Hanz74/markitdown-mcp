"""MarkItDown MCP stdio transport — runs the MCP server over stdin/stdout.

Reuses the FastMCP instance from server.py so all tool definitions
are shared with the SSE transport. Logging goes to stderr to keep stdout
clean for the MCP JSON-RPC channel.

Usage (inside container):
    python mcp/mcp_stdio.py
"""

import asyncio
import logging
import os
import sys

from mcp.server.stdio import stdio_server

from server import mcp

# --- Logging → stderr only (stdout is the MCP channel) ---
logging.basicConfig(
    stream=sys.stderr,
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("markitdown.stdio")

_mcp_server = mcp._mcp_server


async def main() -> None:
    log.info("MarkItDown MCP stdio server ready — waiting for client on stdin")
    async with stdio_server() as (read_stream, write_stream):
        await _mcp_server.run(
            read_stream,
            write_stream,
            _mcp_server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
