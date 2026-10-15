"""
Which MCP server class to use. The same `@server.tool()` code runs on any of three libraries, tried in this order:

  1. `fastmcp`                      (the standalone package)
  2. `mcp.server.fastmcp.FastMCP`   (the official SDK, mcp 1.x)  <- what langchain-mcp-adapters 0.3.x needs on the client side
  3. `mcp.server.mcpserver.MCPServer` (the official SDK, mcp 2.x; FastMCP was renamed)

Why not just `fastmcp`: fastmcp 4.x needs mcp 2.x, while langchain-mcp-adapters 0.3.x needs mcp 1.x, so the two cannot be installed
together. Falling back to the official SDK means the server works with whatever the client side forces on the environment.
Every candidate accepts `Server(name)`, `@server.tool()` and `server.run(transport="stdio")`.
"""
from __future__ import annotations

import importlib


def load_server_class():
    """Return (class, label). Raises ImportError naming every attempt if none is usable."""
    errors = []
    for module, attr in (("fastmcp", "FastMCP"), ("mcp.server.fastmcp", "FastMCP"), ("mcp.server.mcpserver", "MCPServer")):
        try:
            return getattr(importlib.import_module(module), attr), f"{module}.{attr}"
        except (ImportError, AttributeError) as e:
            errors.append(f"{module}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
    raise ImportError("no MCP server library is usable:\n  " + "\n  ".join(errors) + "\nInstall one:  pip install 'mcp<2'")
