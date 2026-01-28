from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Sequence

from langchain.tools import tool as langchain_tool
from mcp.client import ClientSession
from mcp.client.sse import sse_client
from mcp.server.fastmcp import FastMCP

from .config import AppConfig
from .tools_inventory import build_inventory_tools
from .tools_sales import build_sales_tools


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    sse_url: str


def _register_langchain_tool(server: FastMCP, tool) -> None:
    name = tool.name
    description = tool.description or ""

    @server.tool(name=name, description=description)
    def _handler(**kwargs):
        return tool.invoke(kwargs)


def build_salesforce_mcp_server(config: AppConfig) -> FastMCP:
    server = FastMCP("Salesforce")
    for tool in build_sales_tools(config):
        _register_langchain_tool(server, tool)
    return server


def build_sap_business_one_mcp_server(config: AppConfig) -> FastMCP:
    server = FastMCP("SAP Business One")
    for tool in build_inventory_tools(config):
        _register_langchain_tool(server, tool)
    return server


def run_salesforce_mcp_server(config: AppConfig, host: str = "0.0.0.0", port: int = 8000) -> None:
    server = build_salesforce_mcp_server(config)
    server.run(transport="sse", host=host, port=port)


def run_sap_business_one_mcp_server(
    config: AppConfig, host: str = "0.0.0.0", port: int = 8001
) -> None:
    server = build_sap_business_one_mcp_server(config)
    server.run(transport="sse", host=host, port=port)


@dataclass(frozen=True)
class McpSseClient:
    name: str
    sse_url: str

    async def list_tools(self) -> Sequence[Any]:
        async with sse_client(self.sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.list_tools()

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        async with sse_client(self.sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, args)
                if hasattr(result, "content"):
                    return result.content
                return result


def build_salesforce_mcp_client(config: AppConfig) -> McpSseClient:
    return McpSseClient(name="Salesforce", sse_url=config.salesforce_mcp_url)


def build_sap_business_one_mcp_client(config: AppConfig) -> McpSseClient:
    return McpSseClient(name="SAP Business One", sse_url=config.sap_business_one_mcp_url)


def _tool_spec_value(tool_spec, field: str) -> Any:
    if hasattr(tool_spec, field):
        return getattr(tool_spec, field)
    return tool_spec.get(field)


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


def build_mcp_tools(client: McpSseClient) -> list:
    tool_specs = _run_async(client.list_tools())

    def tool_from_spec(tool_spec):
        name = _tool_spec_value(tool_spec, "name")
        description = _tool_spec_value(tool_spec, "description") or f"{client.name} MCP tool"

        @langchain_tool(name=name, description=description)
        def _handler(**kwargs):
            return _run_async(client.call_tool(name, kwargs))

        return _handler

    return [tool_from_spec(spec) for spec in tool_specs]
