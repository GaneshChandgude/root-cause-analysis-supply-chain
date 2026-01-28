from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any, Dict, Optional

import requests
from langchain_core.tools import StructuredTool
from pydantic import create_model

from .toolset_types import Toolset

logger = logging.getLogger(__name__)


def _schema_to_pydantic(name: str, schema: Dict[str, Any]):
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: Dict[str, tuple[Any, Any]] = {}
    for prop_name, prop_schema in properties.items():
        field_type = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }.get(prop_schema.get("type", "string"), Any)
        default = ... if prop_name in required else None
        fields[prop_name] = (field_type, default)
    return create_model(name, **fields)  # type: ignore[arg-type]


class McpSseClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._messages_url = "/messages"
        self._session = requests.Session()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._responses: Dict[str, dict] = {}
        self._response_event = threading.Condition()
        self._initialized = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._session.close()

    def _listen(self) -> None:
        url = f"{self.base_url}/sse"
        logger.info("Connecting to MCP SSE stream %s", url)
        try:
            with self._session.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                event_data: list[str] = []
                for raw_line in response.iter_lines(decode_unicode=True):
                    if self._stop_event.is_set():
                        break
                    if raw_line is None:
                        continue
                    line = raw_line.strip()
                    if not line:
                        if event_data:
                            self._handle_event("\n".join(event_data))
                            event_data = []
                        continue
                    if line.startswith("data:"):
                        event_data.append(line.replace("data:", "", 1).strip())
        except Exception:
            logger.exception("MCP SSE listener failed for %s", self.base_url)

    def _handle_event(self, data: str) -> None:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON SSE payload: %s", data)
            return
        if payload.get("url"):
            self._messages_url = payload["url"]
            return
        message_id = payload.get("id")
        if not message_id:
            return
        with self._response_event:
            self._responses[message_id] = payload
            self._response_event.notify_all()

    def _post_message(self, payload: Dict[str, Any]) -> None:
        url = f"{self.base_url}{self._messages_url}"
        response = self._session.post(url, json=payload, timeout=30)
        response.raise_for_status()

    def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.start()
        request_id = uuid.uuid4().hex
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        self._post_message(payload)
        with self._response_event:
            self._response_event.wait_for(lambda: request_id in self._responses, timeout=30)
            response = self._responses.pop(request_id, None)
        if response is None:
            raise TimeoutError(f"Timed out waiting for MCP response to {method}")
        if "error" in response:
            raise RuntimeError(response["error"].get("message", "Unknown MCP error"))
        return response.get("result", {})

    def list_tools(self) -> list[dict]:
        if not self._initialized:
            self.request("initialize")
            self._initialized = True
        result = self.request("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        result = self.request("tools/call", params={"name": name, "arguments": arguments})
        content = result.get("content", [])
        if content and isinstance(content, list):
            text = content[0].get("text")
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        return result


def build_mcp_toolset(name: str, base_url: str) -> Toolset:
    client = McpSseClient(base_url)
    tools = []
    for tool_def in client.list_tools():
        tool_name = tool_def.get("name", "unknown")
        description = tool_def.get("description", "")
        input_schema = tool_def.get("inputSchema") or {"type": "object", "properties": {}}
        args_schema = _schema_to_pydantic(f"{tool_name}Args", input_schema)

        def _remote_tool(*, _tool_name=tool_name, **kwargs):
            return client.call_tool(_tool_name, kwargs)

        remote_tool = StructuredTool.from_function(
            _remote_tool,
            name=tool_name,
            description=description,
            args_schema=args_schema,
        )
        tools.append(remote_tool)

    return Toolset(
        name=name,
        tools=tools,
        description=f"MCP toolset sourced from {base_url}",
    )
