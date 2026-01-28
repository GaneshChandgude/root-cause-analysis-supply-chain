from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Toolset:
    name: str
    tools: List[object]
    description: str

    def get_tool(self, tool_name: str):
        for tool in self.tools:
            if getattr(tool, "name", None) == tool_name:
                return tool
        raise KeyError(f"Tool '{tool_name}' not found in toolset '{self.name}'.")


def find_tool(toolsets: Iterable[Toolset], tool_name: str):
    for toolset in toolsets:
        for tool in toolset.tools:
            if getattr(tool, "name", None) == tool_name:
                logger.debug("Resolved tool %s from toolset %s", tool_name, toolset.name)
                return tool
    raise KeyError(f"Tool '{tool_name}' not found in toolsets.")
