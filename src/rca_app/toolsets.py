from __future__ import annotations

import logging

from .config import AppConfig
from .mcp_client import build_mcp_toolset
from .toolset_types import Toolset
from .tools_inventory import build_inventory_tools
from .tools_sales import build_sales_tools

logger = logging.getLogger(__name__)


def build_salesforce_toolset(config: AppConfig) -> Toolset:
    if config.salesforce_mcp_url:
        logger.info("Loading Salesforce toolset from MCP %s", config.salesforce_mcp_url)
        return build_mcp_toolset("salesforce", config.salesforce_mcp_url)
    logger.info("Building Salesforce toolset")
    tools = build_sales_tools(config)
    return Toolset(
        name="salesforce",
        tools=tools,
        description="Salesforce MCP toolset for sales and promotions data.",
    )


def build_sap_business_one_toolset(config: AppConfig) -> Toolset:
    if config.sap_mcp_url:
        logger.info("Loading SAP Business One toolset from MCP %s", config.sap_mcp_url)
        return build_mcp_toolset("sap-business-one", config.sap_mcp_url)
    logger.info("Building SAP Business One toolset")
    tools = build_inventory_tools(config)
    return Toolset(
        name="sap-business-one",
        tools=tools,
        description="SAP Business One MCP toolset for inventory operations.",
    )

