"""mcp_quant_engine - 量化引擎MCP服务器工具包

基于 Model Context Protocol (MCP) 的量化金融计算工具集，
为 AI 客户端提供期权定价、组合优化、风险度量和固定收益等计算能力。
"""
from mcp.server.fastmcp import FastMCP

__version__ = "1.0.0"
__author__ = "quant-engine"

# 创建全局 FastMCP 实例，各模块共享此实例注册工具
mcp = FastMCP("quant-engine")

__all__ = ["mcp", "__version__"]
