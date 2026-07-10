"""server.py - 量化引擎 MCP 服务器入口

基于 Model Context Protocol (MCP) 的量化金融计算服务器。
使用 FastMCP 框架，为 AI 客户端提供期权定价、组合优化、
风险度量和固定收益等金融数学计算工具。

启动方式：
    python server.py

或通过 MCP 客户端（如 Claude Desktop）配置：
    {
        "mcpServers": {
            "quant-engine": {
                "command": "python",
                "args": ["path/to/server.py"]
            }
        }
    }
"""
# 导入 FastMCP 实例（在 __init__.py 中创建）
from mcp_quant_engine import mcp

# 导入所有工具模块，触发 @mcp.tool() 装饰器注册
import mcp_quant_engine.utils       # 4 个工具: parse_returns, parse_matrix, format_result, validate_inputs
import mcp_quant_engine.pricing      # 6 个工具: black_scholes_call/put, implied_vol, monte_carlo_option, option_greeks, binomial_tree
import mcp_quant_engine.portfolio    # 5 个工具: mean_variance_optimize, efficient_frontier, black_litterman, hrp_clustering, portfolio_metrics
import mcp_quant_engine.risk         # 5 个工具: var_historical, var_parametric, var_monte_carlo, cvar, max_drawdown
import mcp_quant_engine.fixed_income # 4 个工具: bond_price, bond_duration, bond_convexity, nelson_siegel

# 总计 24 个 MCP 工具


if __name__ == "__main__":
    mcp.run()
