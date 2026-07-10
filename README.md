# 量化引擎 MCP 服务器 (mcp-quant-engine)

基于 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 的量化金融计算服务器，使用 [FastMCP](https://github.com/modelcontextprotocol/python-sdk) 框架，为 AI 客户端（如 Claude）提供专业的金融数学计算工具。

## 功能概览

本服务器提供 **24 个 MCP 工具**，覆盖量化金融四大核心领域：

| 模块 | 工具数量 | 功能 |
|------|----------|------|
| pricing.py | 6 | 期权定价（BS模型、隐含波动率、蒙特卡洛、Greeks、二叉树） |
| portfolio.py | 5 | 组合优化（均值方差、有效前沿、Black-Litterman、HRP、绩效指标） |
| risk.py | 5 | 风险度量（历史VaR、参数法VaR、MC VaR、CVaR、最大回撤） |
| fixed_income.py | 4 | 固定收益（债券定价、久期、凸性、Nelson-Siegel曲线） |
| utils.py | 4 | 工具函数（收益解析、矩阵解析、格式化、输入验证） |

## 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/mcp-quant-engine.git
cd mcp-quant-engine

# 安装依赖
pip install -r requirements.txt
```

## 使用

### 直接运行

```bash
python server.py
```

### 配置 MCP 客户端

在 Claude Desktop 配置文件中添加：

```json
{
    "mcpServers": {
        "quant-engine": {
            "command": "python",
            "args": ["path/to/mcp-quant-engine/server.py"]
        }
    }
}
```

## 工具列表

### 期权定价工具 (pricing.py)

| 工具 | 描述 | 参数 |
|------|------|------|
| `black_scholes_call` | BS看涨期权定价 | S, K, T, r, sigma |
| `black_scholes_put` | BS看跌期权定价 | S, K, T, r, sigma |
| `implied_vol` | 隐含波动率（牛顿迭代法） | price, S, K, T, r, option_type |
| `monte_carlo_option` | 蒙特卡洛期权定价 | S, K, T, r, sigma, n_sims, option_type |
| `option_greeks` | 全部Greeks计算 | S, K, T, r, sigma, option_type |
| `binomial_tree` | 二叉树定价（美式期权） | S, K, T, r, sigma, steps, option_type |

### 组合优化工具 (portfolio.py)

| 工具 | 描述 | 参数 |
|------|------|------|
| `mean_variance_optimize` | 均值方差优化 | returns_str, cov_matrix_str, target_return |
| `efficient_frontier` | 有效前沿计算 | returns_str, cov_matrix_str, n_points |
| `black_litterman` | Black-Litterman模型 | P_str, Q_str, cov_matrix_str, market_weights_str, tau |
| `hrp_clustering` | 层次风险平价 | returns_str |
| `portfolio_metrics` | 组合绩效指标 | weights_str, returns_str, cov_matrix_str, rf |

### 风险度量工具 (risk.py)

| 工具 | 描述 | 参数 |
|------|------|------|
| `var_historical` | 历史模拟法VaR | returns_str, confidence |
| `var_parametric` | 参数法VaR | mean, std, confidence |
| `var_monte_carlo` | 蒙特卡洛VaR | returns_str, confidence, n_sims |
| `cvar` | 条件VaR (CVaR/ES) | returns_str, confidence |
| `max_drawdown` | 最大回撤 | prices_str |

### 固定收益工具 (fixed_income.py)

| 工具 | 描述 | 参数 |
|------|------|------|
| `bond_price` | 债券定价 | face, coupon_rate, ytm, maturity, freq |
| `bond_duration` | 久期计算 | face, coupon_rate, ytm, maturity, freq |
| `bond_convexity` | 凸性计算 | face, coupon_rate, ytm, maturity, freq |
| `nelson_siegel` | NS收益率曲线拟合 | beta0, beta1, beta2, tau, maturities_str |

### 工具函数 (utils.py)

| 工具 | 描述 | 参数 |
|------|------|------|
| `parse_returns` | 解析收益序列 | input_str |
| `parse_matrix` | 解析矩阵 | input_str |
| `format_result` | 格式化数值 | value, precision |
| `validate_inputs` | 输入验证 | args_str |

## 输入格式说明

- **收益序列**：逗号分隔的数值字符串，如 `"0.01,0.02,-0.01"`
- **矩阵**：分号分隔行、逗号分隔列，如 `"0.04,0.01;0.01,0.09"`
- **权重向量**：逗号分隔的数值，如 `"0.3,0.4,0.3"`

## 技术栈

- **MCP SDK**: `mcp.server.fastmcp.FastMCP`
- **数值计算**: NumPy, SciPy
- **数据处理**: Pandas
- **优化求解**: scipy.optimize.minimize (SLSQP)
- **层次聚类**: scipy.cluster.hierarchy
- **统计分布**: scipy.stats.norm

## 理论参考

- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
- Cox, J., Ross, S. & Rubinstein, M. (1979). Option Pricing: A Simplified Approach.
- Markowitz, H. (1952). Portfolio Selection.
- Black, F. & Litterman, R. (1991). Global Portfolio Optimization.
- Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample.
- Nelson, C. & Siegel, A. (1987). Parsimonious Modeling of Yield Curves.
- Jorion, P. (2007). Value at Risk: The New Benchmark for Managing Financial Risk.
- Rockafellar, R. & Uryasev, S. (2002). Conditional Value-at-Risk.

## 项目结构

```
mcp-quant-engine/
├── server.py                # MCP Server 入口
├── mcp_quant_engine/
│   ├── __init__.py           # FastMCP 实例创建
│   ├── pricing.py            # 期权定价工具（6个）
│   ├── portfolio.py          # 组合优化工具（5个）
│   ├── risk.py               # 风险度量工具（5个）
│   ├── fixed_income.py       # 固定收益工具（4个）
│   └── utils.py              # 数学工具函数（4个）
├── README.md
├── SKILL.md
└── requirements.txt
```

## 许可证

MIT License
