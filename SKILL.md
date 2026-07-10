---
name: quant-engine
description: 量化金融计算 MCP 服务器，提供期权定价、组合优化、风险度量和固定收益等 24 个专业金融数学计算工具
version: 1.0.0
author: quant-engine
tags:
  - finance
  - quantitative
  - options
  - portfolio
  - risk
  - fixed-income
  - mcp
---

# 量化引擎 MCP 服务器 (quant-engine)

## 概述

基于 Model Context Protocol (MCP) 的量化金融计算服务器，使用 FastMCP 框架，为 AI 客户端提供 24 个专业金融数学计算工具。

## 能力

### 期权定价 (6个工具)
- **black_scholes_call/put**: Black-Scholes (1973) 欧式期权定价
- **implied_vol**: 牛顿迭代法计算隐含波动率
- **monte_carlo_option**: 蒙特卡洛模拟期权定价（几何布朗运动）
- **option_greeks**: 全部 Greeks 计算（Delta/Gamma/Vega/Theta/Rho）
- **binomial_tree**: CRR 二叉树模型（支持美式期权提前行权）

### 组合优化 (5个工具)
- **mean_variance_optimize**: Markowitz 均值方差优化（SLSQP）
- **efficient_frontier**: 有效前沿计算
- **black_litterman**: Black-Litterman 模型（观点融合）
- **hrp_clustering**: 层次风险平价（无需矩阵求逆）
- **portfolio_metrics**: 组合绩效指标（Sharpe/波动率/最大回撤/风险贡献）

### 风险度量 (5个工具)
- **var_historical**: 历史模拟法 VaR
- **var_parametric**: 参数法 VaR（方差-协方差法）
- **var_monte_carlo**: 蒙特卡洛模拟 VaR
- **cvar**: 条件 VaR / Expected Shortfall
- **max_drawdown**: 最大回撤计算

### 固定收益 (4个工具)
- **bond_price**: 债券定价（现金流贴现法）
- **bond_duration**: Macaulay/Modified 久期
- **bond_convexity**: 凸性计算
- **nelson_siegel**: Nelson-Siegel 收益率曲线拟合

### 工具函数 (4个工具)
- **parse_returns**: 收益序列解析
- **parse_matrix**: 矩阵解析
- **format_result**: 数值格式化
- **validate_inputs**: 输入验证

## 输入格式

| 类型 | 格式 | 示例 |
|------|------|------|
| 收益序列 | 逗号分隔 | `"0.01,0.02,-0.01"` |
| 矩阵 | 分号分隔行，逗号分隔列 | `"0.04,0.01;0.01,0.09"` |
| 权重向量 | 逗号分隔 | `"0.3,0.4,0.3"` |

## 依赖

- mcp >= 1.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0

## 使用示例

### 启动服务器

```bash
python server.py
```

### MCP 客户端配置

```json
{
    "mcpServers": {
        "quant-engine": {
            "command": "python",
            "args": ["path/to/server.py"]
        }
    }
}
```

### 调用示例

```
# BS看涨期权定价
black_scholes_call(S=100, K=105, T=0.5, r=0.05, sigma=0.2)

# 均值方差优化
mean_variance_optimize(returns_str="0.05,0.03,0.04", cov_matrix_str="0.04,0.01,0.005;0.01,0.09,0.002;0.005,0.002,0.02", target_return=0.04)

# 历史VaR
var_historical(returns_str="0.01,-0.02,0.03,-0.01,0.005,0.02,-0.015,0.01", confidence=0.95)

# 债券定价
bond_price(face=1000, coupon_rate=0.05, ytm=0.04, maturity=10, freq=2)
```
