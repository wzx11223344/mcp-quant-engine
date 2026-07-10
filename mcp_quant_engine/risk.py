"""risk.py - 风险度量工具模块

提供历史模拟VaR、参数法VaR、蒙特卡洛VaR、
条件VaR (CVaR/ES) 和最大回撤等风险度量工具。

参考：
    - Jorion (2007) - Value at Risk
    - Rockafellar & Uryasev (2002) - Conditional Value-at-Risk
"""
import numpy as np
from scipy.stats import norm

from mcp_quant_engine import mcp
from mcp_quant_engine.utils import _parse_returns, _validate_inputs


# ---------------------------------------------------------------------------
# 内部计算函数
# ---------------------------------------------------------------------------

def _var_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """历史模拟法 VaR（内部函数）。"""
    percentile = (1 - confidence) * 100
    var_value = -np.percentile(returns, percentile)
    return float(var_value)


def _var_parametric(mean: float, std: float, confidence: float = 0.95) -> float:
    """参数法 VaR（内部函数）。"""
    z = norm.ppf(1 - confidence)  # 左侧分位数（为负）
    var_value = -(mean + std * z)
    return float(var_value)


def _var_monte_carlo(returns: np.ndarray, confidence: float = 0.95,
                     n_sims: int = 100000) -> float:
    """蒙特卡洛 VaR（内部函数）。"""
    np.random.seed(42)
    mean = np.mean(returns)
    std = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
    simulated = np.random.normal(mean, std, n_sims)
    percentile = (1 - confidence) * 100
    var_value = -np.percentile(simulated, percentile)
    return float(var_value)


def _cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """条件 VaR / Expected Shortfall（内部函数）。"""
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= var_threshold]
    if len(tail_losses) == 0:
        return float(-var_threshold)
    cvar_value = -np.mean(tail_losses)
    return float(cvar_value)


def _max_drawdown(prices: np.ndarray) -> dict:
    """最大回撤（内部函数）。"""
    prices = np.asarray(prices, dtype=float)
    running_max = np.maximum.accumulate(prices)
    drawdowns = (prices - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    max_dd_idx = int(np.argmin(drawdowns))
    peak_idx = int(np.argmax(prices[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
    return {
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd * 100,
        "peak_index": peak_idx,
        "trough_index": max_dd_idx,
        "peak_value": float(prices[peak_idx]),
        "trough_value": float(prices[max_dd_idx]),
    }


# ---------------------------------------------------------------------------
# MCP 工具函数
# ---------------------------------------------------------------------------

@mcp.tool()
def var_historical(returns_str: str, confidence: float = 0.95) -> str:
    """历史模拟法 VaR 计算。

    使用历史收益率分位数直接估计 VaR，无需分布假设。

    Args:
        returns_str: 逗号分隔的收益率序列，如 "0.01,-0.02,0.03,-0.01,0.005"
        confidence: 置信水平（默认 0.95）

    Returns:
        历史模拟 VaR 结果（markdown 格式）
    """
    try:
        returns = _parse_returns(returns_str)
        _validate_inputs(confidence)
        if not 0 < confidence < 1:
            raise ValueError("confidence 必须在 (0, 1) 之间")
        var_val = _var_historical(returns, confidence)
        threshold = np.percentile(returns, (1 - confidence) * 100)
        return (
            f"## 历史模拟法 VaR\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 数据数量 | {len(returns)} |\n"
            f"| 置信水平 | {confidence:.2%} |\n\n"
            f"**统计信息**:\n\n"
            f"| 指标 | 值 |\n"
            f"|------|-----|\n"
            f"| 均值 | {np.mean(returns):.6f} |\n"
            f"| 标准差 | {np.std(returns, ddof=1) if len(returns) > 1 else 0:.6f} |\n"
            f"| {(1 - confidence) * 100:.1f}% 分位数 | {threshold:.6f} |\n\n"
            f"**VaR 结果**: **{var_val:.6f}** ({var_val * 100:.2f}%)\n\n"
            f"含义：在 {confidence:.0%} 置信水平下，最大预期损失不超过 {var_val * 100:.2f}%\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def var_parametric(mean: float, std: float, confidence: float = 0.95) -> str:
    """参数法 VaR 计算（方差-协方差法）。

    假设收益率服从正态分布，使用均值和标准差计算 VaR。

    公式: VaR = -(mean + std * Z_alpha)，其中 Z_alpha 为标准正态分布的下分位数。

    Args:
        mean: 收益率均值
        std: 收益率标准差
        confidence: 置信水平（默认 0.95）

    Returns:
        参数法 VaR 结果（markdown 格式）
    """
    try:
        _validate_inputs(mean, std, confidence)
        if not 0 < confidence < 1:
            raise ValueError("confidence 必须在 (0, 1) 之间")
        if std < 0:
            raise ValueError("std 不能为负数")
        var_val = _var_parametric(mean, std, confidence)
        z_score = norm.ppf(1 - confidence)
        return (
            f"## 参数法 VaR (方差-协方差法)\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 均值 (mean) | {mean:.6f} |\n"
            f"| 标准差 (std) | {std:.6f} |\n"
            f"| 置信水平 | {confidence:.2%} |\n\n"
            f"**计算过程**:\n\n"
            f"| 步骤 | 值 |\n"
            f"|------|-----|\n"
            f"| Z 分位数 (Z_alpha) | {z_score:.6f} |\n"
            f"| mean + std * Z_alpha | {mean + std * z_score:.6f} |\n\n"
            f"**VaR 结果**: **{var_val:.6f}** ({var_val * 100:.2f}%)\n\n"
            f"含义：在 {confidence:.0%} 置信水平下，最大预期损失不超过 {var_val * 100:.2f}%\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def var_monte_carlo(returns_str: str, confidence: float = 0.95,
                     n_sims: int = 100000) -> str:
    """蒙特卡洛模拟 VaR 计算。

    从历史收益率拟合正态分布，模拟大量场景后取分位数。

    Args:
        returns_str: 逗号分隔的收益率序列
        confidence: 置信水平（默认 0.95）
        n_sims: 模拟次数（默认 100000）

    Returns:
        蒙特卡洛 VaR 结果（markdown 格式）
    """
    try:
        returns = _parse_returns(returns_str)
        _validate_inputs(confidence)
        if not 0 < confidence < 1:
            raise ValueError("confidence 必须在 (0, 1) 之间")
        if n_sims <= 0:
            raise ValueError("n_sims 必须为正整数")
        var_mc = _var_monte_carlo(returns, confidence, n_sims)
        # 对比历史VaR
        var_hist = _var_historical(returns, confidence)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
        var_param = _var_parametric(mean, std, confidence)
        return (
            f"## 蒙特卡洛 VaR\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 数据数量 | {len(returns)} |\n"
            f"| 拟合均值 | {mean:.6f} |\n"
            f"| 拟合标准差 | {std:.6f} |\n"
            f"| 模拟次数 | {n_sims} |\n"
            f"| 置信水平 | {confidence:.2%} |\n\n"
            f"**VaR 对比**:\n\n"
            f"| 方法 | VaR 值 | 占比 |\n"
            f"|------|--------|------|\n"
            f"| 蒙特卡洛 | {var_mc:.6f} | {var_mc * 100:.2f}% |\n"
            f"| 历史模拟 (对比) | {var_hist:.6f} | {var_hist * 100:.2f}% |\n"
            f"| 参数法 (对比) | {var_param:.6f} | {var_param * 100:.2f}% |\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def cvar(returns_str: str, confidence: float = 0.95) -> str:
    """条件 VaR (CVaR / Expected Shortfall) 计算。

    CVaR 是超过 VaR 的尾部损失的期望值，衡量极端风险。

    公式: CVaR = -E[returns | returns <= VaR_threshold]

    Args:
        returns_str: 逗号分隔的收益率序列
        confidence: 置信水平（默认 0.95）

    Returns:
        CVaR 结果（markdown 格式）
    """
    try:
        returns = _parse_returns(returns_str)
        _validate_inputs(confidence)
        if not 0 < confidence < 1:
            raise ValueError("confidence 必须在 (0, 1) 之间")
        cvar_val = _cvar(returns, confidence)
        var_val = _var_historical(returns, confidence)
        threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_count = np.sum(returns <= threshold)
        tail_mean = np.mean(returns[returns <= threshold]) if tail_count > 0 else threshold
        return (
            f"## 条件 VaR (CVaR / Expected Shortfall)\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 数据数量 | {len(returns)} |\n"
            f"| 置信水平 | {confidence:.2%} |\n\n"
            f"**计算过程**:\n\n"
            f"| 步骤 | 值 |\n"
            f"|------|-----|\n"
            f"| VaR 阈值分位数 | {threshold:.6f} |\n"
            f"| 尾部数据数量 | {tail_count} |\n"
            f"| 尾部均值 | {tail_mean:.6f} |\n\n"
            f"**风险度量结果**:\n\n"
            f"| 指标 | 值 | 占比 |\n"
            f"|------|-----|------|\n"
            f"| VaR | {var_val:.6f} | {var_val * 100:.2f}% |\n"
            f"| CVaR | {cvar_val:.6f} | {cvar_val * 100:.2f}% |\n"
            f"| CVaR/VaR 比 | {cvar_val / var_val if var_val > 0 else 0:.4f} | - |\n\n"
            f"含义：在最坏 {1 - confidence:.0%} 的场景中，平均损失为 {cvar_val * 100:.2f}%\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def max_drawdown(prices_str: str) -> str:
    """最大回撤计算。

    计算价格序列从历史峰值到谷值的最大跌幅，
    衡量投资组合可能面临的最严重亏损。

    Args:
        prices_str: 逗号分隔的价格序列，如 "100,102,98,95,97,103"

    Returns:
        最大回撤结果（markdown 格式）
    """
    try:
        prices = _parse_returns(prices_str)
        if len(prices) < 2:
            raise ValueError("至少需要 2 个价格数据")
        result = _max_drawdown(prices)
        return (
            f"## 最大回撤计算\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 数据数量 | {len(prices)} |\n"
            f"| 起始价格 | {prices[0]:.4f} |\n"
            f"| 结束价格 | {prices[-1]:.4f} |\n\n"
            f"**回撤详情**:\n\n"
            f"| 指标 | 值 |\n"
            f"|------|-----|\n"
            f"| 最大回撤 | {result['max_drawdown']:.6f} |\n"
            f"| 最大回撤百分比 | {result['max_drawdown_pct']:.2f}% |\n"
            f"| 峰值位置 (索引) | {result['peak_index']} |\n"
            f"| 谷值位置 (索引) | {result['trough_index']} |\n"
            f"| 峰值价格 | {result['peak_value']:.4f} |\n"
            f"| 谷值价格 | {result['trough_value']:.4f} |\n\n"
            f"**总收益**: {(prices[-1] / prices[0] - 1) * 100:.2f}%\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"
