"""pricing.py - 期权定价工具模块

提供 Black-Scholes 模型、隐含波动率、蒙特卡洛模拟、
Greeks 计算、二叉树模型等期权定价工具。

公式参考：
    - Black-Scholes (1973)
    - Cox-Ross-Rubinstein (CRR) 二叉树模型
    - Newton-Raphson 隐含波动率迭代法
"""
import numpy as np
from scipy.stats import norm

from mcp_quant_engine import mcp
from mcp_quant_engine.utils import _validate_inputs


# ---------------------------------------------------------------------------
# 内部计算函数
# ---------------------------------------------------------------------------

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """计算 Black-Scholes 模型中的 d1 和 d2。"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes 看涨期权定价（内部函数）。"""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes 看跌期权定价（内部函数）。"""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """计算 Black-Scholes 模型的 Vega（内部函数）。"""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return float(S * np.sqrt(T) * norm.pdf(d1))


def _implied_vol(price: float, S: float, K: float, T: float,
                 r: float, option_type: str = "call") -> float:
    """牛顿迭代法计算隐含波动率（内部函数）。"""
    if option_type.lower() not in ("call", "put"):
        raise ValueError("option_type 必须为 'call' 或 'put'")

    sigma = 0.3  # 初始猜测
    for _ in range(200):
        if option_type.lower() == "call":
            bs_price = _bs_call(S, K, T, r, sigma)
        else:
            bs_price = _bs_put(S, K, T, r, sigma)
        vega = _bs_vega(S, K, T, r, sigma)
        if vega < 1e-12:
            break
        diff = bs_price - price
        if abs(diff) < 1e-8:
            break
        sigma = sigma - diff / vega
        sigma = max(sigma, 1e-6)  # 防止负值
    return float(sigma)


def _monte_carlo_option(S: float, K: float, T: float, r: float, sigma: float,
                        n_sims: int, option_type: str = "call") -> float:
    """蒙特卡洛模拟期权定价（内部函数）。"""
    np.random.seed(42)  # 可复现性
    Z = np.random.standard_normal(n_sims)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    if option_type.lower() == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    return float(np.exp(-r * T) * np.mean(payoff))


def _option_greeks(S: float, K: float, T: float, r: float, sigma: float,
                   option_type: str = "call") -> dict:
    """计算全部 Greeks（内部函数）。"""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    # Delta
    if option_type.lower() == "call":
        delta = float(norm.cdf(d1))
    else:
        delta = float(norm.cdf(d1) - 1)

    # Gamma（看涨看跌相同）
    gamma = float(pdf_d1 / (S * sigma * sqrt_T))

    # Vega（看涨看跌相同）
    vega = float(S * pdf_d1 * sqrt_T)

    # Theta（年化）
    if option_type.lower() == "call":
        theta = float(-S * pdf_d1 * sigma / (2 * sqrt_T)
                      - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        theta = float(-S * pdf_d1 * sigma / (2 * sqrt_T)
                      + r * K * np.exp(-r * T) * norm.cdf(-d2))

    # Rho
    if option_type.lower() == "call":
        rho = float(K * T * np.exp(-r * T) * norm.cdf(d2))
    else:
        rho = float(-K * T * np.exp(-r * T) * norm.cdf(-d2))

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def _binomial_tree(S: float, K: float, T: float, r: float, sigma: float,
                   steps: int, option_type: str = "call") -> float:
    """CRR 二叉树美式期权定价（内部函数）。"""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # 初始化到期日资产价格
    prices = np.zeros(steps + 1)
    for j in range(steps + 1):
        prices[j] = S * (u ** j) * (d ** (steps - j))

    # 初始化到期日期权价值
    if option_type.lower() == "call":
        values = np.maximum(prices - K, 0.0)
    else:
        values = np.maximum(K - prices, 0.0)

    # 向后归纳（含提前行权判断）
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            prices[j] = S * (u ** j) * (d ** (i - j))
            values[j] = discount * (p * values[j + 1] + (1 - p) * values[j])
            # 美式期权提前行权
            if option_type.lower() == "call":
                values[j] = max(values[j], prices[j] - K)
            else:
                values[j] = max(values[j], K - prices[j])

    return float(values[0])


# ---------------------------------------------------------------------------
# MCP 工具函数
# ---------------------------------------------------------------------------

@mcp.tool()
def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> str:
    """Black-Scholes 看涨期权定价。

    使用 Black-Scholes (1973) 模型计算欧式看涨期权价格。

    公式: C = S * N(d1) - K * e^(-rT) * N(d2)
    其中 d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))

    Args:
        S: 标的资产当前价格
        K: 期权行权价
        T: 到期时间（年，如 0.5 表示 6 个月）
        r: 无风险利率（如 0.05 表示 5%）
        sigma: 标的资产波动率（如 0.2 表示 20%）

    Returns:
        看涨期权定价结果（markdown 格式）
    """
    try:
        _validate_inputs(S, K, T, r, sigma)
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("S, K, T, sigma 必须为正数")
        price = _bs_call(S, K, T, r, sigma)
        d1, d2 = _d1_d2(S, K, T, r, sigma)
        return (
            f"## Black-Scholes 看涨期权定价\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 标的资产价格 (S) | {S:.4f} |\n"
            f"| 行权价 (K) | {K:.4f} |\n"
            f"| 到期时间 (T) | {T:.4f} 年 |\n"
            f"| 无风险利率 (r) | {r:.4f} |\n"
            f"| 波动率 (sigma) | {sigma:.4f} |\n\n"
            f"**中间变量**:\n\n"
            f"| 变量 | 值 |\n"
            f"|------|-----|\n"
            f"| d1 | {d1:.6f} |\n"
            f"| d2 | {d2:.6f} |\n"
            f"| N(d1) | {norm.cdf(d1):.6f} |\n"
            f"| N(d2) | {norm.cdf(d2):.6f} |\n\n"
            f"**定价结果**: **{price:.6f}**\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> str:
    """Black-Scholes 看跌期权定价。

    使用 Black-Scholes (1973) 模型计算欧式看跌期权价格。

    公式: P = K * e^(-rT) * N(-d2) - S * N(-d1)

    Args:
        S: 标的资产当前价格
        K: 期权行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 标的资产波动率

    Returns:
        看跌期权定价结果（markdown 格式）
    """
    try:
        _validate_inputs(S, K, T, r, sigma)
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("S, K, T, sigma 必须为正数")
        price = _bs_put(S, K, T, r, sigma)
        d1, d2 = _d1_d2(S, K, T, r, sigma)
        return (
            f"## Black-Scholes 看跌期权定价\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 标的资产价格 (S) | {S:.4f} |\n"
            f"| 行权价 (K) | {K:.4f} |\n"
            f"| 到期时间 (T) | {T:.4f} 年 |\n"
            f"| 无风险利率 (r) | {r:.4f} |\n"
            f"| 波动率 (sigma) | {sigma:.4f} |\n\n"
            f"**中间变量**:\n\n"
            f"| 变量 | 值 |\n"
            f"|------|-----|\n"
            f"| d1 | {d1:.6f} |\n"
            f"| d2 | {d2:.6f} |\n"
            f"| N(-d1) | {norm.cdf(-d1):.6f} |\n"
            f"| N(-d2) | {norm.cdf(-d2):.6f} |\n\n"
            f"**定价结果**: **{price:.6f}**\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def implied_vol(price: float, S: float, K: float, T: float,
                r: float, option_type: str = "call") -> str:
    """隐含波动率计算（牛顿迭代法）。

    给定期权市场价格，反解出隐含的波动率。
    使用 Newton-Raphson 迭代法，以 BS 模型 Vega 为导数。

    Args:
        price: 期权市场价格
        S: 标的资产价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        option_type: 期权类型，"call" 或 "put"

    Returns:
        隐含波动率计算结果（markdown 格式）
    """
    try:
        _validate_inputs(price, S, K, T, r)
        if S <= 0 or K <= 0 or T <= 0 or price <= 0:
            raise ValueError("S, K, T, price 必须为正数")
        iv = _implied_vol(price, S, K, T, r, option_type)
        # 验证：用隐含波动率重新定价
        if option_type.lower() == "call":
            check_price = _bs_call(S, K, T, r, iv)
        else:
            check_price = _bs_put(S, K, T, r, iv)
        return (
            f"## 隐含波动率计算结果\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 期权价格 | {price:.4f} |\n"
            f"| 标的资产价格 (S) | {S:.4f} |\n"
            f"| 行权价 (K) | {K:.4f} |\n"
            f"| 到期时间 (T) | {T:.4f} 年 |\n"
            f"| 无风险利率 (r) | {r:.4f} |\n"
            f"| 期权类型 | {option_type} |\n\n"
            f"**隐含波动率 (IV)**: **{iv:.6f}** ({iv * 100:.2f}%)\n\n"
            f"**验证**: 用 IV={iv:.6f} 重新定价得到 {check_price:.6f} "
            f"(市场价格 {price:.6f})\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def monte_carlo_option(S: float, K: float, T: float, r: float, sigma: float,
                       n_sims: int = 100000, option_type: str = "call") -> str:
    """蒙特卡洛模拟期权定价。

    通过模拟标的资产价格路径，计算期权期望支付的贴现值。
    使用几何布朗运动模型 S_T = S * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z)。

    Args:
        S: 标的资产当前价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        n_sims: 模拟次数（默认 100000）
        option_type: "call" 或 "put"

    Returns:
        蒙特卡洛定价结果（markdown 格式）
    """
    try:
        _validate_inputs(S, K, T, r, sigma)
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("S, K, T, sigma 必须为正数")
        if n_sims <= 0:
            raise ValueError("n_sims 必须为正整数")
        mc_price = _monte_carlo_option(S, K, T, r, sigma, n_sims, option_type)
        # 与 BS 解析解对比
        if option_type.lower() == "call":
            bs_price = _bs_call(S, K, T, r, sigma)
        else:
            bs_price = _bs_put(S, K, T, r, sigma)
        error = abs(mc_price - bs_price)
        return (
            f"## 蒙特卡洛期权定价\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 标的资产价格 (S) | {S:.4f} |\n"
            f"| 行权价 (K) | {K:.4f} |\n"
            f"| 到期时间 (T) | {T:.4f} 年 |\n"
            f"| 无风险利率 (r) | {r:.4f} |\n"
            f"| 波动率 (sigma) | {sigma:.4f} |\n"
            f"| 模拟次数 | {n_sims} |\n"
            f"| 期权类型 | {option_type} |\n\n"
            f"**定价结果**:\n\n"
            f"| 方法 | 价格 |\n"
            f"|------|------|\n"
            f"| 蒙特卡洛 | {mc_price:.6f} |\n"
            f"| Black-Scholes (对比) | {bs_price:.6f} |\n"
            f"| 误差 | {error:.6f} |\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def option_greeks(S: float, K: float, T: float, r: float, sigma: float,
                   option_type: str = "call") -> str:
    """期权 Greeks 全部计算。

    计算 Black-Scholes 模型下的全部希腊字母：
    Delta, Gamma, Vega, Theta, Rho。

    Args:
        S: 标的资产价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: "call" 或 "put"

    Returns:
        全部 Greeks 计算结果（markdown 格式）
    """
    try:
        _validate_inputs(S, K, T, r, sigma)
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("S, K, T, sigma 必须为正数")
        greeks = _option_greeks(S, K, T, r, sigma, option_type)
        d1, d2 = _d1_d2(S, K, T, r, sigma)
        return (
            f"## 期权 Greeks 计算结果\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 标的资产价格 (S) | {S:.4f} |\n"
            f"| 行权价 (K) | {K:.4f} |\n"
            f"| 到期时间 (T) | {T:.4f} 年 |\n"
            f"| 无风险利率 (r) | {r:.4f} |\n"
            f"| 波动率 (sigma) | {sigma:.4f} |\n"
            f"| 期权类型 | {option_type} |\n\n"
            f"**中间变量**: d1={d1:.6f}, d2={d2:.6f}\n\n"
            f"**Greeks 结果**:\n\n"
            f"| Greek | 值 | 含义 |\n"
            f"|-------|-----|------|\n"
            f"| Delta | {greeks['delta']:.6f} | 标的资产价格变化1单位时期权价格变化 |\n"
            f"| Gamma | {greeks['gamma']:.6f} | 标的资产价格变化1单位时Delta变化 |\n"
            f"| Vega  | {greeks['vega']:.6f} | 波动率变化1%(绝对)时期权价格变化 |\n"
            f"| Theta | {greeks['theta']:.6f} | 每日时间衰减（年化/365） |\n"
            f"| Rho   | {greeks['rho']:.6f} | 利率变化1%时期权价格变化 |\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def binomial_tree(S: float, K: float, T: float, r: float, sigma: float,
                  steps: int = 100, option_type: str = "call") -> str:
    """二叉树期权定价（CRR模型，支持美式期权）。

    使用 Cox-Ross-Rubinstein (CRR) 二叉树模型定价，
    支持美式期权的提前行权。

    Args:
        S: 标的资产价格
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        steps: 二叉树步数（默认 100）
        option_type: "call" 或 "put"

    Returns:
        二叉树定价结果（markdown 格式）
    """
    try:
        _validate_inputs(S, K, T, r, sigma)
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("S, K, T, sigma 必须为正数")
        if steps <= 0:
            raise ValueError("steps 必须为正整数")
        price = _binomial_tree(S, K, T, r, sigma, steps, option_type)
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp(r * dt) - d) / (u - d)
        # 与欧式 BS 对比
        if option_type.lower() == "call":
            bs_price = _bs_call(S, K, T, r, sigma)
        else:
            bs_price = _bs_put(S, K, T, r, sigma)
        return (
            f"## 二叉树期权定价 (CRR模型)\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 标的资产价格 (S) | {S:.4f} |\n"
            f"| 行权价 (K) | {K:.4f} |\n"
            f"| 到期时间 (T) | {T:.4f} 年 |\n"
            f"| 无风险利率 (r) | {r:.4f} |\n"
            f"| 波动率 (sigma) | {sigma:.4f} |\n"
            f"| 步数 (steps) | {steps} |\n"
            f"| 期权类型 | {option_type} |\n\n"
            f"**模型参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| dt | {dt:.6f} |\n"
            f"| u (上涨因子) | {u:.6f} |\n"
            f"| d (下跌因子) | {d:.6f} |\n"
            f"| p (上涨概率) | {p:.6f} |\n\n"
            f"**定价结果**:\n\n"
            f"| 方法 | 价格 |\n"
            f"|------|------|\n"
            f"| 二叉树 (美式) | {price:.6f} |\n"
            f"| Black-Scholes (欧式对比) | {bs_price:.6f} |\n"
            f"| 差异 (美式溢价) | {price - bs_price:.6f} |\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"
