"""fixed_income.py - 固定收益工具模块

提供债券定价、久期计算、凸性计算和
Nelson-Siegel 收益率曲线拟合等固定收益工具。

参考：
    - Fabozzi - Fixed Income Mathematics
    - Nelson & Siegel (1987) - Yield Curve Fitting
"""
import numpy as np

from mcp_quant_engine import mcp
from mcp_quant_engine.utils import _parse_returns, _validate_inputs


# ---------------------------------------------------------------------------
# 内部计算函数
# ---------------------------------------------------------------------------

def _bond_cashflows(face: float, coupon_rate: float, ytm: float,
                    maturity: float, freq: int = 2) -> tuple:
    """计算债券现金流和贴现因子（内部函数）。

    Returns:
        (periods, cashflows, discount_factors) 三元组
    """
    n_periods = int(maturity * freq)
    coupon = face * coupon_rate / freq
    periods = np.arange(1, n_periods + 1, dtype=float)
    cashflows = np.full(n_periods, coupon, dtype=float)
    cashflows[-1] += face  # 最后一期加上面值
    discount_rate = 1 + ytm / freq
    discount_factors = discount_rate ** periods
    return periods, cashflows, discount_factors


def _bond_price(face: float, coupon_rate: float, ytm: float,
                maturity: float, freq: int = 2) -> float:
    """债券定价（内部函数）。"""
    periods, cashflows, df = _bond_cashflows(
        face, coupon_rate, ytm, maturity, freq
    )
    pv = np.sum(cashflows / df)
    return float(pv)


def _bond_duration(face: float, coupon_rate: float, ytm: float,
                   maturity: float, freq: int = 2) -> dict:
    """久期计算（内部函数），返回 Macaulay 和 Modified 久期。"""
    periods, cashflows, df = _bond_cashflows(
        face, coupon_rate, ytm, maturity, freq
    )
    pv_array = cashflows / df
    price = np.sum(pv_array)
    # Macaulay 久期（以年为单位）
    macaulay = np.sum(periods / freq * pv_array) / price
    # Modified 久期
    modified = macaulay / (1 + ytm / freq)
    return {
        "macaulay": float(macaulay),
        "modified": float(modified),
        "price": float(price),
    }


def _bond_convexity(face: float, coupon_rate: float, ytm: float,
                     maturity: float, freq: int = 2) -> float:
    """凸性计算（内部函数）。"""
    periods, cashflows, df = _bond_cashflows(
        face, coupon_rate, ytm, maturity, freq
    )
    pv_array = cashflows / df
    price = np.sum(pv_array)
    t_years = periods / freq
    # 凸性 = sum[t*(t+1/f) * CF / (1+y/f)^(t+2)] / (P * (1+y/f)^2)
    # 简化公式: sum[t*(t+1) * CF / (1+y/f)^t] / (P * (1+y/f)^2 * f^2)
    convexity = np.sum(t_years * (t_years + 1.0 / freq) * pv_array) / (
        price * (1 + ytm / freq) ** 2
    )
    return float(convexity)


def _nelson_siegel(beta0: float, beta1: float, beta2: float,
                   tau: float, maturities: np.ndarray) -> np.ndarray:
    """Nelson-Siegel 收益率曲线拟合（内部函数）。"""
    maturities = np.asarray(maturities, dtype=float)
    tau = max(tau, 1e-10)  # 防止除零
    x = maturities / tau
    # 避免对零求除
    factor1 = np.where(
        maturities > 1e-10,
        (1 - np.exp(-x)) / x,
        1.0  # lim(x->0) (1-e^{-x})/x = 1
    )
    factor2 = np.where(
        maturities > 1e-10,
        factor1 - np.exp(-x),
        0.0  # lim(x->0) [(1-e^{-x})/x - e^{-x}] = 0
    )
    yields = beta0 + beta1 * factor1 + beta2 * factor2
    return yields


# ---------------------------------------------------------------------------
# MCP 工具函数
# ---------------------------------------------------------------------------

@mcp.tool()
def bond_price(face: float, coupon_rate: float, ytm: float,
              maturity: float, freq: int = 2) -> str:
    """债券定价。

    使用现金流贴现法（DCF）计算债券价格。

    公式: P = sum[CF_t / (1 + y/f)^t] + F / (1 + y/f)^N

    Args:
        face: 债券面值
        coupon_rate: 年化票面利率（如 0.05 表示 5%）
        ytm: 年化到期收益率
        maturity: 到期年限
        freq: 每年付息频率（默认 2，即半年付息）

    Returns:
        债券定价结果（markdown 格式）
    """
    try:
        _validate_inputs(face, coupon_rate, ytm, maturity)
        if face <= 0 or maturity <= 0 or freq <= 0:
            raise ValueError("face, maturity, freq 必须为正数")
        if coupon_rate < 0 or ytm < 0:
            raise ValueError("coupon_rate, ytm 不能为负数")
        price = _bond_price(face, coupon_rate, ytm, maturity, freq)
        n_periods = int(maturity * freq)
        coupon = face * coupon_rate / freq
        # 票面价格 vs 市场价格
        if price > face:
            status = "溢价交易 (Premium)"
        elif price < face:
            status = "折价交易 (Discount)"
        else:
            status = "平价交易 (Par)"
        return (
            f"## 债券定价结果\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 面值 | {face:.2f} |\n"
            f"| 票面利率 | {coupon_rate:.4f} ({coupon_rate * 100:.2f}%) |\n"
            f"| 到期收益率 (YTM) | {ytm:.4f} ({ytm * 100:.2f}%) |\n"
            f"| 到期年限 | {maturity:.2f} 年 |\n"
            f"| 付息频率 | 每年 {freq} 次 |\n\n"
            f"**现金流信息**:\n\n"
            f"| 指标 | 值 |\n"
            f"|------|-----|\n"
            f"| 总期数 | {n_periods} |\n"
            f"| 每期票息 | {coupon:.4f} |\n"
            f"| 最后期现金流 | {coupon + face:.4f} |\n\n"
            f"**定价结果**: **{price:.6f}**\n\n"
            f"**交易状态**: {status}\n"
            f"（价格 {'>' if price > face else '<' if price < face else '='} 面值 {face:.2f}）\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def bond_duration(face: float, coupon_rate: float, ytm: float,
                  maturity: float, freq: int = 2) -> str:
    """债券久期计算。

    计算 Macaulay 久期和 Modified 久期。

    Macaulay 久期: D = sum[t * PV_t] / P
    Modified 久期: D_mod = D / (1 + y/f)

    Args:
        face: 债券面值
        coupon_rate: 年化票面利率
        ytm: 年化到期收益率
        maturity: 到期年限
        freq: 每年付息频率（默认 2）

    Returns:
        久期计算结果（markdown 格式）
    """
    try:
        _validate_inputs(face, coupon_rate, ytm, maturity)
        if face <= 0 or maturity <= 0 or freq <= 0:
            raise ValueError("face, maturity, freq 必须为正数")
        if coupon_rate < 0 or ytm < 0:
            raise ValueError("coupon_rate, ytm 不能为负数")
        result = _bond_duration(face, coupon_rate, ytm, maturity, freq)
        return (
            f"## 债券久期计算结果\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 面值 | {face:.2f} |\n"
            f"| 票面利率 | {coupon_rate:.4f} ({coupon_rate * 100:.2f}%) |\n"
            f"| 到期收益率 (YTM) | {ytm:.4f} ({ytm * 100:.2f}%) |\n"
            f"| 到期年限 | {maturity:.2f} 年 |\n"
            f"| 付息频率 | 每年 {freq} 次 |\n\n"
            f"**计算结果**:\n\n"
            f"| 指标 | 值 | 含义 |\n"
            f"|------|-----|------|\n"
            f"| 债券价格 | {result['price']:.6f} | 当前市场价格 |\n"
            f"| Macaulay 久期 | {result['macaulay']:.6f} 年 | 现金流加权平均时间 |\n"
            f"| Modified 久期 | {result['modified']:.6f} | 利率敏感度（价格弹性） |\n\n"
            f"**利率敏感度**: YTM 每变动 1%，价格约变动 {result['modified']:.4f}%\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def bond_convexity(face: float, coupon_rate: float, ytm: float,
                    maturity: float, freq: int = 2) -> str:
    """债券凸性计算。

    凸性衡量价格-收益率曲线的弯曲程度，
    是久期的二阶修正。

    公式: Convexity = sum[t*(t+1/f) * CF / (1+y/f)^t] / (P * (1+y/f)^2)

    Args:
        face: 债券面值
        coupon_rate: 年化票面利率
        ytm: 年化到期收益率
        maturity: 到期年限
        freq: 每年付息频率（默认 2）

    Returns:
        凸性计算结果（markdown 格式）
    """
    try:
        _validate_inputs(face, coupon_rate, ytm, maturity)
        if face <= 0 or maturity <= 0 or freq <= 0:
            raise ValueError("face, maturity, freq 必须为正数")
        if coupon_rate < 0 or ytm < 0:
            raise ValueError("coupon_rate, ytm 不能为负数")
        convexity = _bond_convexity(face, coupon_rate, ytm, maturity, freq)
        duration_result = _bond_duration(face, coupon_rate, ytm, maturity, freq)
        mod_duration = duration_result["modified"]
        price = duration_result["price"]
        # 价格变动的二阶近似
        delta_y = 0.01  # 1% 利率变动
        price_change_pct = (
            -mod_duration * delta_y
            + 0.5 * convexity * delta_y ** 2
        ) * 100
        return (
            f"## 债券凸性计算结果\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 面值 | {face:.2f} |\n"
            f"| 票面利率 | {coupon_rate:.4f} ({coupon_rate * 100:.2f}%) |\n"
            f"| 到期收益率 (YTM) | {ytm:.4f} ({ytm * 100:.2f}%) |\n"
            f"| 到期年限 | {maturity:.2f} 年 |\n"
            f"| 付息频率 | 每年 {freq} 次 |\n\n"
            f"**计算结果**:\n\n"
            f"| 指标 | 值 |\n"
            f"|------|-----|\n"
            f"| 债券价格 | {price:.6f} |\n"
            f"| Modified 久期 | {mod_duration:.6f} |\n"
            f"| 凸性 | {convexity:.6f} |\n\n"
            f"**利率敏感度分析** (YTM 变动 1%):\n\n"
            f"| 近似方法 | 价格变动 |\n"
            f"|----------|----------|\n"
            f"| 一阶近似 (仅久期) | {-mod_duration * 0.01 * 100:.4f}% |\n"
            f"| 二阶近似 (久期+凸性) | {price_change_pct:.4f}% |\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def nelson_siegel(beta0: float, beta1: float, beta2: float,
                  tau: float, maturities_str: str) -> str:
    """Nelson-Siegel 收益率曲线拟合。

    使用 Nelson-Siegel (1987) 模型生成收益率曲线。

    公式: y(tau) = beta0 + beta1 * (1-e^{-t/tau})/(t/tau)
                    + beta2 * [(1-e^{-t/tau})/(t/tau) - e^{-t/tau}]

    参数含义:
        - beta0: 长期利率水平
        - beta1: 短期利率偏离（斜率因子）
        - beta2: 中期利率偏离（曲率因子）
        - tau: 衰减参数（转折点）

    Args:
        beta0: 长期利率水平参数
        beta1: 短期斜率参数
        beta2: 中期曲率参数
        tau: 衰减参数
        maturities_str: 逗号分隔的期限序列（年），如 "0.5,1,2,5,10,30"

    Returns:
        NS 收益率曲线结果（markdown 格式）
    """
    try:
        _validate_inputs(beta0, beta1, beta2, tau)
        if tau <= 0:
            raise ValueError("tau 必须为正数")
        maturities = _parse_returns(maturities_str)
        if len(maturities) == 0:
            raise ValueError("至少需要一个期限")
        yields = _nelson_siegel(beta0, beta1, beta2, tau, maturities)
        table_rows = "\n".join([
            f"| {maturities[i]:.2f} | {yields[i]:.6f} | {yields[i] * 100:.4f}% |"
            for i in range(len(maturities))
        ])
        return (
            f"## Nelson-Siegel 收益率曲线\n\n"
            f"**模型参数**:\n\n"
            f"| 参数 | 值 | 含义 |\n"
            f"|------|-----|------|\n"
            f"| beta0 | {beta0:.6f} | 长期利率水平 |\n"
            f"| beta1 | {beta1:.6f} | 短期斜率因子 |\n"
            f"| beta2 | {beta2:.6f} | 中期曲率因子 |\n"
            f"| tau | {tau:.6f} | 衰减参数 |\n\n"
            f"**收益率曲线**:\n\n"
            f"| 期限 (年) | 收益率 | 年化利率 |\n"
            f"|-----------|--------|----------|\n"
            f"{table_rows}\n\n"
            f"**曲线特征**:\n\n"
            f"- 短期利率 (1年): {yields[0] if maturities[0] == 1 else 'N/A':.4f}\n"
            f"- 长期利率 (最大期限): {yields[-1]:.4f}\n"
            f"- 期限利差: {yields[-1] - yields[0]:.6f}\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"
