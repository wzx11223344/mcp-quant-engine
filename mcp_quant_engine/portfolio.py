"""portfolio.py - 组合优化工具模块

提供均值方差优化、有效前沿、Black-Litterman 模型、
层次风险平价 (HRP) 和组合绩效指标等组合优化工具。

参考：
    - Markowitz (1952) - Portfolio Selection
    - Black & Litterman (1991) - Global Portfolio Optimization
    - Lopez de Prado (2016) - Building Diversified Portfolios that
      Outperform Out-of-Sample (Hierarchical Risk Parity)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from mcp_quant_engine import mcp
from mcp_quant_engine.utils import (
    _parse_returns,
    _parse_matrix,
    _validate_inputs,
    _format_result,
)


# ---------------------------------------------------------------------------
# 内部计算函数
# ---------------------------------------------------------------------------

def _mean_variance_optimize(mean_returns: np.ndarray, cov: np.ndarray,
                            target_return: float) -> np.ndarray:
    """均值方差优化（内部函数）。

    在给定目标收益率下最小化组合方差，使用 SLSQP 方法求解。

    Args:
        mean_returns: 各资产期望收益向量 (N,)
        cov: 协方差矩阵 (N x N)
        target_return: 目标收益率

    Returns:
        最优权重向量 (N,)
    """
    n = len(mean_returns)

    def portfolio_variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: float(w @ mean_returns) - target_return},
    ]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        portfolio_variance, x0, method="SLSQP",
        constraints=constraints, bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"优化失败: {result.message}")
    return result.x


def _efficient_frontier(mean_returns: np.ndarray, cov: np.ndarray,
                        n_points: int = 20) -> list:
    """有效前沿计算（内部函数）。

    Returns:
        列表，每项为 (return, volatility, weights) 三元组
    """
    min_ret = float(np.min(mean_returns))
    max_ret = float(np.max(mean_returns))
    target_returns = np.linspace(min_ret, max_ret, n_points)
    results = []
    for tr in target_returns:
        try:
            w = _mean_variance_optimize(mean_returns, cov, tr)
            port_ret = float(w @ mean_returns)
            port_vol = float(np.sqrt(w @ cov @ w))
            results.append((port_ret, port_vol, w))
        except Exception:
            continue
    return results


def _black_litterman(P: np.ndarray, Q: np.ndarray, cov: np.ndarray,
                     market_weights: np.ndarray,
                     tau: float = 0.05) -> dict:
    """Black-Litterman 模型（内部函数）。

    将市场隐含均衡收益与投资者观点融合，得到后验期望收益。

    Args:
        P: 观点矩阵 (K x N)，每行表示一个观点涉及的资产权重
        Q: 观点收益向量 (K,)
        cov: 协方差矩阵 (N x N)
        market_weights: 市场组合权重 (N,)
        tau: 缩放参数（默认 0.05）

    Returns:
        包含先验收益、后验收益和后验协方差的字典
    """
    # 风险厌恶系数 delta（典型值 2.5）
    delta = 2.5

    # 先验：隐含均衡收益 pi = delta * cov * w_market
    pi = delta * (cov @ market_weights)

    # 观点不确定性矩阵 Omega = diag(diag(P @ (tau*cov) @ P^T))
    tau_cov = tau * cov
    omega = np.diag(np.diag(P @ tau_cov @ P.T))

    # 后验收益: mu = [(tau*cov)^{-1} + P^T * Omega^{-1} * P]^{-1}
    #               * [(tau*cov)^{-1} * pi + P^T * Omega^{-1} * Q]
    inv_tau_cov = np.linalg.inv(tau_cov)
    inv_omega = np.linalg.inv(omega)
    A = inv_tau_cov + P.T @ inv_omega @ P
    b = inv_tau_cov @ pi + P.T @ inv_omega @ Q
    posterior_returns = np.linalg.solve(A, b)
    posterior_cov = np.linalg.inv(A)

    return {
        "prior_returns": pi,
        "posterior_returns": posterior_returns,
        "posterior_cov": posterior_cov,
        "delta": delta,
    }


def _cluster_variance(cov: np.ndarray, indices: list) -> float:
    """计算子聚类的方差（使用逆方差加权）（内部函数）。"""
    sub_cov = cov[np.ix_(indices, indices)]
    diag_vals = np.diag(sub_cov)
    diag_vals = np.where(diag_vals > 1e-12, diag_vals, 1e-12)
    inv_var = 1.0 / diag_vals
    w = inv_var / np.sum(inv_var)
    return float(w @ sub_cov @ w)


def _hrp_clustering(returns: np.ndarray) -> dict:
    """层次风险平价 (HRP)（内部函数）。

    通过层次聚类和递归二分法分配权重，无需矩阵求逆。

    Args:
        returns: 收益率时间序列矩阵 (T x N)，每列为一个资产

    Returns:
        包含权重、聚类顺序和链接矩阵的字典
    """
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    n_assets = returns.shape[1]

    # 协方差和相关系数矩阵
    cov = np.cov(returns, rowvar=False)
    if n_assets == 1:
        return {
            "weights": np.array([1.0]),
            "order": np.array([0]),
            "linkage": None,
            "cov": cov,
            "corr": np.array([[1.0]]),
        }
    corr = np.corrcoef(returns, rowvar=False)

    # 距离矩阵: d = sqrt(0.5 * (1 - corr))
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)

    # 转换为压缩距离向量并聚类
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")

    # 准对角化：获取叶子节点顺序
    order = leaves_list(link)

    # 递归二分法分配权重
    weights = np.ones(n_assets)
    clusters = [list(order)]

    while len(clusters) < n_assets:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                new_clusters.append(cluster)
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            left_var = _cluster_variance(cov, left)
            right_var = _cluster_variance(cov, right)

            if left_var + right_var < 1e-12:
                alpha = 0.5
            else:
                alpha = 1.0 - left_var / (left_var + right_var)

            weights[left] *= alpha
            weights[right] *= (1.0 - alpha)

            new_clusters.append(left)
            new_clusters.append(right)
        clusters = new_clusters

    return {
        "weights": weights,
        "order": order,
        "linkage": link,
        "cov": cov,
        "corr": corr,
    }


def _portfolio_metrics(weights: np.ndarray, mean_returns: np.ndarray,
                       cov: np.ndarray, rf: float = 0.02) -> dict:
    """组合绩效指标计算（内部函数）。

    Args:
        weights: 组合权重 (N,)
        mean_returns: 各资产期望收益 (N,)
        cov: 协方差矩阵 (N x N)
        rf: 无风险利率（年化）

    Returns:
        包含收益率、波动率、Sharpe 比率和最大回撤的字典
    """
    port_return = float(weights @ mean_returns)
    port_var = float(weights @ cov @ weights)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    sharpe = (port_return - rf) / port_vol if port_vol > 1e-12 else 0.0

    # 蒙特卡洛模拟最大回撤
    np.random.seed(42)
    n_days = 252
    daily_mean = port_return / n_days
    daily_vol = port_vol / np.sqrt(n_days)
    sim_returns = np.random.normal(daily_mean, daily_vol, n_days)
    cum_values = np.cumprod(1.0 + sim_returns)
    running_max = np.maximum.accumulate(cum_values)
    drawdowns = (cum_values - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    # 组合中各资产贡献
    asset_contrib = weights * mean_returns

    return {
        "return": port_return,
        "volatility": port_vol,
        "variance": port_var,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "asset_contrib": asset_contrib,
        "rf": rf,
    }


# ---------------------------------------------------------------------------
# MCP 工具函数
# ---------------------------------------------------------------------------

@mcp.tool()
def mean_variance_optimize(returns_str: str, cov_matrix_str: str,
                           target_return: float) -> str:
    """均值方差优化 (Markowitz)。

    在给定目标收益率约束下最小化组合方差，
    使用 scipy SLSQP 方法求解，支持做多约束。

    Args:
        returns_str: 逗号分隔的期望收益向量，如 "0.05,0.03,0.04"
        cov_matrix_str: 分号分隔行、逗号分隔列的协方差矩阵，
            如 "0.04,0.01;0.01,0.09"
        target_return: 目标收益率（如 0.04 表示 4%）

    Returns:
        最优权重和组合统计信息（markdown 格式）
    """
    try:
        mean_returns = _parse_returns(returns_str)
        cov = _parse_matrix(cov_matrix_str)
        _validate_inputs(target_return)

        n = len(mean_returns)
        if cov.shape[0] != n or cov.shape[1] != n:
            raise ValueError(
                f"协方差矩阵维度 ({cov.shape[0]}x{cov.shape[1]}) "
                f"与收益向量长度 ({n}) 不匹配"
            )

        weights = _mean_variance_optimize(mean_returns, cov, target_return)
        port_ret = float(weights @ mean_returns)
        port_vol = float(np.sqrt(weights @ cov @ weights))
        sharpe = (port_ret - 0.02) / port_vol if port_vol > 0 else 0

        asset_list = "\n".join([
            f"| 资产{i + 1} | {weights[i]:.6f} | "
            f"{weights[i] * 100:.2f}% | {mean_returns[i]:.6f} |"
            for i in range(n)
        ])

        return (
            f"## 均值方差优化结果\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 资产数量 | {n} |\n"
            f"| 目标收益率 | {target_return:.6f} ({target_return * 100:.2f}%) |\n\n"
            f"**最优权重**:\n\n"
            f"| 资产 | 权重 | 占比 | 期望收益 |\n"
            f"|------|------|------|----------|\n"
            f"{asset_list}\n\n"
            f"**组合统计**:\n\n"
            f"| 指标 | 值 |\n"
            f"|------|-----|\n"
            f"| 组合收益率 | {port_ret:.6f} ({port_ret * 100:.2f}%) |\n"
            f"| 组合波动率 | {port_vol:.6f} ({port_vol * 100:.2f}%) |\n"
            f"| Sharpe 比率 (rf=2%) | {sharpe:.6f} |\n"
            f"| 权重和 | {np.sum(weights):.6f} |\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def efficient_frontier(returns_str: str, cov_matrix_str: str,
                        n_points: int = 20) -> str:
    """有效前沿计算。

    在最小收益到最大收益之间生成多个目标收益率，
    对每个目标求解均值方差优化，得到有效前沿曲线。

    Args:
        returns_str: 逗号分隔的期望收益向量
        cov_matrix_str: 分号分隔行、逗号分隔列的协方差矩阵
        n_points: 前沿上的点数（默认 20）

    Returns:
        有效前沿数据表（markdown 格式）
    """
    try:
        mean_returns = _parse_returns(returns_str)
        cov = _parse_matrix(cov_matrix_str)
        _validate_inputs(n_points)
        if n_points <= 1:
            raise ValueError("n_points 必须大于 1")

        n = len(mean_returns)
        if cov.shape[0] != n or cov.shape[1] != n:
            raise ValueError("协方差矩阵维度与收益向量不匹配")

        frontier = _efficient_frontier(mean_returns, cov, n_points)

        if not frontier:
            raise RuntimeError("有效前沿计算失败，无有效结果")

        rows = "\n".join([
            f"| {i + 1} | {r:.6f} | {v:.6f} | "
            f"{(_format_result(w, 4))} |"
            for i, (r, v, w) in enumerate(frontier)
        ])

        # 找到最小方差组合
        min_vol_idx = int(np.argmin([v for _, v, _ in frontier]))
        min_vol = frontier[min_vol_idx]

        # 找到最大 Sharpe 组合（假设 rf=2%）
        rf = 0.02
        sharpe_list = [(r - rf) / v if v > 0 else 0 for r, v, _ in frontier]
        max_sharpe_idx = int(np.argmax(sharpe_list))
        max_sharpe = frontier[max_sharpe_idx]

        return (
            f"## 有效前沿计算结果\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 资产数量 | {n} |\n"
            f"| 前沿点数 | {n_points} |\n"
            f"| 有效结果数 | {len(frontier)} |\n\n"
            f"**有效前沿数据**:\n\n"
            f"| 序号 | 收益率 | 波动率 | 最优权重 |\n"
            f"|------|--------|--------|----------|\n"
            f"{rows}\n\n"
            f"**关键组合**:\n\n"
            f"| 组合类型 | 收益率 | 波动率 | Sharpe (rf=2%) |\n"
            f"|----------|--------|--------|----------------|\n"
            f"| 最小方差组合 | {min_vol[0]:.6f} | {min_vol[1]:.6f} | "
            f"{(min_vol[0] - rf) / min_vol[1] if min_vol[1] > 0 else 0:.6f} |\n"
            f"| 最大Sharpe组合 | {max_sharpe[0]:.6f} | {max_sharpe[1]:.6f} | "
            f"{sharpe_list[max_sharpe_idx]:.6f} |\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def black_litterman(P_str: str, Q_str: str, cov_matrix_str: str,
                    market_weights_str: str, tau: float = 0.05) -> str:
    """Black-Litterman 模型。

    将市场隐含均衡收益（先验）与投资者主观观点融合，
    得到后验期望收益，用于改进组合优化。

    模型公式：
        先验: pi = delta * cov * w_market
        后验: mu = [(tau*cov)^{-1} + P^T * Omega^{-1} * P]^{-1}
                    * [(tau*cov)^{-1} * pi + P^T * Omega^{-1} * Q]

    Args:
        P_str: 观点矩阵字符串（分号分隔行，逗号分隔列），
            如 "1,0,0;0,1,-1" 表示两个观点
        Q_str: 逗号分隔的观点收益向量，如 "0.05,0.02"
        cov_matrix_str: 协方差矩阵字符串
        market_weights_str: 逗号分隔的市场组合权重
        tau: 缩放参数（默认 0.05）

    Returns:
        BL 模型后验收益和对比信息（markdown 格式）
    """
    try:
        P = _parse_matrix(P_str)
        Q = _parse_returns(Q_str)
        cov = _parse_matrix(cov_matrix_str)
        market_weights = _parse_returns(market_weights_str)
        _validate_inputs(tau)

        n = len(market_weights)
        k = len(Q)

        if cov.shape[0] != n or cov.shape[1] != n:
            raise ValueError("协方差矩阵维度与市场权重不匹配")
        if P.shape[1] != n:
            raise ValueError("观点矩阵列数必须等于资产数量")
        if P.shape[0] != k:
            raise ValueError("观点矩阵行数必须等于观点数量")
        if tau <= 0:
            raise ValueError("tau 必须为正数")

        result = _black_litterman(P, Q, cov, market_weights, tau)

        prior = result["prior_returns"]
        posterior = result["posterior_returns"]
        delta = result["delta"]

        # 后验收益下的最优组合（最小方差，无目标约束）
        post_weights = _mean_variance_optimize(
            posterior, cov, float(np.mean(posterior))
        )

        comparison_rows = "\n".join([
            f"| 资产{i + 1} | {market_weights[i]:.6f} | "
            f"{prior[i]:.6f} | {posterior[i]:.6f} | "
            f"{posterior[i] - prior[i]:.6f} | {post_weights[i]:.6f} |"
            for i in range(n)
        ])

        return (
            f"## Black-Litterman 模型结果\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 资产数量 | {n} |\n"
            f"| 观点数量 | {k} |\n"
            f"| 风险厌恶系数 (delta) | {delta:.4f} |\n"
            f"| 缩放参数 (tau) | {tau:.6f} |\n\n"
            f"**观点矩阵 P**:\n```\n{P}\n```\n\n"
            f"**观点收益 Q**: {_format_result(Q, 6)}\n\n"
            f"**收益对比**:\n\n"
            f"| 资产 | 市场权重 | 先验收益(隐含) | 后验收益 | "
            f"调整量 | 后验最优权重 |\n"
            f"|------|----------|----------------|----------|"
            f"--------|----------------|\n"
            f"{comparison_rows}\n\n"
            f"**关键统计**:\n\n"
            f"| 指标 | 值 |\n"
            f"|------|-----|\n"
            f"| 先验收益均值 | {np.mean(prior):.6f} |\n"
            f"| 后验收益均值 | {np.mean(posterior):.6f} |\n"
            f"| 先验收益标准差 | {np.std(prior):.6f} |\n"
            f"| 后验收益标准差 | {np.std(posterior):.6f} |\n"
            f"| 平均调整量 | {np.mean(posterior - prior):.6f} |\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def hrp_clustering(returns_str: str) -> str:
    """层次风险平价 (HRP)。

    通过层次聚类和递归二分法分配组合权重，
    无需矩阵求逆，适合高维或病态协方差矩阵。

    算法步骤：
        1. 计算相关系数矩阵和距离矩阵
        2. 使用层次聚类得到树状结构
        3. 准对角化协方差矩阵
        4. 递归二分法按逆方差比例分配权重

    Args:
        returns_str: 收益率矩阵字符串（分号分隔行，逗号分隔列），
            每行是一个时间点，每列是一个资产，
            如 "0.01,0.02,0.015;0.02,-0.01,0.03;-0.01,0.03,0.01"

    Returns:
        HRP 权重和聚类信息（markdown 格式）
    """
    try:
        returns = _parse_matrix(returns_str)
        if returns.ndim != 2 or returns.shape[0] < 2:
            raise ValueError("收益率矩阵至少需要 2 行 x 2 列")

        n_assets = returns.shape[1]
        result = _hrp_clustering(returns)
        weights = result["weights"]
        order = result["order"]
        corr = result["corr"]

        # 聚类顺序
        order_str = " -> ".join([f"资产{int(o) + 1}" for o in order])

        weight_rows = "\n".join([
            f"| 资产{int(order[i]) + 1} | {i + 1} | "
            f"{weights[int(order[i])]:.6f} | "
            f"{weights[int(order[i])] * 100:.2f}% |"
            for i in range(n_assets)
        ])

        # 等权组合对比
        equal_weights = np.ones(n_assets) / n_assets
        cov = result["cov"]
        hrp_vol = float(np.sqrt(weights @ cov @ weights))
        eq_vol = float(np.sqrt(equal_weights @ cov @ equal_weights))

        return (
            f"## 层次风险平价 (HRP) 结果\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 资产数量 | {n_assets} |\n"
            f"| 时间序列长度 | {returns.shape[0]} |\n\n"
            f"**聚类顺序**: {order_str}\n\n"
            f"**HRP 权重分配**:\n\n"
            f"| 资产 | 聚类顺序 | 权重 | 占比 |\n"
            f"|------|----------|------|------|\n"
            f"{weight_rows}\n\n"
            f"**组合对比**:\n\n"
            f"| 方法 | 组合波动率 | 分散化收益 |\n"
            f"|------|-----------|------------|\n"
            f"| HRP | {hrp_vol:.6f} | - |\n"
            f"| 等权重 | {eq_vol:.6f} | "
            f"{(eq_vol - hrp_vol) / eq_vol * 100:.2f}% |\n\n"
            f"**相关系数矩阵**:\n```\n{np.array2string(corr, precision=4, separator=', ')}\n```\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def portfolio_metrics(weights_str: str, returns_str: str,
                       cov_matrix_str: str, rf: float = 0.02) -> str:
    """组合绩效指标计算。

    计算组合的收益率、波动率、Sharpe 比率和最大回撤等指标。

    Args:
        weights_str: 逗号分隔的权重向量，如 "0.3,0.4,0.3"
        returns_str: 逗号分隔的期望收益向量，如 "0.05,0.03,0.04"
        cov_matrix_str: 分号分隔行、逗号分隔列的协方差矩阵
        rf: 无风险利率（年化，默认 0.02 即 2%）

    Returns:
        组合绩效指标结果（markdown 格式）
    """
    try:
        weights = _parse_returns(weights_str)
        mean_returns = _parse_returns(returns_str)
        cov = _parse_matrix(cov_matrix_str)
        _validate_inputs(rf)

        n = len(weights)
        if len(mean_returns) != n:
            raise ValueError("权重向量与收益向量长度不匹配")
        if cov.shape[0] != n or cov.shape[1] != n:
            raise ValueError("协方差矩阵维度与权重不匹配")

        metrics = _portfolio_metrics(weights, mean_returns, cov, rf)

        asset_rows = "\n".join([
            f"| 资产{i + 1} | {weights[i]:.6f} | "
            f"{weights[i] * 100:.2f}% | {mean_returns[i]:.6f} | "
            f"{metrics['asset_contrib'][i]:.6f} |"
            for i in range(n)
        ])

        # 风险贡献
        marginal_risk = cov @ weights
        risk_contrib = weights * marginal_risk
        total_risk = float(np.sum(risk_contrib))
        risk_contrib_pct = risk_contrib / total_risk * 100 if total_risk > 0 else risk_contrib

        risk_rows = "\n".join([
            f"| 资产{i + 1} | {risk_contrib[i]:.6f} | "
            f"{risk_contrib_pct[i]:.2f}% |"
            for i in range(n)
        ])

        return (
            f"## 组合绩效指标\n\n"
            f"**输入参数**:\n\n"
            f"| 参数 | 值 |\n"
            f"|------|-----|\n"
            f"| 资产数量 | {n} |\n"
            f"| 无风险利率 | {rf:.4f} ({rf * 100:.2f}%) |\n\n"
            f"**权重与收益分解**:\n\n"
            f"| 资产 | 权重 | 占比 | 期望收益 | 收益贡献 |\n"
            f"|------|------|------|----------|----------|\n"
            f"{asset_rows}\n\n"
            f"**风险贡献分解**:\n\n"
            f"| 资产 | 风险贡献 | 占比 |\n"
            f"|------|----------|------|\n"
            f"{risk_rows}\n\n"
            f"**组合绩效指标**:\n\n"
            f"| 指标 | 值 | 含义 |\n"
            f"|------|-----|------|\n"
            f"| 组合收益率 | {metrics['return']:.6f} ({metrics['return'] * 100:.2f}%) | 年化期望收益 |\n"
            f"| 组合波动率 | {metrics['volatility']:.6f} ({metrics['volatility'] * 100:.2f}%) | 年化标准差 |\n"
            f"| 组合方差 | {metrics['variance']:.6f} | 年化方差 |\n"
            f"| Sharpe 比率 | {metrics['sharpe']:.6f} | 单位风险超额收益 |\n"
            f"| 最大回撤 (模拟) | {metrics['max_drawdown']:.6f} ({metrics['max_drawdown'] * 100:.2f}%) | 蒙特卡洛模拟252天 |\n"
        )
    except Exception as e:
        return f"## 计算错误\n\n**错误信息**: {str(e)}"
