"""utils.py - 数学工具函数模块

提供输入解析、结果格式化和输入验证等通用工具函数。
包含内部辅助函数（_前缀）供其他模块调用，以及MCP工具供AI客户端直接使用。
"""
import numpy as np
from typing import Any, Union

from mcp_quant_engine import mcp


# ---------------------------------------------------------------------------
# 内部辅助函数（供其他模块调用）
# ---------------------------------------------------------------------------

def _parse_returns(input_data: Union[str, list, np.ndarray]) -> np.ndarray:
    """解析收益序列为 numpy 一维数组（内部使用）。

    支持以下输入格式：
        - 逗号分隔的字符串: "0.01,0.02,-0.01"
        - 列表: [0.01, 0.02, -0.01]
        - numpy 数组（直接返回）

    Args:
        input_data: 输入数据

    Returns:
        numpy 一维浮点数组
    """
    if isinstance(input_data, np.ndarray):
        return input_data.astype(float)
    if isinstance(input_data, (list, tuple)):
        return np.array(input_data, dtype=float)
    # 字符串解析
    values = [float(x.strip()) for x in str(input_data).split(",")]
    return np.array(values, dtype=float)


def _parse_matrix(input_data: Union[str, list, np.ndarray]) -> np.ndarray:
    """解析矩阵为 numpy 二维数组（内部使用）。

    支持以下输入格式：
        - 分号分隔行、逗号分隔列的字符串: "0.01,0.002;0.002,0.01"
        - 嵌套列表: [[0.01, 0.002], [0.002, 0.01]]
        - numpy 二维数组（直接返回）

    Args:
        input_data: 输入数据

    Returns:
        numpy 二维浮点数组
    """
    if isinstance(input_data, np.ndarray):
        return input_data.astype(float)
    if isinstance(input_data, (list, tuple)):
        # 检查是否为嵌套列表
        if len(input_data) > 0 and isinstance(input_data[0], (list, tuple)):
            return np.array(input_data, dtype=float)
        else:
            return np.array([input_data], dtype=float)
    # 字符串解析：分号分隔行，逗号分隔列
    rows = str(input_data).strip().split(";")
    matrix = []
    for row in rows:
        row = row.strip()
        if row:
            values = [float(x.strip()) for x in row.split(",")]
            matrix.append(values)
    if not matrix:
        return np.array([[]], dtype=float)
    return np.array(matrix, dtype=float)


def _format_result(value: Any, precision: int = 4) -> str:
    """格式化数值为字符串（内部使用）。

    Args:
        value: 要格式化的值（支持标量、数组、列表）
        precision: 小数位数

    Returns:
        格式化后的字符串
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return f"{float(value):.{precision}f}"
        elif value.ndim == 1:
            return "[" + ", ".join([f"{x:.{precision}f}" for x in value]) + "]"
        else:
            return np.array2string(value, precision=precision, separator=", ")
    if isinstance(value, (list, tuple)):
        if len(value) > 0 and isinstance(value[0], (list, tuple)):
            # 嵌套列表
            lines = []
            for row in value:
                lines.append("[" + ", ".join([f"{float(x):.{precision}f}" for x in row]) + "]")
            return "\n".join(lines)
        return "[" + ", ".join([f"{float(x):.{precision}f}" for x in value]) + "]"
    if isinstance(value, (int, float, np.floating, np.integer)):
        return f"{float(value):.{precision}f}"
    return str(value)


def _validate_inputs(*args) -> bool:
    """验证输入参数是否有效（内部使用）。

    检查参数是否为 None、NaN 或 Inf。

    Args:
        *args: 要验证的数值参数

    Returns:
        True 如果所有参数有效

    Raises:
        ValueError: 如果任何参数无效
    """
    for i, arg in enumerate(args):
        if arg is None:
            raise ValueError(f"第 {i + 1} 个参数不能为 None")
        if isinstance(arg, (int, float, np.floating, np.integer)):
            if np.isnan(float(arg)):
                raise ValueError(f"第 {i + 1} 个参数不能为 NaN")
            if np.isinf(float(arg)):
                raise ValueError(f"第 {i + 1} 个参数不能为 Inf")
    return True


# ---------------------------------------------------------------------------
# MCP 工具函数
# ---------------------------------------------------------------------------

@mcp.tool()
def parse_returns(input_str: str) -> str:
    """解析收益序列字符串为数值数组。

    将逗号分隔的数值字符串解析为 numpy 数组，并返回基本统计信息。

    输入格式：逗号分隔的数值，如 "0.01,0.02,-0.01,0.03"

    Args:
        input_str: 逗号分隔的收益序列字符串

    Returns:
        解析结果及统计信息（markdown 格式）
    """
    try:
        result = _parse_returns(input_str)
        return (
            f"## 收益序列解析结果\n\n"
            f"**输入**: `{input_str}`\n\n"
            f"**解析数组**:\n```\n{result.tolist()}\n```\n\n"
            f"**统计信息**:\n\n"
            f"| 指标 | 值 |\n"
            f"|------|-----|\n"
            f"| 数据数量 | {len(result)} |\n"
            f"| 均值 | {np.mean(result):.6f} |\n"
            f"| 标准差 | {np.std(result, ddof=1) if len(result) > 1 else 0:.6f} |\n"
            f"| 最小值 | {np.min(result):.6f} |\n"
            f"| 最大值 | {np.max(result):.6f} |\n"
            f"| 中位数 | {np.median(result):.6f} |\n"
        )
    except Exception as e:
        return f"## 解析错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def parse_matrix(input_str: str) -> str:
    """解析矩阵字符串为二维数值数组。

    将分号分隔行、逗号分隔列的字符串解析为 numpy 二维数组。

    输入格式：分号分隔行，逗号分隔列，如 "0.01,0.002;0.002,0.01"

    Args:
        input_str: 分号分隔行、逗号分隔列的矩阵字符串

    Returns:
        解析结果（markdown 格式）
    """
    try:
        result = _parse_matrix(input_str)
        rows, cols = result.shape
        header = " | ".join([f"列{j}" for j in range(cols)])
        separator = "|".join(["---------"] * (cols + 1))
        body = "\n".join([
            f"| {i} | " + " | ".join([f"{result[i, j]:.6f}" for j in range(cols)]) + " |"
            for i in range(rows)
        ])
        det_str = f"{np.linalg.det(result):.6f}" if rows == cols else "N/A (非方阵)"
        return (
            f"## 矩阵解析结果\n\n"
            f"**输入**: `{input_str}`\n\n"
            f"**矩阵维度**: {rows} x {cols}\n\n"
            f"**解析矩阵**:\n\n"
            f"| 行\\\\列 | {header} |\n"
            f"|{separator}|\n"
            f"{body}\n\n"
            f"**矩阵行列式**: {det_str}\n"
        )
    except Exception as e:
        return f"## 解析错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def format_result(value: float, precision: int = 4) -> str:
    """格式化数值为指定精度的字符串。

    Args:
        value: 要格式化的数值
        precision: 小数位数（默认 4 位）

    Returns:
        格式化结果（markdown 格式）
    """
    try:
        formatted = _format_result(value, precision)
        return (
            f"## 数值格式化结果\n\n"
            f"**输入值**: {value}\n\n"
            f"**精度**: {precision} 位小数\n\n"
            f"**格式化结果**: `{formatted}`\n"
        )
    except Exception as e:
        return f"## 格式化错误\n\n**错误信息**: {str(e)}"


@mcp.tool()
def validate_inputs(args_str: str) -> str:
    """验证输入参数是否有效（非 None、非 NaN、非 Inf）。

    输入格式：逗号分隔的数值，如 "100,105,0.5,0.05,0.2"

    Args:
        args_str: 逗号分隔的数值字符串

    Returns:
        验证结果（markdown 格式）
    """
    try:
        values = _parse_returns(args_str)
        all_valid = True
        lines = []
        for i, v in enumerate(values):
            is_valid = not (np.isnan(v) or np.isinf(v))
            status = "有效" if is_valid else "无效"
            lines.append(f"| {i + 1} | {v} | {status} |")
            if not is_valid:
                all_valid = False
        status_text = "通过" if all_valid else "失败"
        return (
            f"## 输入验证结果\n\n"
            f"**状态**: {status_text}\n\n"
            f"**验证参数数量**: {len(values)}\n\n"
            f"**参数列表**:\n\n"
            f"| 序号 | 值 | 状态 |\n"
            f"|------|-----|------|\n"
            + "\n".join(lines)
            + "\n"
        )
    except Exception as e:
        return f"## 验证错误\n\n**错误信息**: {str(e)}"
