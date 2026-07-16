"""Tests for mcp-quant-engine risk and pricing functions."""

import numpy as np
import pytest
from mcp_quant_engine.risk import _var_historical, _var_parametric, _max_drawdown, _cvar


class TestVarHistorical:
    """Tests for historical VaR calculation."""

    def test_var_95_percent(self):
        """Test 95% VaR with known data."""
        returns = np.array([-0.02, -0.01, 0.01, 0.02, 0.03, -0.005, 0.015, 0.01])
        var = _var_historical(returns, confidence=0.95)
        assert var > 0
        assert isinstance(var, float)

    def test_var_99_percent(self):
        """Test 99% VaR (should be higher than 95%)."""
        returns = np.array([-0.02, -0.01, 0.01, 0.02, 0.03, -0.005, 0.015, 0.01, -0.03, -0.04])
        var95 = _var_historical(returns, confidence=0.95)
        var99 = _var_historical(returns, confidence=0.99)
        assert var99 >= var95

    def test_var_all_positive(self):
        """Test VaR with all positive returns."""
        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])
        var = _var_historical(returns, confidence=0.95)
        assert isinstance(var, float)

    def test_var_single_value(self):
        """Test VaR with a single return value."""
        returns = np.array([0.05])
        var = _var_historical(returns, confidence=0.95)
        assert isinstance(var, float)


class TestVarParametric:
    """Tests for parametric VaR calculation."""

    def test_var_parametric_basic(self):
        """Test parametric VaR with known mean and std."""
        var = _var_parametric(mean=0.001, std=0.02, confidence=0.95)
        assert var > 0
        assert isinstance(var, float)

    def test_var_parametric_zero_mean(self):
        """Test parametric VaR with zero mean."""
        var = _var_parametric(mean=0.0, std=0.01, confidence=0.95)
        assert var > 0

    def test_var_parametric_high_confidence(self):
        """Test that higher confidence gives higher VaR."""
        var95 = _var_parametric(mean=0.001, std=0.02, confidence=0.95)
        var99 = _var_parametric(mean=0.001, std=0.02, confidence=0.99)
        assert var99 > var95

    def test_var_parametric_negative_mean(self):
        """Test parametric VaR with negative mean returns."""
        var = _var_parametric(mean=-0.001, std=0.02, confidence=0.95)
        assert var > 0


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_max_drawdown_increasing(self):
        """Test max drawdown with monotonically increasing prices."""
        prices = np.array([100, 110, 120, 130, 140, 150])
        result = _max_drawdown(prices)
        assert result["max_drawdown"] <= 0
        assert result["max_drawdown"] == 0.0

    def test_max_drawdown_with_drop(self):
        """Test max drawdown with a known drop."""
        prices = np.array([100, 120, 80, 90, 110])
        result = _max_drawdown(prices)
        assert result["max_drawdown"] < 0
        assert isinstance(result["max_drawdown_pct"], float)

    def test_max_drawdown_result_keys(self):
        """Test that result dict contains expected keys."""
        prices = np.array([100, 110, 90, 105])
        result = _max_drawdown(prices)
        assert "max_drawdown" in result
        assert "max_drawdown_pct" in result
        assert "peak_index" in result
        assert "trough_index" in result

    def test_max_drawdown_single_value(self):
        """Test max drawdown with a single price."""
        prices = np.array([100])
        result = _max_drawdown(prices)
        assert result["max_drawdown"] == 0.0


class TestCVaR:
    """Tests for Conditional VaR (Expected Shortfall)."""

    def test_cvar_basic(self):
        """Test CVaR calculation."""
        returns = np.array([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03])
        cvar_val = _cvar(returns, confidence=0.95)
        assert isinstance(cvar_val, float)
        assert cvar_val >= 0

    def test_cvar_vs_var(self):
        """Test that CVaR is greater than or equal to VaR."""
        returns = np.array([-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
        var = _var_historical(returns, confidence=0.95)
        cvar_val = _cvar(returns, confidence=0.95)
        assert cvar_val >= var
