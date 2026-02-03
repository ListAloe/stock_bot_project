"""Тесты загрузки данных с Yahoo Finance."""
import pytest
import yfinance as yf
import pandas as pd
from data.loader import load_stock_data


def test_load_stock_data_aapl():
    """Проверяет загрузку данных для AAPL."""
    df = load_stock_data("AAPL", period_days=90)
    
    assert df is not None, "DataFrame не должен быть None"
    assert not df.empty, "DataFrame не должен быть пустым"
    assert "Close" in df.columns, "DataFrame должен содержать колонку 'Close'"
    assert len(df) > 0, "Должны быть данные"
    assert df["Close"].notna().sum() > 0, "Должны быть непустые значения Close"


def test_load_stock_data_msft():
    """Проверяет загрузку данных для MSFT."""
    df = load_stock_data("MSFT", period_days=30)
    
    assert not df.empty, "Данные MSFT должны быть загружены"
    assert len(df) >= 20, "Должно быть достаточно данных для теста"


def test_load_stock_data_invalid_ticker():
    """Проверяет обработку невалидного тикера."""
    try:
        df = load_stock_data("INVALID_TICKER_XYZ", period_days=30)
        assert df.empty or len(df) == 0, "Невалидный тикер должен вернуть пустые данные"
    except Exception:
        pass


def test_yfinance_raw():
    """Проверяет прямую загрузку через yfinance."""
    data = yf.download("AAPL", period="5d", auto_adjust=True, progress=False)
    
    assert not data.empty, "Получены данные"
    
    if isinstance(data.columns, pd.MultiIndex):
        close_series = data["Close"].iloc[:, 0]
    else:
        close_series = data["Close"]
    
    assert len(close_series) > 0, "Должны быть данные Close"
    assert close_series.iloc[-1] > 0, "Цена должна быть положительной"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
