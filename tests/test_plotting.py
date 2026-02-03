"""Тесты построения графиков."""
import pytest
import io
from data.loader import load_stock_data
from models.model_selector import ModelSelector
from utils.plotting import plot_forecast


def test_plot_forecast_returns_buffer():
    """Проверяет, что функция возвращает буфер."""
    df = load_stock_data("AAPL", period_days=365)
    series = df["Close"]
    
    selector = ModelSelector(forecast_days=30)
    _, _, _, forecast = selector.select_best(series)
    
    buf = plot_forecast(series, forecast, "AAPL")
    
    assert buf is not None, "Буфер не должен быть None"
    assert isinstance(buf, io.BytesIO), "Должен быть BytesIO объект"


def test_plot_forecast_has_content():
    """Проверяет, что буфер содержит данные."""
    df = load_stock_data("AAPL", period_days=365)
    series = df["Close"]
    
    selector = ModelSelector(forecast_days=30)
    _, _, _, forecast = selector.select_best(series)
    
    buf = plot_forecast(series, forecast, "AAPL")
    
    assert buf.getbuffer().nbytes > 0, "Буфер должен содержать данные"


def test_plot_forecast_different_tickers():
    """Проверяет построение графиков для разных тикеров."""
    for ticker in ['AAPL', 'MSFT']:
        df = load_stock_data(ticker, period_days=365)
        series = df["Close"]
        
        selector = ModelSelector(forecast_days=30)
        _, _, _, forecast = selector.select_best(series)
        
        buf = plot_forecast(series, forecast, ticker)
        
        assert buf is not None, f"График для {ticker} должен быть создан"
        assert buf.getbuffer().nbytes > 0, f"График для {ticker} должен содержать данные"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
