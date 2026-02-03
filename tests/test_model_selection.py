"""Тесты выбора лучшей модели."""
import pytest
from data.loader import load_stock_data
from models.model_selector import ModelSelector


def test_model_selector_select_best():
    """Проверяет выбор лучшей модели."""
    df = load_stock_data("AAPL", period_days=730)
    series = df["Close"]
    
    selector = ModelSelector(forecast_days=30)
    best_model, name, metrics, forecast = selector.select_best(series)
    
    assert best_model is not None, "Лучшая модель не должна быть None"
    assert name in ['RandomForest', 'Prophet', 'LSTM'], "Имя модели должно быть валидным"
    assert metrics is not None, "Метрики не должны быть None"
    assert forecast is not None, "Прогноз не должен быть None"


def test_model_selector_metrics():
    """Проверяет, что метрики содержат нужные поля."""
    df = load_stock_data("AAPL", period_days=730)
    series = df["Close"]
    
    selector = ModelSelector(forecast_days=30)
    _, _, metrics, _ = selector.select_best(series)
    
    assert 'mape' in metrics, "Метрики должны содержать MAPE"
    assert 'rmse' in metrics, "Метрики должны содержать RMSE"
    assert metrics['mape'] >= 0, "MAPE должен быть неотрицательным"


def test_model_selector_forecast_length():
    """Проверяет длину прогноза."""
    df = load_stock_data("AAPL", period_days=730)
    series = df["Close"]
    
    forecast_days = 30
    selector = ModelSelector(forecast_days=forecast_days)
    _, _, _, forecast = selector.select_best(series)
    
    assert len(forecast) == forecast_days, f"Прогноз должен содержать {forecast_days} дней"


def test_model_selector_different_tickers():
    """Проверяет выбор модели для разных тикеров."""
    for ticker in ['AAPL', 'MSFT']:
        df = load_stock_data(ticker, period_days=365)
        series = df["Close"]
        
        selector = ModelSelector(forecast_days=30)
        best_model, name, metrics, forecast = selector.select_best(series)
        
        assert name is not None, f"Модель для {ticker} должна быть выбрана"
        assert metrics['mape'] >= 0, f"MAPE для {ticker} должен быть валиден"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
