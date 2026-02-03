"""Тесты модели Prophet."""
import pytest
from data.loader import load_stock_data
from models.prophet import ProphetForecastModel


def test_prophet_fit():
    """Проверяет обучение Prophet модели."""
    df = load_stock_data("AAPL", period_days=365)
    series = df["Close"]
    
    model = ProphetForecastModel(forecast_horizon_days=30)
    model.fit(series)
    
    assert model is not None, "Модель должна быть инициализирована"
    assert hasattr(model, 'prophet_model'), "Модель должна иметь атрибут 'prophet_model'"


def test_prophet_evaluate():
    """Проверяет оценку качества Prophet модели."""
    df = load_stock_data("AAPL", period_days=365)
    series = df["Close"]
    
    model = ProphetForecastModel(forecast_horizon_days=30)
    model.fit(series)
    metrics = model.evaluate()
    
    assert metrics is not None, "Метрики не должны быть None"
    assert 'mape' in metrics, "Метрики должны содержать MAPE"
    assert 'rmse' in metrics, "Метрики должны содержать RMSE"
    assert metrics['mape'] >= 0, "MAPE должен быть неотрицательным"


def test_prophet_forecast():
    """Проверяет прогноз Prophet модели."""
    df = load_stock_data("AAPL", period_days=365)
    series = df["Close"]
    
    model = ProphetForecastModel(forecast_horizon_days=30)
    model.fit(series)
    forecast = model.forecast(steps=5)
    
    assert forecast is not None, "Прогноз не должен быть None"
    assert len(forecast) == 5, "Прогноз должен содержать 5 значений"
    assert all(v > 0 for v in forecast), "Все значения прогноза должны быть положительными"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
