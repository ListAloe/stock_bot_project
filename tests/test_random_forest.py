"""Тесты модели Random Forest."""
import pytest
from data.loader import load_stock_data
from models.random_forest import RandomForestForecastModel


def test_random_forest_fit():
    """Проверяет обучение Random Forest модели."""
    df = load_stock_data("AAPL", period_days=365)
    series = df["Close"]
    
    model = RandomForestForecastModel(n_lags=10)
    model.fit(series)
    
    assert model is not None, "Модель должна быть инициализирована"
    assert hasattr(model, 'forest_model'), "Модель должна иметь атрибут 'forest_model'"


def test_random_forest_evaluate():
    """Проверяет оценку качества модели."""
    df = load_stock_data("AAPL", period_days=365)
    series = df["Close"]
    
    model = RandomForestForecastModel(n_lags=10)
    model.fit(series)
    metrics = model.evaluate()
    
    assert metrics is not None, "Метрики не должны быть None"
    assert 'mape' in metrics, "Метрики должны содержать MAPE"
    assert 'rmse' in metrics, "Метрики должны содержать RMSE"
    assert metrics['mape'] >= 0, "MAPE должен быть неотрицательным"


def test_random_forest_forecast():
    """Проверяет прогноз модели."""
    df = load_stock_data("AAPL", period_days=365)
    series = df["Close"]
    
    model = RandomForestForecastModel(n_lags=10)
    model.fit(series)
    
    last_vals = series.values[-10:]
    forecast = model.forecast(last_vals, steps=5)
    
    assert forecast is not None, "Прогноз не должен быть None"
    assert len(forecast) == 5, "Прогноз должен содержать 5 значений"
    assert all(v > 0 for v in forecast), "Все значения прогноза должны быть положительными"


def test_random_forest_hyperparams():
    """Проверяет инициализацию с разными параметрами."""
    df = load_stock_data("AAPL", period_days=365)
    series = df["Close"]
    
    for n_lags in [5, 10, 15]:
        model = RandomForestForecastModel(n_lags=n_lags)
        model.fit(series)
        metrics = model.evaluate()
        assert metrics['mape'] >= 0, f"MAPE с n_lags={n_lags} должен быть валиден"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
