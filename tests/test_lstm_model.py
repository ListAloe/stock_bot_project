"""Тесты модели LSTM."""
import pytest
from data.loader import load_stock_data
from models.lstm import LSTMForecastModel


def test_lstm_fit():
    """Проверяет обучение LSTM модели."""
    df = load_stock_data("AAPL", period_days=730)
    series = df["Close"]
    
    model = LSTMForecastModel(n_lags=30, epochs=5)
    model.fit(series)
    
    assert model is not None, "Модель должна быть инициализирована"
    assert hasattr(model, 'neural_network'), "Модель должна иметь атрибут 'neural_network'"


def test_lstm_evaluate():
    """Проверяет оценку качества LSTM модели."""
    df = load_stock_data("AAPL", period_days=730)
    series = df["Close"]
    
    model = LSTMForecastModel(n_lags=30, epochs=5)
    model.fit(series)
    metrics = model.evaluate()
    
    assert metrics is not None, "Метрики не должны быть None"
    assert 'mape' in metrics, "Метрики должны содержать MAPE"
    assert 'rmse' in metrics, "Метрики должны содержать RMSE"
    assert metrics['mape'] >= 0, "MAPE должен быть неотрицательным"


def test_lstm_forecast():
    """Проверяет прогноз LSTM модели."""
    df = load_stock_data("AAPL", period_days=730)
    series = df["Close"]
    
    model = LSTMForecastModel(n_lags=30, epochs=5)
    model.fit(series)
    
    last_vals = series.values[-30:]
    forecast = model.forecast(last_vals, steps=5)
    
    assert forecast is not None, "Прогноз не должен быть None"
    assert len(forecast) == 5, "Прогноз должен содержать 5 значений"
    assert all(v > 0 for v in forecast), "Все значения прогноза должны быть положительными"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
