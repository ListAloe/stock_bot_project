"""Тесты торговой стратегии."""
import pytest
from datetime import datetime, timedelta
from data.loader import load_stock_data
from models.model_selector import ModelSelector
from utils.trading import calculate_profit, generate_forecast_dates, find_trading_signals


def test_generate_forecast_dates():
    """Проверяет генерацию дат прогноза."""
    start_date = datetime.now()
    days = 30
    
    dates = generate_forecast_dates(start_date, days)
    
    assert dates is not None, "Даты не должны быть None"
    assert len(dates) == days, f"Должно быть {days} дат"


def test_find_trading_signals():
    """Проверяет поиск торговых сигналов."""
    df = load_stock_data("AAPL", period_days=365)
    series = df["Close"]
    last_date = series.index[-1]
    
    selector = ModelSelector(forecast_days=30)
    _, _, _, forecast = selector.select_best(series)
    
    dates = generate_forecast_dates(last_date, 30)
    signals = find_trading_signals(forecast, dates)
    
    assert signals is not None, "Сигналы не должны быть None"
    assert isinstance(signals, dict), "Сигналы должны быть словарём"
    assert 'buy_signals' in signals, "Должны быть buy_signals"
    assert 'sell_signals' in signals, "Должны быть sell_signals"


def test_calculate_profit():
    """Проверяет расчет прибыли."""
    df = load_stock_data("AAPL", period_days=365)
    series = df["Close"]
    current_price = series.iloc[-1]
    last_date = series.index[-1]
    
    selector = ModelSelector(forecast_days=30)
    _, _, _, forecast = selector.select_best(series)
    
    dates = generate_forecast_dates(last_date, 30)
    result = calculate_profit(
        investment=1000,
        forecast_values=forecast,
        signal_dates=dates,
        current_price=current_price
    )
    
    assert result is not None, "Результат не должен быть None"
    assert 'recommendation' in result, "Результат должен содержать рекомендацию"
    assert 'profit' in result, "Результат должен содержать прибыль"
    assert 'profit_pct' in result, "Результат должен содержать процент прибыли"
    assert 'transactions' in result, "Результат должен содержать транзакции"


def test_calculate_profit_different_investments():
    """Проверяет расчет прибыли для разных сумм инвестиции."""
    df = load_stock_data("AAPL", period_days=365)
    series = df["Close"]
    current_price = series.iloc[-1]
    last_date = series.index[-1]
    
    selector = ModelSelector(forecast_days=30)
    _, _, _, forecast = selector.select_best(series)
    
    dates = generate_forecast_dates(last_date, 30)
    
    for investment in [500, 1000, 5000]:
        result = calculate_profit(
            investment=investment,
            forecast_values=forecast,
            signal_dates=dates,
            current_price=current_price
        )
        
        assert result is not None, f"Результат для инвестиции {investment} должен быть валиден"
        assert isinstance(result['profit'], (int, float)), "Прибыль должна быть числом"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
