"""Торговая стратегия: поиск сигналов и расчёт ожидаемой прибыли."""
import numpy as np
from scipy.signal import argrelextrema
from typing import List, Dict, Any
from datetime import timedelta, date, datetime


def find_trading_signals(forecast_values: np.ndarray, signal_dates: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Найти точки покупки и продажи в прогнозе.
    
    Определяет локальные минимумы (точки покупки) и максимумы (точки продажи)
    в прогнозном ряду котировок.
    
    Args:
        forecast_values: Массив прогнозируемых цен
        signal_dates: Список дат соответствующих прогнозам
        
    Returns:
        Словарь с ключами 'buy_signals' и 'sell_signals', каждый содержит
        список сигналов (день, дата, цена)
    """
    prices = np.asarray(forecast_values)
    if prices.size < 3:
        return {"buy_signals": [], "sell_signals": []}

    buy_idx = argrelextrema(prices, np.less, order=1)[0]
    sell_idx = argrelextrema(prices, np.greater, order=1)[0]

    buy_signals = [{"day": int(i), "date": signal_dates[i], "price": float(prices[i])} for i in buy_idx]
    sell_signals = [{"day": int(i), "date": signal_dates[i], "price": float(prices[i])} for i in sell_idx]

    return {"buy_signals": buy_signals, "sell_signals": sell_signals}


def calculate_profit(investment: float, forecast_values: np.ndarray, signal_dates: List[str], current_price: float) -> Dict[str, Any]:
    """Вычисляет результат простой торговой стратегии на основе найденных сигналов.

    Стратегия: покупаем в первый локальный минимум, продаём в первом следующем локальном максимуме.
    Повторяем до конца списка сигналов.
    """
    signals = find_trading_signals(forecast_values, signal_dates)
    buy_signals = sorted(signals["buy_signals"], key=lambda x: x["day"])
    sell_signals = sorted(signals["sell_signals"], key=lambda x: x["day"]) 

    transaction_log: List[Dict[str, Any]] = []
    cash = float(investment)
    sell_cursor = 0

    for buy in buy_signals:
        # Ищем следующий сигнал продажи
        sell = None
        for j in range(sell_cursor, len(sell_signals)):
            if sell_signals[j]["day"] > buy["day"]:
                sell = sell_signals[j]
                sell_cursor = j + 1
                break
        
        if not sell:
            break

        # Выполняем транзакцию
        shares_bought = cash / buy["price"] if buy["price"] > 0 else 0.0
        transaction_log.append({
            "action": "ПОКУПКА",
            "date": buy["date"],
            "price": round(buy["price"], 4),
            "shares": round(shares_bought, 8),
        })

        cash = shares_bought * sell["price"]
        transaction_log.append({
            "action": "ПРОДАЖА",
            "date": sell["date"],
            "price": round(sell["price"], 4),
            "shares": round(shares_bought, 8),
        })

    # Вычисляем итоговые метрики
    final_value = cash
    profit = final_value - investment
    profit_pct = (profit / investment * 100) if investment > 0 else 0.0

    recommendation = "Следуйте сигналам ниже для возможной прибыли." if transaction_log else "Нет чётких сигналов для торговли в ближайшие 30 дней."

    return {
        "recommendation": recommendation,
        "transactions": transaction_log,
        "final_value": round(final_value, 2),
        "profit": round(profit, 2),
        "profit_pct": round(profit_pct, 2),
        "signals": signals,
    }


def generate_forecast_dates(last_date: datetime, days: int = 30) -> List[str]:
    """Генерирует список дат (строк) для прогноза: от следующего дня до days включительно."""
    return [
        (last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days + 1)
    ]
