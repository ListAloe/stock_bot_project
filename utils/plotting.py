"""Визуализация исторических данных и прогнозов."""
from io import BytesIO
from datetime import timedelta
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_forecast(
    historical_series: pd.Series,
    forecast_values: Iterable[float],
    ticker: str,
    days_history: int = 90,
    forecast_days: int = 30,
) -> BytesIO:
    """Нарисовать график с историей и прогнозом цены.
    
    Args:
        historical_series: Временной ряд исторических цен (pandas.Series)
        forecast_values: Прогнозные значения на указанный период
        ticker: Тикер акции для заголовка графика
        days_history: Количество последних дней истории для отображения
        forecast_days: Количество дней прогноза
        
    Returns:
        BytesIO с PNG-изображением графика
        
    Raises:
        ValueError: Если исторические данные отсутствуют
    """
    hist_recent = historical_series.dropna().tail(int(days_history))
    if hist_recent.empty:
        raise ValueError("Нет исторических данных для визуализации")

    hist_dates = hist_recent.index
    hist_values = hist_recent.values

    last_date = hist_dates[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, int(forecast_days) + 1)]

    plt.figure(figsize=(12, 6))
    plt.plot(hist_dates, hist_values, label="История", color="#1f77b4", linewidth=2)
    plt.plot(forecast_dates, np.asarray(list(forecast_values)), label="Прогноз", color="#ff7f0e", linestyle="--", linewidth=2)

    plt.title(f"Прогноз цены акций {ticker} на {forecast_days} дней", fontsize=14)
    plt.xlabel("Дата")
    plt.ylabel("Цена, USD")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return buf
