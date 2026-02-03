"""Модуль загрузки данных — аккуратные, проверенные функции для получения исторических цен."""
from typing import Optional
import pandas as pd
import yfinance as yf


def load_stock_data(ticker: str, period_days: int = 730) -> pd.DataFrame:
    """Загружает котировки с Yahoo Finance за указанный период.
    
    Возвращает DataFrame с историческими ценами закрытия (колонка 'Close').
    При возникновении проблем выбрасывает ValueError с описанием ошибки.
    
    Args:
        ticker: Тикер акции (например, 'AAPL')
        period_days: Период загрузки в днях (по умолчанию 730 дней = 2 года)
        
    Returns:
        DataFrame с индексом-датой и колонкой 'Close'
    
    Raises:
        ValueError: Если тикер пуст, данные недоступны или отсутствует колонка 'Close'
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Пустой тикер")

    try:
        hist = yf.download(ticker, period=f"{period_days}d", auto_adjust=True, progress=False)
    except Exception as exc:
        raise ValueError(f"Ошибка при подключении к источнику данных: {exc}")

    if hist is None or hist.empty:
        raise ValueError(f"Нет данных для тикера: {ticker}")

    # Обработка мультииндекса колонок
    if isinstance(hist.columns, pd.MultiIndex):
        close_series = hist["Close"].iloc[:, 0] if "Close" in hist.columns.levels[0] else hist.iloc[:, 3]
    else:
        if "Close" not in hist.columns:
            raise ValueError(f"В данных отсутствует колонка Close для {ticker}")
        close_series = hist["Close"]

    df = close_series.to_frame(name="Close")
    df.index.name = "Date"
    df = df.dropna().astype(float)

    if df.empty:
        raise ValueError(f"После очистки данных не осталось записей для {ticker}")

    return df


def get_latest_price(ticker: str) -> Optional[float]:
    """Получает последнюю известную цену закрытия для тикера.
    
    Args:
        ticker: Тикер акции
        
    Returns:
        Последняя цена закрытия или None при ошибке загрузки
    """
    try:
        df = load_stock_data(ticker, period_days=7)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None
