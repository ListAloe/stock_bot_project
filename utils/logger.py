"""Логирование запросов пользователей в CSV файл `logs/logs.csv`."""
import csv
import os
from datetime import datetime
from typing import Any

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "logs.csv")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Поля для CSV логов
CSV_HEADERS = [
    "user_id", "timestamp", "ticker", "investment",
    "best_model", "metric_name", "metric_value", "profit"
]


def _ensure_log_directory() -> None:
    """Создаёт директорию для логов, если её нет."""
    os.makedirs(LOG_DIR, exist_ok=True)


def log_request(
    user_id: int,
    timestamp: datetime,
    ticker: str,
    investment: float,
    best_model: str,
    metric_name: str,
    metric_value: float,
    profit: float,
) -> None:
    """Записать запрос пользователя в лог-файл.
    
    Сохраняет информацию о запросе в CSV-файл logs/logs.csv.
    Ошибки при логировании подавляются, чтобы не нарушить работу бота.
    
    Args:
        user_id: ID пользователя Telegram
        timestamp: Дата и время запроса
        ticker: Тикер проанализированной акции
        investment: Сумма инвестиции
        best_model: Название выбранной модели
        metric_name: Название метрики качества
        metric_value: Значение метрики
        profit: Рассчитанная прибыль/убыток
    """
    try:
        _ensure_log_directory()
        file_exists = os.path.isfile(LOG_FILE)

        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(CSV_HEADERS)
            
            writer.writerow([
                user_id,
                timestamp.strftime(TIMESTAMP_FORMAT),
                ticker,
                round(float(investment), 2),
                best_model,
                metric_name,
                round(float(metric_value), 6),
                round(float(profit), 2),
            ])
    except Exception:
        # Не поднимаем исключение — логирование не должно ломать бота
        pass
