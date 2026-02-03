"""Автоматический выбор лучшей модели прогнозирования по метрике MAPE."""
import logging
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np

from models.random_forest import RandomForestForecastModel
from models.prophet import ProphetForecastModel
from models.lstm import LSTMForecastModel

logger = logging.getLogger(__name__)


class ModelSelector:
    def __init__(self, forecast_days: int = 30):
        self.forecast_days = forecast_days
        self.available_models = {
            "RandomForest": (RandomForestForecastModel(n_lags=10), 10, 30),
            "Prophet": (ProphetForecastModel(forecast_horizon_days=30), None, 30),
            "LSTM": (LSTMForecastModel(n_lags=30, epochs=30), 30, 30),
        }
        self.best_model_name = None
        self.best_model = None
        self.best_metrics = None

    def select_best(self, series: pd.Series) -> Tuple[Any, str, Dict[str, float], Any]:
        """Обучает все модели, выбирает лучшую по MAPE.
        
        Возвращает: (модель, имя, метрики, прогноз)
        """
        min_mape = float("inf")
        best_forecast = None

        for name, (model, n_lags, forecast_steps) in self.available_models.items():
            try:
                logger.info("Обучение модели: %s", name)
                model.fit(series)
                metrics = model.evaluate()

                logger.info("%s — MAPE: %.2f%%, RMSE: %.2f", name, metrics.get("mape", 0.0), metrics.get("rmse", 0.0))

                if metrics.get("mape", float("inf")) < min_mape:
                    min_mape = float(metrics.get("mape", min_mape))
                    self.best_model_name = name
                    self.best_model = model
                    self.best_metrics = metrics

            except Exception as exc:
                logger.exception("Ошибка при обучении %s: %s", name, exc)
                continue

        if self.best_model is None:
            raise RuntimeError("Ни одна модель не смогла обучиться!")

        # Генерация прогноза
        best_forecast = self._generate_forecast(series)

        return self.best_model, self.best_model_name, self.best_metrics, best_forecast

    def _generate_forecast(self, series: pd.Series) -> np.ndarray:
        """Генерирует прогноз для выбранной лучшей модели."""
        if self.best_model_name == "RandomForest":
            last_vals = series.values[-10:]
            return self.best_model.forecast(last_vals, steps=self.forecast_days)
        elif self.best_model_name == "Prophet":
            return self.best_model.forecast(steps=self.forecast_days)
        elif self.best_model_name == "LSTM":
            last_vals = series.values[-30:]
            return self.best_model.forecast(last_vals, steps=self.forecast_days)
        
        raise RuntimeError(f"Неизвестная модель: {self.best_model_name}")
