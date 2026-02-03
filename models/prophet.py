"""Модель Prophet от Meta для статистического прогнозирования цен акций."""
from typing import Dict, Any

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error


class ProphetForecastModel:
    """Адаптер для модели Prophet от Meta.
    
    Использует статистическое разложение временного ряда на компоненты:
    тренд, сезонность и праздники. Хорошо работает с экономметрическими данными.
    
    Имеет унифицированный интерфейс с другими моделями (fit, evaluate, forecast).
    """

    def __init__(self, forecast_horizon_days: int = 30) -> None:
        self.forecast_horizon = int(forecast_horizon_days)
        self.prophet_model = Prophet(
            yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, interval_width=0.95
        )
        self.training_data: pd.DataFrame | None = None
        self.testing_data: pd.DataFrame | None = None

    def _prepare_prophet_df(self, series: pd.Series) -> pd.DataFrame:
        df = series.reset_index()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        return df

    def fit(self, series: pd.Series) -> None:
        df = self._prepare_prophet_df(series.dropna())
        if len(df) <= self.forecast_horizon + 5:
            raise ValueError("Недостаточно данных для обучения Prophet")

        split_idx = len(df) - self.forecast_horizon
        self.training_data = df.iloc[:split_idx].reset_index(drop=True).copy()
        self.testing_data = df.iloc[split_idx:].reset_index(drop=True).copy()

        self.prophet_model.fit(self.training_data)

    def evaluate(self) -> Dict[str, Any]:
        if self.testing_data is None:
            raise ValueError("Модель Prophet не обучена")

        future_test = self.testing_data[["ds"]].copy()
        forecast_test = self.prophet_model.predict(future_test)

        y_true = self.testing_data["y"].values
        y_pred = forecast_test["yhat"].values

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = float(np.nanmean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100)
            if np.isnan(mape):
                mape = float("inf")

        return {"model_name": "Prophet", "rmse": rmse, "mape": mape, "y_pred": y_pred, "y_true": y_true}

    def forecast(self, steps: int = 30) -> np.ndarray:
        if self.training_data is None:
            raise ValueError("Модель Prophet не обучена")

        last_date = self.training_data["ds"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=int(steps), freq="D")
        future_df = pd.DataFrame({"ds": future_dates})

        forecast_result = self.prophet_model.predict(future_df)
        return forecast_result["yhat"].values
