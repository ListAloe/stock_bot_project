"""Модель Random Forest с лаговыми признаками для прогнозирования цен акций."""
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


class RandomForestForecastModel:
    """Модель Random Forest с лаговыми признаками.
    
    Использует ансамбль решающих деревьев на основе прошлых значений ряда.
    Простая, быстрая и интерпретируемая модель для временных рядов.
    """

    def __init__(self, n_lags: int = 10) -> None:
        self.sequence_length = int(n_lags)
        self.forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self._X_test: pd.DataFrame | None = None
        self._y_test: pd.Series | None = None

    def _create_lag_features(self, series: pd.Series) -> pd.DataFrame:
        """Создаёт датафрейм с лаговыми признаками и целевой переменной."""
        df = pd.DataFrame({"target": series}).copy()
        for lag in range(1, self.sequence_length + 1):
            df[f"lag_{lag}"] = df["target"].shift(lag)
        df = df.dropna()
        return df

    def fit(self, series: pd.Series) -> None:
        df = self._create_lag_features(series)
        if df.shape[0] < 10:
            raise ValueError("Недостаточно данных для обучения RandomForest")

        X = df[[f"lag_{i}" for i in range(1, self.sequence_length + 1)]].values
        y = df["target"].values

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.forest_model.fit(X_train, y_train)
        self._X_test = X_test
        self._y_test = y_test

    def evaluate(self) -> Dict[str, Any]:
        if self._X_test is None or self._y_test is None:
            raise ValueError("Модель не обучена или нет тестовых данных")

        y_pred = self.forest_model.predict(self._X_test)
        y_true = self._y_test

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)

        return {"model_name": "RandomForestLag", "rmse": rmse, "mape": mape, "y_pred": y_pred, "y_true": y_true}

    def forecast(self, last_values: np.ndarray, steps: int = 30) -> np.ndarray:
        """Рекурсивный прогноз: принимает массив последних sequence_length значений и выдаёт прогноз на steps."""
        arr = np.asarray(last_values, dtype=float).flatten()
        if arr.size != self.sequence_length:
            raise ValueError(f"Ожидалось {self.sequence_length} последних значений, получено {arr.size}")

        preds = []
        window = arr.copy()
        for _ in range(int(steps)):
            pred = float(self.forest_model.predict(window.reshape(1, -1))[0])
            preds.append(pred)
            window = np.roll(window, -1)
            window[-1] = pred

        return np.array(preds)
