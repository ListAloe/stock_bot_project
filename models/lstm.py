"""Модель LSTM для прогнозирования цен акций на основе долгой краткосрочной памяти."""
import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


logger = logging.getLogger(__name__)


class LSTMForecastModel:
    """Нейросетевая LSTM-модель для прогнозирования цен акций.
    
    Использует долгую краткосрочную память для анализа временных рядов.
    Разделяет данные на обучающую (80%) и тестовую (20%) выборки.
    
    Методы:
        fit() — обучить модель на историческом ряду
        evaluate() — получить метрики качества (MAPE, RMSE)
        forecast() — сделать прогноз на указанное число дней
    """

    def __init__(self, n_lags: int = 30, epochs: int = 50, batch_size: int = 32) -> None:
        self.sequence_length = int(n_lags)
        self.training_epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.neural_network: Sequential | None = None
        self.price_scaler = MinMaxScaler()
        self._X_test: np.ndarray | None = None
        self._y_test: np.ndarray | None = None

    def _build_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Создаёт скользящие окна для обучения сети.
        
        Преобразует одномерный временной ряд в набор последовательностей
        длины sequence_length и соответствующих целевых значений.
        """
        X = []
        y = []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(self, series: pd.Series) -> None:
        """Обучает LSTM на переданном `series` (pandas.Series с индексом-дата).

        Выбрасывает ValueError при недостатке данных.
        """
        values = series.dropna().values.reshape(-1, 1)
        if len(values) < self.sequence_length + 10:
            raise ValueError("Недостаточно данных для обучения LSTM")

        scaled = self.price_scaler.fit_transform(values).flatten()

        split_idx = int(len(scaled) * 0.8)
        train_data = scaled[:split_idx]
        test_data = scaled[split_idx:]

        X_train, y_train = self._build_sequences(train_data)
        X_test, y_test = self._build_sequences(test_data)

        if X_train.size == 0 or X_test.size == 0:
            raise ValueError("Недостаточно данных для формирования обучающих последовательностей")

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        self.neural_network = Sequential(
            [
                LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1),
            ]
        )
        self.neural_network.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        self.neural_network.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.training_epochs, verbose=0)
        self._X_test = X_test
        self._y_test = y_test

    def evaluate(self) -> Dict[str, Any]:
        """Оценивает модель на отложенной тестовой выборке и возвращает словарь метрик."""
        if self.neural_network is None or self._X_test is None or self._y_test is None:
            raise ValueError("Модель LSTM не обучена или нет тестовых данных")

        y_pred_scaled = self.neural_network.predict(self._X_test, verbose=0).flatten()
        y_true_scaled = self._y_test.flatten()

        y_pred = self.price_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.price_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)

        return {"model_name": "LSTM", "rmse": rmse, "mape": mape, "y_pred": y_pred, "y_true": y_true}

    def forecast(self, last_window: np.ndarray, steps: int = 30) -> np.ndarray:
        """Делает рекурсивный прогноз по последнему окну значений (не масштабированному).

        last_window — одномерный массив длины `sequence_length`.
        """
        if self.neural_network is None:
            raise ValueError("Модель LSTM не обучена")

        last = np.asarray(last_window).reshape(-1, 1)
        if last.shape[0] != self.sequence_length:
            raise ValueError(f"Ожидалось {self.sequence_length} последних значений, получено {last.shape[0]}")

        last_scaled = self.price_scaler.transform(last).flatten()
        window = last_scaled.tolist()
        preds = []

        for _ in range(int(steps)):
            x = np.array(window[-self.sequence_length:]).reshape(1, self.sequence_length, 1)
            pred_scaled = float(self.neural_network.predict(x, verbose=0)[0, 0])
            preds.append(pred_scaled)
            window.append(pred_scaled)

        preds = np.array(preds).reshape(-1, 1)
        return self.price_scaler.inverse_transform(preds).flatten()
