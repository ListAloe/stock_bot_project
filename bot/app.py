"""Основной модуль Telegram-бота: интерфейс пользователя и бизнес-логика."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import logging
from dotenv import load_dotenv
import os

from data.loader import load_stock_data
from utils.plotting import plot_forecast
from utils.trading import calculate_profit, generate_forecast_dates
from models.model_selector import ModelSelector
from utils.logger import log_request


load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    TOKEN = ""


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Константы сообщений
DISCLAIMER = (
    "\n\n⚠️ <b>Внимание:</b> Это образовательный проект. "
    "Прогнозы не являются финансовой рекомендацией."
)
EMOJI_BUY = "🟢 Купить"
EMOJI_SELL = "🔴 Продать"
EMOJI_UP = "📈"
EMOJI_DOWN = "📉"


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _sanitize_ticker(text: str) -> str:
    return text.strip().upper()


def create_bot() -> tuple[Any, Any, Any]:
    """Инициализировать Telegram-бота и его компоненты.
    
    Создаёт объекты бота (Bot), диспетчера (Dispatcher) и роутера (Router)
    с использованием переменной окружения BOT_TOKEN.
    
    Returns:
        Кортеж (bot, dispatcher, router)
        
    Raises:
        ValueError: Если BOT_TOKEN не найден в переменных окружения
    """
    from aiogram import Bot, Dispatcher, Router
    from aiogram.fsm.storage.memory import MemoryStorage

    if not TOKEN:
        raise ValueError("BOT_TOKEN не найден в окружении. Создайте файл .env с переменной BOT_TOKEN.")

    bot = Bot(token=TOKEN)
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    router = Router()
    dp.include_router(router)
    return bot, dp, router


def register_handlers(router: Any) -> None:
    """Зарегистрировать обработчики команд и сообщений.
    
    Подключает все обработчики к роутеру бота для правильной обработки
    команд пользователя и взаимодействия с ним через конечный автомат состояний (FSM).
    
    Args:
        router: Объект aiogram.Router для регистрации обработчиков
    """
    from aiogram.types import Message, BufferedInputFile
    from aiogram.filters import Command
    from aiogram.fsm.context import FSMContext
    from aiogram.fsm.state import State, StatesGroup

    # Класс состояний создаём локально
    class UserInput(StatesGroup):
        waiting_for_ticker = State()
        waiting_for_amount = State()

    @router.message(Command("start"))
    async def cmd_start(message: Message, state: FSMContext) -> None:
        await message.answer(
            "📈 <b>Прогнозирование цен акций</b>\n\n"
            "Введите тикер акции (AAPL, MSFT, TSLA и т.д.):" + DISCLAIMER,
            parse_mode="HTML",
        )
        await state.set_state(UserInput.waiting_for_ticker)


    @router.message(UserInput.waiting_for_ticker)
    async def get_ticker(message: Message, state: FSMContext) -> None:
        ticker = _sanitize_ticker(message.text)
        if not ticker or len(ticker) > 20:
            await message.answer("❌ Неверный тикер. Введите корректное значение (например: AAPL)")
            return

        await state.update_data(ticker=ticker)
        await message.answer(f"✅ Тикер <b>{ticker}</b> принят.\n\nТеперь введите сумму инвестиции в USD (например: 1000):")
        await state.set_state(UserInput.waiting_for_amount)


    @router.message(UserInput.waiting_for_amount)
    async def get_amount(message: Message, state: FSMContext) -> None:
        data = await state.get_data()
        ticker = data.get("ticker")
        try:
            amount = float(message.text.replace(",", ".").strip())
            if amount <= 0 or amount > 1e9:
                raise ValueError
        except Exception:
            await message.answer("❌ Пожалуйста, введите корректное число (от 1 до 1 млрд).")
            return

        await message.answer(f"✅ Данные приняты\n\n<b>{ticker}</b> | {_format_currency(amount)}\n\n⏳ Загружаю и анализирую данные...")

        try:
            df = load_stock_data(ticker, period_days=730)
            current_price = float(df["Close"].iloc[-1])
            last_date = df.index[-1]

            selector = ModelSelector(forecast_days=30)
            best_model, model_name, metrics, forecast_values = selector.select_best(df["Close"])

            forecast_dates = generate_forecast_dates(last_date, 30)

            strategy_result = calculate_profit(
                investment=amount,
                forecast_values=forecast_values,
                signal_dates=forecast_dates,
                current_price=current_price,
            )

            buf = plot_forecast(df["Close"], forecast_values, ticker, days_history=90, forecast_days=30)
            photo_file = BufferedInputFile(buf.getvalue(), filename=f"{ticker}_forecast.png")
            await message.answer_photo(photo=photo_file, caption="📊 История (90 дней) и прогноз (30 дней)")

            price_change_pct = ((forecast_values[-1] - current_price) / current_price) * 100
            change_emoji = EMOJI_UP if price_change_pct >= 0 else EMOJI_DOWN
            await message.answer(
                f"{change_emoji} <b>Прогноз на 30 дней</b>\n\n"
                f"Текущая цена: {_format_currency(current_price)}\n"
                f"Прогноз: {_format_currency(forecast_values[-1])} ({price_change_pct:+.2f}%)"
            )

            rec_text = f"💡 <b>Рекомендация:</b> {strategy_result['recommendation']}\n\n"
            if strategy_result.get("transactions"):
                rec_text += "📈 <b>Торговая стратегия:</b>\n"
                for t in strategy_result["transactions"]:
                    action = EMOJI_BUY if t["action"] == "ПОКУПКА" else EMOJI_SELL
                    rec_text += f"{action} {t['date']} @ {_format_currency(t['price'])}\n"
                rec_text += f"\n💰 <b>Ожидаемая прибыль:</b> {_format_currency(strategy_result['profit'])} ({strategy_result['profit_pct']:+.2f}%)"
            else:
                rec_text += f"\n💰 <b>Прибыль:</b> {_format_currency(0)} (нет сигналов)"

            await message.answer(rec_text)
            await message.answer(
                "🔄 Хотите проанализировать другую акцию? Введите новый тикер или нажмите /start." + DISCLAIMER,
                parse_mode="HTML",
            )

            # Логируем запрос
            try:
                log_request(
                    user_id=message.from_user.id,
                    timestamp=datetime.now(),
                    ticker=ticker,
                    investment=amount,
                    best_model=model_name,
                    metric_name="mape",
                    metric_value=float(metrics.get("mape", 0)),
                    profit=float(strategy_result.get("profit", 0.0)),
                )
            except Exception as exc:
                logger.warning("Ошибка при логировании запроса: %s", exc)

        except Exception as exc:
            logger.exception("Ошибка при обработке запроса")
            await message.answer(f"❌ Не удалось обработать запрос.\n\nПроверьте тикер и попробуйте снова.")

        await state.clear()


    @router.message()
    async def handle_unexpected_input(message: Message, state: FSMContext) -> None:
        current_state = await state.get_state()
        if current_state is None:
            await message.answer(
                "ℹ️ Введите тикер акции (например: GOOGL) или используйте /start для начала." + DISCLAIMER,
                parse_mode="HTML",
            )


async def start_polling() -> None:
    """Запустить бота в режиме опроса (polling).
    
    Инициализирует бота, регистрирует все обработчики и запускает
    длительный опрос Telegram API для получения новых сообщений.
    """
    bot, dp, router = create_bot()
    register_handlers(router)
    await dp.start_polling(bot)
