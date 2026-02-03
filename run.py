"""Точка входа для запуска бота: `python run.py`."""
from __future__ import annotations

import asyncio

from bot.app import start_polling


if __name__ == "__main__":
    asyncio.run(start_polling())
