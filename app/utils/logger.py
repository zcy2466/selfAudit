from datetime import datetime
from pathlib import Path
from loguru import logger
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent

LOGS_DIR = BASE_DIR / "logs"
if not LOGS_DIR.exists():
    LOGS_DIR.mkdir(parents=True)

console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}"

logger.remove()

logger.add(sys.stdout, format=console_format, level="DEBUG")

file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"

current_date = datetime.now().strftime("%Y-%m-%d")
daily_log_file = LOGS_DIR / f"selfaudit_{current_date}.log"
logger.add(daily_log_file, format=file_format, rotation="00:00", retention="30 days", encoding="utf-8")