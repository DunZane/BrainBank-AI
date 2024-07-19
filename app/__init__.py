import os

from dotenv import load_dotenv
from loguru import logger
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    logger.error(env_file)


class BaseLogger:
    log_dir = Path(__file__).parent.parent / "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = '{time:YYYY-MM-DD}.log'
    logger.add(log_dir / log_file)

    def __init__(self) -> None:
        self.info = logger.info
        self.debug = logger.debug
        self.warning = logger.warning
        self.error = logger.error


logger = BaseLogger()
