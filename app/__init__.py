import os
from pathlib import Path
from loguru import logger

from app.config.application import log_setting

# 设置日志目录
log_dir = Path(__file__).parent.parent / "logs"
os.makedirs(log_dir, exist_ok=True)

# 设置日志文件名格式
log_file = log_setting.LOG_NAME_FORMAT

# 添加日志文件句柄,每天0点创建新的日志文件,不输出到控制台
logger.add(log_dir / log_file, rotation="00:00")
