from pydantic_settings import BaseSettings
from environs import Env

env = Env()
env.read_env()


class LogSettings(BaseSettings):
    LOG_NAME_FORMAT: str = "{time:YYYY-MM-DD}.log"


class Settings(BaseSettings):
    # elasticsearch setting
    ES_URL: str = "http://127.0.0.1:9200"
    ES_PWD: str = "brainbank_pwd"
    CHAT_HISTORY_INDEX: str = "conversation-history"


settings = Settings()
log_setting = LogSettings()
