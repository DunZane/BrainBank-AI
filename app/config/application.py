from pydantic_settings import BaseSettings
from environs import Env

env = Env()
env.read_env()


class LogSettings(BaseSettings):
    LOG_NAME_FORMAT: str = "{time:YYYY-MM-DD}.log"


class Settings(BaseSettings):
    # elasticsearch setting
    ES_URL: str = env.str("ES_URL", "http://localhost:9200")
    ES_PWD: str = env.str("ES_PWD", "brainbank_pwd")
    CHAT_HISTORY_INDEX: str = env.str("CHAT_HISTORY_INDEX", "conversation-history")


settings = Settings()
log_setting = LogSettings()
