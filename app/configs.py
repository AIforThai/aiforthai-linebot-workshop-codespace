from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Configs(BaseSettings):
    AIFORTHAI_APIKEY: str
    LINE_CHANNEL_ACCESS_TOKEN: str
    LINE_CHANNEL_SECRET: str

    # Basic NLP VARIABLES
    WAV_URL: str
    WAV_FILE: str
    DIR_FILE: str
    URL_PARTII: str
    URL_VAJA: str

    # Image VARIABLES
    URL_MAEWMONG: str
    IMG_RESULT: str
    URL_PERSON_DETEC: str
    URL_CAPGEN: str

    # Pathumma LLM
    URL_AUDIOQA: str
    URL_TEXTQA: str
    URL_VQA: str
    URL_PATHUMMA_CHAT: str
    
    def __init__(self, **values: Any) -> None:
        super().__init__(**values)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore",str_strip_whitespace=True)
