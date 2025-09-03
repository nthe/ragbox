import logging
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


logger = logging.getLogger("uvicorn")


class Settings(BaseSettings):
    ragbox_db_path: Path = Path("./data/db")
    ragbox_db_table: str = "documents"

    ragbox_embedding_model: str = "intfloat/multilingual-e5-small"
    ragbox_embedding_model_device: Literal["cpu", "mps", "cuda"] = "cpu"

    ragbox_chat_model: str = "azure/Ministral-3B"
    ragbox_chat_history_limit: int = 15

    azure_api_base: str
    azure_api_key: SecretStr
    azure_api_version: str

    tokenizers_parallelism: bool = True
    hf_home: Path = Path("./data/hf")

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings.model_validate({})
logger.info(
    "[settings] initialized with %s",
    settings.model_dump_json(indent=4),
)
