from dataclasses import dataclass
import os
from typing import Literal, List

@dataclass(frozen=True)
class EnvConfig:
    SLACK_BOT_TOKEN: str = ""
    SLACK_SIGNING_SECRET: str = ""
    DYNAMODB_TABLE_NAME: str = ""
    OPENAI_ORG_ID: str|None = None
    OPENAI_API_KEY: str|None = None
    OPENAI_MODEL: str = ""
    IMAGE_MODEL: str = ""
    IMAGE_QUALITY: Literal["standard", "hd"] = "hd"
    IMAGE_SIZE: Literal['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792'] = "1024x1024"
    IMAGE_STYLE: Literal["vivid", "natural"] = "vivid"
    SYSTEM_MESSAGE: str|None = None
    TEMPERATURE: float = 0
    MAX_LEN_SLACK: int = 0
    MAX_LEN_OPENAI: int = 0

image_size_list: List[Literal['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792']] = ['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792']
image_size = os.environ.get("IMAGE_SIZE")

env_config = EnvConfig(
    SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"],
    SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"],
    DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "slack-ai-bot-context"),
    OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID", None),
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None),
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o"),
    IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "dall-e-3"),
    IMAGE_QUALITY = "standard" if os.environ.get("IMAGE_QUALITY") == "standard" else "hd",  # standard, hd
    IMAGE_SIZE = image_size_list in ['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792'] and image_size_list or "1024x1024",
    IMAGE_STYLE = "natural" if os.environ.get("IMAGE_STYLE") == "natural" else "vivid",  # vivid, natural
    SYSTEM_MESSAGE = os.environ.get("SYSTEM_MESSAGE", None),
    TEMPERATURE = float(os.environ.get("TEMPERATURE", 0)),
    MAX_LEN_SLACK = int(os.environ.get("MAX_LEN_SLACK", 3000)),
    MAX_LEN_OPENAI = int(os.environ.get("MAX_LEN_OPENAI", 4000)),
)