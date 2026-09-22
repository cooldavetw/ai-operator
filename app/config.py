import os
import re
import unicodedata
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().casefold()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Domain(StrictModel):
    # Optional legacy fields keep stored conversation snapshots readable.
    display_name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=2000)
    examples: list[str] = Field(default_factory=list, max_length=30)
    exclusions: list[str] = Field(default_factory=list, max_length=30)
    exact_matches: list[str] = Field(default_factory=list, max_length=100)


class Clarification(StrictModel):
    text: str = Field(min_length=1, max_length=300)


class Policy(StrictModel):
    max_clarifications: int = Field(default=2, ge=0, le=2)
    session_ttl_minutes: int = Field(default=30, ge=1, le=1440)


class DomainConfig(StrictModel):
    version: str = Field(min_length=1, max_length=100)
    policy: Policy = Field(default_factory=Policy)
    domains: dict[str, Domain] = Field(min_length=1, max_length=20)
    clarifications: dict[str, Clarification] = Field(min_length=2, max_length=20)

    @model_validator(mode="after")
    def check_keys(self):
        matches = set()
        for key, domain in self.domains.items():
            if key == "UNKNOWN" or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", key):
                raise ValueError("Domain IDs must be uppercase identifiers; UNKNOWN is reserved")
            for text in domain.exact_matches:
                value = normalize(text)
                if not value or value in matches:
                    raise ValueError("Exact-match rules must be nonempty and globally unique")
                matches.add(value)
        for key in self.clarifications:
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key):
                raise ValueError("Invalid clarification ID")
        return self


def load_config(path: Path) -> DomainConfig:
    return DomainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class Settings(StrictModel):
    api_key: SecretStr
    config_path: Path = Path("config/domains.yaml")
    database_path: Path = Path("data/classifier.sqlite3")
    model_path: Path = Path("models/model.gguf")
    model_sha256: str = Field(default="", pattern=r"^([a-fA-F0-9]{64})?$")
    chat_format: str | None = None
    n_ctx: int = Field(default=4096, ge=1024, le=32768)
    n_threads: int = Field(default=4, ge=1, le=256)
    max_tokens: int = Field(default=192, ge=32, le=512)
    temperature: float = Field(default=0, ge=0, le=2)
    inference_timeout_seconds: float = Field(default=30, gt=0, le=300)
    startup_timeout_seconds: float = Field(default=180, gt=0, le=600)
    queue_timeout_seconds: float = Field(default=30, gt=0, le=300)
    queue_capacity: int = Field(default=8, ge=0, le=1000)
    expired_retention_seconds: int = Field(default=86400, ge=0, le=604800)

    @model_validator(mode="after")
    def validate_key(self):
        key = self.api_key.get_secret_value()
        if not key.strip() or key == "replace-with-a-random-secret":
            raise ValueError("API_KEY must be nonempty; replace the example")
        return self

    @classmethod
    def from_env(cls):
        values = {key: os.environ[key.upper()] for key in cls.model_fields if key.upper() in os.environ}
        if os.environ.get("API_KEY_FILE"):
            values["api_key"] = Path(os.environ["API_KEY_FILE"]).read_text().strip()
        if values.get("chat_format") == "":
            values["chat_format"] = None
        return cls.model_validate(values)

