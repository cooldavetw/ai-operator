from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from app.config import StrictModel


class ClassifyRequest(StrictModel):
    request_id: UUID
    conversation_id: UUID | None = None
    expected_version: int = Field(default=0, ge=0)
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value

    @model_validator(mode="after")
    def version_matches_conversation(self):
        if self.conversation_id is None and self.expected_version != 0:
            raise ValueError("New conversations require expected_version=0")
        if self.conversation_id is not None and self.expected_version == 0:
            raise ValueError("Existing conversations require expected_version>=1")
        return self


class Question(StrictModel):
    id: str
    question: str


class ClassifyResponse(StrictModel):
    request_id: UUID
    conversation_id: UUID
    version: int
    status: Literal["CLASSIFIED", "NEEDS_CLARIFICATION", "UNKNOWN"]
    domain: str | None
    clarification_count: int = Field(ge=0, le=2)
    clarification: Question | None
    reason_code: Literal[
        "RULE_MATCH", "SEMANTIC_MATCH", "INSUFFICIENT_INFORMATION",
        "MULTIPLE_DOMAINS", "OUT_OF_SCOPE", "CLARIFICATION_LIMIT",
    ]
    is_final: bool
    config_version: str


class Decision(StrictModel):
    decision: Literal["CLASSIFY", "CLARIFY", "UNKNOWN"]
    domain: str | None
    clarification_id: str | None
    reason: Literal["MATCH", "INSUFFICIENT_INFORMATION", "MULTIPLE_DOMAINS", "OUT_OF_SCOPE"]

    @model_validator(mode="after")
    def consistent(self):
        if self.decision == "CLASSIFY":
            valid = self.domain is not None and self.clarification_id is None and self.reason == "MATCH"
        elif self.decision == "CLARIFY":
            valid = self.domain is None and self.clarification_id is not None and self.reason in {
                "INSUFFICIENT_INFORMATION", "MULTIPLE_DOMAINS"
            }
        else:
            valid = self.domain is None and self.clarification_id is None and self.reason == "OUT_OF_SCOPE"
        if not valid:
            raise ValueError("Inconsistent model decision")
        return self

