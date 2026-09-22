"""PydanticAI model adapter for in-process, offline llama.cpp inference."""
from pydantic_ai.messages import (
    ModelResponse, RetryPromptPart, SystemPromptPart, TextPart, UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.profiles import ModelProfile

from app.errors import ServiceError


class LocalLlamaModel(Model):
    def __init__(self, llama, options):
        super().__init__(profile=ModelProfile(supports_json_schema_output=True))
        self.llama = llama
        self.options = options

    @property
    def model_name(self):
        return "local-gguf"

    @property
    def system(self):
        return "llama.cpp"

    async def request(self, messages, model_settings, model_request_parameters):
        _, parameters = self.prepare_request(model_settings, model_request_parameters)
        chat = []
        for message in messages:
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    role, content = "system", part.content
                elif isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    role, content = "user", part.content
                elif isinstance(part, TextPart):
                    role, content = "assistant", part.content
                elif isinstance(part, RetryPromptPart):
                    role, content = "user", part.model_response()
                else:
                    raise ValueError("Local adapter only supports text and validation feedback")
                # Some chat templates require alternating roles.
                if chat and chat[-1]["role"] == role:
                    chat[-1]["content"] += "\n\n" + content
                else:
                    chat.append({"role": role, "content": content})
        input_tokens = sum(len(self.llama.tokenize(m["content"].encode())) for m in chat)
        if input_tokens + self.options["max_tokens"] + 512 > self.options["n_ctx"]:
            raise ServiceError(422, "CONTEXT_TOO_LONG", "對話超過模型容量，請縮短問題或開始新對話。")
        if parameters.output_object is None:
            raise ValueError("Local adapter requires native structured output")
        try:
            result = self.llama.create_chat_completion(
                messages=chat,
                response_format={"type": "json_object", "schema": parameters.output_object.json_schema},
                temperature=self.options["temperature"], max_tokens=self.options["max_tokens"],
            )
        except ValueError as exc:
            if "context window" in str(exc).lower():
                raise ServiceError(422, "CONTEXT_TOO_LONG", "對話超過模型容量。") from exc
            raise
        choice = result["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ServiceError(502, "INVALID_MODEL_OUTPUT", "模型輸出不完整，請重試。")
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ServiceError(502, "INVALID_MODEL_OUTPUT", "模型未回傳有效 JSON。")
        return ModelResponse(parts=[TextPart(content)], model_name=self.model_name)
