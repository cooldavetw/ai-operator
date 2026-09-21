import hashlib
import json
import threading
import time
from uuid import uuid4

from app.config import DomainConfig, Settings, normalize
from app.errors import ServiceError
from app.runtime import Runtime, validate_decision
from app.schemas import ClassifyRequest, ClassifyResponse, Decision, Question
from app.store import Store


class Classifier:
    def __init__(self, settings: Settings, config: DomainConfig, runtime: Runtime, store: Store):
        self.settings = settings
        self.config = config
        self.runtime = runtime
        self.store = store
        self.lock = threading.Lock()

    def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        with self.lock:
            return self._classify(request)

    def _classify(self, request: ClassifyRequest) -> ClassifyResponse:
        now = time.time()
        self.store.purge(now - self.settings.expired_retention_seconds)
        payload = request.model_dump_json()
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        with self.store.connect() as db:
            replay = db.execute("SELECT * FROM requests WHERE id=?", (str(request.request_id),)).fetchone()
            if replay:
                if replay["fingerprint"] != fingerprint:
                    raise ServiceError(409, "REQUEST_ID_REUSED", "此 request_id 已用於不同內容。")
                if replay["expires_at"] <= now:
                    raise ServiceError(410, "CONVERSATION_EXPIRED", "對話已過期，請開始新對話。")
                return ClassifyResponse.model_validate_json(replay["response"])

            if request.conversation_id:
                row = db.execute("SELECT * FROM conversations WHERE id=?", (str(request.conversation_id),)).fetchone()
                if row is None:
                    raise ServiceError(404, "CONVERSATION_NOT_FOUND", "找不到對話，請開始新對話。")
                if row["expires_at"] <= now:
                    raise ServiceError(410, "CONVERSATION_EXPIRED", "對話已過期，請開始新對話。")
                state = json.loads(row["state"])
                if row["version"] != request.expected_version:
                    raise ServiceError(409, "VERSION_CONFLICT", "對話版本不符，請使用最新回應版本。")
                if state["is_final"]:
                    raise ServiceError(409, "CONVERSATION_FINISHED", "對話已完成，請開始新對話。")
                config = DomainConfig.model_validate(state["config"])
                conversation_id = request.conversation_id
                # Absolute TTL, fixed at conversation creation.
                expires_at = row["expires_at"]
            else:
                config = self.config
                conversation_id = uuid4()
                expires_at = now + config.policy.session_ttl_minutes * 60
                state = {"history": [], "asked": [], "config": config.model_dump(), "is_final": False}

        history = [*state["history"], {"role": "user", "content": request.message}]
        # Only first-turn, whole-message allowlisted rules bypass semantic inference.
        domain = None
        if not state["history"]:
            for key, definition in config.domains.items():
                if normalize(request.message) in {normalize(text) for text in definition.exact_matches}:
                    domain = key
                    break
        if domain:
            decision = Decision(decision="CLASSIFY", domain=domain, clarification_id=None, reason="MATCH")
            source = "RULE_MATCH"
        else:
            decision = validate_decision(self.runtime.infer(history, config), config)
            source = "SEMANTIC_MATCH"

        asked = list(state["asked"])
        question = None
        if decision.decision == "CLASSIFY":
            status, result_domain, reason = "CLASSIFIED", decision.domain, source
        elif decision.decision == "UNKNOWN":
            status, result_domain, reason = "UNKNOWN", "UNKNOWN", "OUT_OF_SCOPE"
        elif len(asked) >= config.policy.max_clarifications:
            status, result_domain, reason = "UNKNOWN", "UNKNOWN", "CLARIFICATION_LIMIT"
        else:
            question_id = decision.clarification_id
            if question_id in asked:
                question_id = next(key for key in config.clarifications if key not in asked)
            question = Question(id=question_id, question=config.clarifications[question_id].text)
            asked.append(question_id)
            history.append({"role": "assistant", "content": question.question})
            status, result_domain, reason = "NEEDS_CLARIFICATION", None, decision.reason

        response = ClassifyResponse(
            request_id=request.request_id, conversation_id=conversation_id,
            version=request.expected_version + 1, status=status, domain=result_domain,
            clarification_count=len(asked), clarification=question, reason_code=reason,
            is_final=status != "NEEDS_CLARIFICATION", config_version=config.version,
        )
        state.update(history=history, asked=asked, is_final=response.is_final)
        if time.time() >= expires_at:
            raise ServiceError(410, "CONVERSATION_EXPIRED", "對話在處理期間已過期，請開始新對話。")
        # No state or replay is committed on model failure. Both writes are atomic.
        with self.store.connect() as db:
            db.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET version=excluded.version, state=excluded.state",
                (str(conversation_id), response.version, json.dumps(state, ensure_ascii=False), expires_at),
            )
            db.execute("INSERT INTO requests VALUES (?, ?, ?, ?, ?)", (
                str(request.request_id), fingerprint, str(conversation_id), response.model_dump_json(), expires_at,
            ))
        return response
