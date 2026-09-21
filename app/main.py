import asyncio
import hmac
import logging
import sqlite3
import time
from contextlib import asynccontextmanager, suppress

from fastapi import Depends, FastAPI, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from app.config import Settings, load_config
from app.errors import ServiceError
from app.runtime import LlamaRuntime, Runtime
from app.schemas import ClassifyRequest, ClassifyResponse
from app.service import Classifier
from app.store import Store

logger = logging.getLogger("classifier")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class BodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """Bound actual streamed bytes, including requests without Content-Length."""

    def __init__(self, app, limit=16384):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > self.limit:
                    raise BodyTooLarge()
            return message

        # Buffer only the bounded body so framework parsing cannot turn our
        # size exception into an unrelated JSON parsing error.
        chunks = []
        try:
            while True:
                message = await bounded_receive()
                if message["type"] == "http.disconnect":
                    return
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
        except BodyTooLarge:
            return await JSONResponse(
                {"error": {"code": "BODY_TOO_LARGE", "message": "請求內容超過 16 KiB。"}},
                status_code=413,
            )(scope, receive, send)
        delivered = False

        async def buffered_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        return await self.app(scope, buffered_receive, send)


class Admission:
    """One running inference plus a bounded queue; cancellation never frees a busy worker."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = asyncio.Lock()
        self.count = 0
        self.tasks = set()

    async def run(self, function, argument):
        if self.count >= self.settings.queue_capacity + 1:
            raise ServiceError(429, "QUEUE_FULL", "服務忙碌中，請稍後重試。")
        self.count += 1
        try:
            await asyncio.wait_for(self.lock.acquire(), self.settings.queue_timeout_seconds)
        except asyncio.TimeoutError as exc:
            self.count -= 1
            raise ServiceError(429, "QUEUE_TIMEOUT", "排隊逾時，請稍後重試。") from exc
        except BaseException:
            self.count -= 1
            raise

        task = asyncio.create_task(asyncio.to_thread(function, argument))
        self.tasks.add(task)

        def completed(future):
            self.tasks.discard(future)
            self.count -= 1
            self.lock.release()
            # Retrieve exceptions even if the HTTP caller disconnected.
            if not future.cancelled():
                future.exception()

        task.add_done_callback(completed)
        return await asyncio.shield(task)

    async def drain(self):
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    settings = settings or Settings.from_env()
    config = load_config(settings.config_path)
    runtime = runtime or LlamaRuntime(settings)
    store = Store(settings.database_path)
    classifier = Classifier(settings, config, runtime, store)

    async def maintenance():
        while True:
            await asyncio.sleep(60)
            try:
                await asyncio.to_thread(store.purge, time.time() - settings.expired_retention_seconds)
            except sqlite3.Error:
                logger.error("database_cleanup_failed")

    @asynccontextmanager
    async def lifespan(app):
        store.open()
        app.state.admission = Admission(settings)
        cleanup = None
        try:
            await asyncio.to_thread(store.purge, time.time() - settings.expired_retention_seconds)
            try:
                await asyncio.to_thread(runtime.start)
            except ServiceError as exc:
                logger.error("model_start_failed code=%s", exc.code)
            cleanup = asyncio.create_task(maintenance())
            yield
        finally:
            if cleanup:
                cleanup.cancel()
                with suppress(asyncio.CancelledError):
                    await cleanup
            await app.state.admission.drain()
            await asyncio.to_thread(runtime.close)
            store.close()

    app = FastAPI(
        title="海洋大學地端 AI 總機分類服務", version="0.1.0",
        description="本機分類與最多兩次澄清；不回答業務問題。",
        lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None,
    )
    app.add_middleware(BodyLimitMiddleware)
    app.state.classifier = classifier

    async def authorize(key: str | None = Security(api_key_header)):
        if key is None or not hmac.compare_digest(key.encode(), settings.api_key.get_secret_value().encode()):
            raise ServiceError(401, "UNAUTHORIZED", "需要有效的 X-API-Key。")

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        headers = {"Retry-After": "1"} if exc.status in {429, 503} else None
        return JSONResponse(
            {"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status, headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # FastAPI's default validation response echoes user input. Keep it private.
        return JSONResponse({"error": {
            "code": "INVALID_REQUEST", "message": "請求格式不正確。",
            "fields": [{"loc": e["loc"], "type": e["type"]} for e in exc.errors()],
        }}, status_code=422)

    @app.exception_handler(sqlite3.Error)
    async def database_error(request: Request, exc: sqlite3.Error):
        logger.error("database_operation_failed")
        return JSONResponse({"error": {"code": "STORAGE_UNAVAILABLE", "message": "狀態儲存服務暫時無法使用。"}}, status_code=503)

    @app.get("/health/live", tags=["health"])
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"])
    async def ready():
        def database_ready():
            with store.connect() as db:
                db.execute("SELECT 1 FROM conversations LIMIT 1")
        try:
            await asyncio.to_thread(database_ready)
            available = runtime.ready
        except sqlite3.Error:
            available = False
        return JSONResponse({"status": "ready" if available else "not_ready"}, status_code=200 if available else 503)

    @app.get("/openapi.json", dependencies=[Depends(authorize)], include_in_schema=False)
    async def openapi():
        return app.openapi()

    @app.post(
        "/api/v1/classify", response_model=ClassifyResponse,
        dependencies=[Depends(authorize)], tags=["classification"],
        responses={code: {"description": text} for code, text in {
            401: "Invalid API key", 404: "Conversation not found", 409: "State or request conflict",
            410: "Conversation expired", 413: "Body too large", 422: "Invalid request or context too long",
            429: "Queue full or timed out", 502: "Invalid model output",
            503: "Model or storage unavailable", 504: "Inference timed out",
        }.items()},
    )
    async def classify(request: ClassifyRequest):
        started = time.monotonic()
        try:
            result = await app.state.admission.run(classifier.classify, request)
            logger.info("classification request_id=%s status=%s domain=%s elapsed_ms=%.0f",
                        request.request_id, result.status, result.domain, (time.monotonic() - started) * 1000)
            return result
        except ServiceError as exc:
            logger.warning("classification_failed request_id=%s code=%s", request.request_id, exc.code)
            raise

    return app
