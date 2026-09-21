from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import Decision

ROOT = Path(__file__).resolve().parents[1]
KEY = "unit-test-key-with-at-least-24-characters"


def clarify(question="identify_system", reason="INSUFFICIENT_INFORMATION"):
    return Decision(decision="CLARIFY", domain=None, clarification_id=question, reason=reason)


def matched(domain="LIBRARY_SYSTEM"):
    return Decision(decision="CLASSIFY", domain=domain, clarification_id=None, reason="MATCH")


class FakeRuntime:
    def __init__(self, decisions=None):
        self.decisions = list(decisions or [])
        self.calls = []
        self.ready = False

    def start(self):
        self.ready = True

    def close(self):
        self.ready = False

    def infer(self, history, config):
        self.calls.append((history, config))
        result = self.decisions.pop(0) if self.decisions else clarify()
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def settings(tmp_path):
    config = tmp_path / "domains.yaml"
    config.write_text((ROOT / "config/domains.yaml").read_text(), encoding="utf-8")
    return Settings(api_key=KEY, config_path=config, database_path=tmp_path / "state.sqlite3")


@pytest.fixture
def runtime():
    return FakeRuntime()


@pytest.fixture
def client(settings, runtime):
    with TestClient(create_app(settings, runtime), headers={"X-API-Key": KEY}) as client:
        yield client

