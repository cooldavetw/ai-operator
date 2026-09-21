"""Optional real-GGUF integration check, never silently replaced with a fake."""
import os
from pathlib import Path

import pytest

from app.config import load_config
from app.runtime import LlamaRuntime


@pytest.mark.model
@pytest.mark.skipif(not os.environ.get("TEST_MODEL_PATH"), reason="Set TEST_MODEL_PATH to a vetted local GGUF")
def test_real_cpu_model(settings):
    settings.model_path = Path(os.environ["TEST_MODEL_PATH"])
    settings.chat_format = os.environ.get("CHAT_FORMAT") or None
    settings.inference_timeout_seconds = 120
    runtime = LlamaRuntime(settings)
    try:
        runtime.start()
        decision = runtime.infer(
            [{"role": "user", "content": "我借的那本書還沒讀完，要延長借閱時間"}],
            load_config(settings.config_path),
        )
        assert decision.decision == "CLASSIFY"
        assert decision.domain == "LIBRARY_SYSTEM"
    finally:
        runtime.close()

