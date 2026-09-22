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
        for message in [
            "請提供番茄炒蛋的食譜。",
            "請教我怎麼煮番茄炒蛋，與學校、選課、成績或圖書館無關。",
            "請幫我規劃東京五天的觀光行程。",
        ]:
            decision = runtime.infer(
                [{"role": "user", "content": message}],
                load_config(settings.config_path),
            )
            assert decision.decision == "UNKNOWN", (message, decision)
            assert decision.reason == "OUT_OF_SCOPE"
            assert decision.domain is None
    finally:
        runtime.close()

