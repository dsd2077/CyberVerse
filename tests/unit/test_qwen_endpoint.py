from inference.plugins.qwen_endpoint import (
    dashscope_base_url,
    dashscope_realtime_ws_url,
)


def test_dashscope_base_url_reads_env(monkeypatch):
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    assert dashscope_base_url() == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_realtime_ws_url_derived_from_base_url(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_WS_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_TTS_WS_URL", raising=False)
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    assert (
        dashscope_realtime_ws_url("qwen3-tts-flash-realtime", "DASHSCOPE_TTS_WS_URL")
        == "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-tts-flash-realtime"
    )


def test_realtime_ws_url_uses_service_env_and_model_query(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_WS_URL", "wss://generic.example.com/realtime")
    monkeypatch.setenv(
        "DASHSCOPE_ASR_WS_URL",
        "wss://asr.example.com/realtime?foo=bar&model=old",
    )

    assert (
        dashscope_realtime_ws_url("qwen3-asr-flash-realtime", "DASHSCOPE_ASR_WS_URL")
        == "wss://asr.example.com/realtime?foo=bar&model=qwen3-asr-flash-realtime"
    )
