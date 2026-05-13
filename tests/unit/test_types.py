import numpy as np

from inference.core.types import (
    AudioChunk,
    LLMResponseChunk,
    PluginConfig,
    TranscriptEvent,
    VideoChunk,
    VoiceLLMInputEvent,
    VoiceLLMSessionConfig,
)


def test_audio_chunk_defaults():
    chunk = AudioChunk(data=b"\x00" * 100)
    assert chunk.sample_rate == 16000
    assert chunk.channels == 1
    assert chunk.format == "float32"
    assert chunk.is_final is False


def test_video_chunk():
    frames = np.zeros((28, 512, 512, 3), dtype=np.uint8)
    chunk = VideoChunk(frames=frames, fps=25, chunk_index=1)
    assert chunk.frames.shape == (28, 512, 512, 3)
    assert chunk.fps == 25
    assert chunk.is_final is False


def test_transcript_event():
    event = TranscriptEvent(text="hello", is_final=True, confidence=0.95)
    assert event.text == "hello"
    assert event.is_final is True


def test_llm_response_chunk():
    chunk = LLMResponseChunk(token="world", accumulated_text="hello world", is_sentence_end=True)
    assert chunk.token == "world"
    assert chunk.is_sentence_end is True
    assert chunk.is_final is False


def test_plugin_config():
    config = PluginConfig(plugin_name="test", params={"key": "value"})
    assert config.plugin_name == "test"
    assert config.params["key"] == "value"


def test_plugin_config_default_params():
    config = PluginConfig(plugin_name="test")
    assert config.params == {}
    assert config.shared == {}


def test_plugin_config_shared():
    config = PluginConfig(
        plugin_name="test",
        shared={"warmup": {"enabled": True}},
    )
    assert config.shared["warmup"]["enabled"] is True


def test_voice_llm_input_response_instructions_default():
    event = VoiceLLMInputEvent()
    assert event.response_instructions is None


def test_voice_llm_session_can_defer_response():
    config = VoiceLLMSessionConfig(defer_response=True)
    assert config.defer_response is True
