"""Tests for OpenAI LLM and TTS plugins with mocked OpenAI client."""
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from inference.core.types import PluginConfig
from inference.plugins.llm.openai_plugin import OpenAILLMPlugin, SENTENCE_ENDERS
from inference.plugins.tts.openai_tts_plugin import OpenAITTSPlugin


# --- LLM Plugin Tests ---

class TestOpenAILLMPlugin:
    def test_name(self):
        assert OpenAILLMPlugin.name == "llm.openai"

    def test_sentence_enders(self):
        assert "." in SENTENCE_ENDERS
        assert "。" in SENTENCE_ENDERS
        assert "!" in SENTENCE_ENDERS

    @pytest.mark.asyncio
    async def test_generate_stream_with_mock(self):
        plugin = OpenAILLMPlugin()
        plugin.client = MagicMock()
        plugin.model = "gpt-4o"
        plugin.temperature = 0.7
        plugin.system_prompt = "You are helpful."

        # Mock streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world."

        async def mock_stream():
            yield mock_chunk1
            yield mock_chunk2

        plugin.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [{"role": "user", "content": "Hi"}]
        results = []
        async for chunk in plugin.generate_stream(messages):
            results.append(chunk)

        assert len(results) == 3  # 2 tokens + 1 final
        assert results[0].token == "Hello"
        assert results[0].is_sentence_end is False
        assert results[1].token == " world."
        assert results[1].is_sentence_end is True  # ends with "."
        assert results[2].is_final is True
        assert results[2].accumulated_text == "Hello world."

    @pytest.mark.asyncio
    async def test_system_prompt_prepended(self):
        plugin = OpenAILLMPlugin()
        plugin.client = MagicMock()
        plugin.system_prompt = "Be concise."

        async def empty_stream():
            return
            yield  # make it an async generator

        plugin.client.chat.completions.create = AsyncMock(return_value=empty_stream())

        messages = [{"role": "user", "content": "Hi"}]
        async for _ in plugin.generate_stream(messages):
            pass

        call_args = plugin.client.chat.completions.create.call_args
        sent_messages = call_args.kwargs["messages"]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[0]["content"] == "Be concise."

    def test_formats_image_messages_for_chat_completions(self):
        message = OpenAILLMPlugin._format_message({
            "role": "user",
            "content": "What is visible?",
            "images": [
                {
                    "data": b"\xff\xd8\xff",
                    "mime_type": "image/jpeg",
                }
            ],
        })

        assert message["role"] == "user"
        assert message["content"][0] == {"type": "text", "text": "What is visible?"}
        assert message["content"][1]["type"] == "image_url"
        assert message["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,/9j/"

    @pytest.mark.asyncio
    async def test_shutdown(self):
        plugin = OpenAILLMPlugin()
        plugin.client = MagicMock()
        await plugin.shutdown()
        assert plugin.client is None


# --- TTS Plugin Tests ---

class TestOpenAITTSPlugin:
    def test_name(self):
        assert OpenAITTSPlugin.name == "tts.openai"

    def test_resample(self):
        audio = np.sin(np.linspace(0, 2 * np.pi, 24000)).astype(np.float32)
        resampled = OpenAITTSPlugin._resample(audio, 24000, 16000)
        # 24000 * (2/3) = 16000 exact
        assert len(resampled) == 16000
        assert resampled.dtype == np.float32

    def test_resample_same_rate(self):
        audio = np.ones(16000, dtype=np.float32)
        resampled = OpenAITTSPlugin._resample(audio, 16000, 16000)
        assert len(resampled) == 16000
        np.testing.assert_array_equal(resampled, audio)

    def test_resample_preserves_frequency(self):
        """Polyphase resampling should preserve dominant frequency content."""
        freq = 440  # Hz
        sr_orig = 24000
        t = np.arange(sr_orig) / sr_orig
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
        resampled = OpenAITTSPlugin._resample(audio, sr_orig, 16000)
        # Energy should be preserved approximately
        assert abs(np.std(resampled) - np.std(audio)) < 0.1

    @pytest.mark.asyncio
    async def test_synthesize_stream_with_mock(self):
        plugin = OpenAITTSPlugin()
        plugin.client = MagicMock()
        plugin.model = "tts-1"
        plugin.voice = "nova"
        plugin._openai_sample_rate = 24000
        plugin.rechunker.chunk_samples = 1000  # small for test

        # Mock TTS response: 3000 samples at 24kHz -> resampled to 2000 at 16kHz
        fake_audio = np.zeros(3000, dtype=np.int16)
        mock_response = MagicMock()
        mock_response.content = fake_audio.tobytes()
        plugin.client.audio.speech.create = AsyncMock(return_value=mock_response)

        async def text_stream():
            yield "Hello world."

        results = []
        async for chunk in plugin.synthesize_stream(text_stream()):
            results.append(chunk)

        # 3000 samples at 24kHz -> 2000 at 16kHz -> 2 chunks of 1000
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_synthesize_skips_empty(self):
        plugin = OpenAITTSPlugin()
        plugin.client = MagicMock()
        plugin.rechunker.chunk_samples = 1000

        async def text_stream():
            yield ""
            yield "   "

        results = []
        async for chunk in plugin.synthesize_stream(text_stream()):
            results.append(chunk)

        assert len(results) == 0
        plugin.client.audio.speech.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_handles_api_error(self):
        """TTS should skip sentences on API error, not crash."""
        plugin = OpenAITTSPlugin()
        plugin.client = MagicMock()
        plugin.rechunker.chunk_samples = 1000
        plugin.client.audio.speech.create = AsyncMock(side_effect=Exception("rate limited"))

        async def text_stream():
            yield "Hello world."

        results = []
        async for chunk in plugin.synthesize_stream(text_stream()):
            results.append(chunk)

        assert len(results) == 0  # API error skipped, no audio produced

    @pytest.mark.asyncio
    async def test_shutdown(self):
        plugin = OpenAITTSPlugin()
        plugin.client = MagicMock()
        await plugin.shutdown()
        assert plugin.client is None
