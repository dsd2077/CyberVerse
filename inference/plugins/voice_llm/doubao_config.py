from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference.core.types import PluginConfig, VoiceLLMSessionConfig

logger = logging.getLogger(__name__)

# SC2.0 official voices: Chinese display name -> speaker_id
SC20_VOICES: dict[str, str] = {
    "傲娇女友": "saturn_zh_female_aojiaonvyou_tob",
    "冰娇姐姐": "saturn_zh_female_bingjiaojiejie_tob",
    "成熟姐姐": "saturn_zh_female_chengshujiejie_tob",
    "可爱女生": "saturn_zh_female_keainvsheng_tob",
    "暖心学姐": "saturn_zh_female_nuanxinxuejie_tob",
    "贴心女友": "saturn_zh_female_tiexinnvyou_tob",
    "温柔文雅": "saturn_zh_female_wenrouwenya_tob",
    "妩媚御姐": "saturn_zh_female_wumeiyujie_tob",
    "性感御姐": "saturn_zh_female_xingganyujie_tob",
    "爱气凌人": "saturn_zh_male_aiqilingren_tob",
    "傲娇公子": "saturn_zh_male_aojiaogongzi_tob",
    "傲娇精英": "saturn_zh_male_aojiaojingying_tob",
    "傲慢少爷": "saturn_zh_male_aomanshaoye_tob",
    "霸道少爷": "saturn_zh_male_badaoshaoye_tob",
    "冰娇白莲": "saturn_zh_male_bingjiaobailian_tob",
    "不羁青年": "saturn_zh_male_bujiqingnian_tob",
    "成熟总裁": "saturn_zh_male_chengshuzongcai_tob",
    "磁性男嗓": "saturn_zh_male_cixingnansang_tob",
    "醋精男友": "saturn_zh_male_cujingnanyou_tob",
    "风发少年": "saturn_zh_male_fengfashaonian_tob",
    "腹黑公子": "saturn_zh_male_fuheigongzi_tob",
}


@dataclass
class DoubaoSessionConfig:
    """Configuration for Doubao realtime session."""

    access_token: str
    app_id: str
    ws_url: str = "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"
    voice_type: str = "温柔文雅"
    bot_name: str = "豆包"
    system_prompt: str = ""
    speaking_style: str = "你的说话风格简洁明了，语速适中，语调自然。"
    model: str = "2.2.0.0"
    end_smooth_window_ms: int = 1500
    say_hello_content: str = ""
    recv_timeout: int = 120
    input_mod: str = "keep_alive"
    conversation_id: str = ""
    output_sample_rate: int = 24000
    output_audio_format: str = "pcm_s16le"
    compression: str = "gzip"
    max_retries: int = 3
    retry_backoff_base: float = 1.0
    retry_backoff_max: float = 10.0
    dialog_context: list[dict] = field(default_factory=list)

    @classmethod
    def from_plugin_config(cls, config: "PluginConfig") -> "DoubaoSessionConfig":
        """
        Create DoubaoSessionConfig from PluginConfig.

        Args:
            config: PluginConfig instance with params dict

        Returns:
            DoubaoSessionConfig instance

        Raises:
            ValueError: If required fields (access_token, ws_url) are missing or empty
        """
        # Extract token - prefer access_token, fallback to api_key
        token = config.params.get("access_token", "") or config.params.get("api_key", "")
        app_id = config.params.get("app_id", "")
        ws_url = config.params.get("ws_url", "wss://openspeech.bytedance.com/api/v3/realtime/dialogue")

        # Validate required fields
        if not token:
            raise ValueError("access_token (or api_key) is required but not provided")
        if not ws_url:
            raise ValueError("ws_url is required but not provided")

        # Extract other config values with defaults
        voice_type = config.params.get("voice_type", "温柔文雅")
        bot_name = config.params.get("bot_name", "豆包")
        system_prompt = config.params.get("system_prompt", "")
        speaking_style = config.params.get("speaking_style", "你的说话风格简洁明了，语速适中，语调自然。")
        model = config.params.get("model", "2.2.0.0")
        say_hello_content = str(config.params.get("say_hello_content", "") or "")
        end_smooth_window_ms = int(config.params.get("end_smooth_window_ms", 1500))
        recv_timeout = int(config.params.get("recv_timeout", 120))
        input_mod = config.params.get("input_mod", "keep_alive")
        output_sample_rate = int(config.params.get("output_sample_rate", 24000))
        output_audio_format = config.params.get("output_audio_format", "pcm_s16le")
        compression = str(config.params.get("compression", "gzip")).lower()
        max_retries = int(config.params.get("max_retries", 3))
        retry_backoff_base = float(config.params.get("retry_backoff_base", 1.0))
        retry_backoff_max = float(config.params.get("retry_backoff_max", 10.0))

        return cls(
            access_token=token,
            app_id=app_id,
            ws_url=ws_url,
            voice_type=voice_type,
            bot_name=bot_name,
            system_prompt=system_prompt,
            speaking_style=speaking_style,
            model=model,
            end_smooth_window_ms=end_smooth_window_ms,
            say_hello_content=say_hello_content,
            recv_timeout=recv_timeout,
            input_mod=input_mod,
            output_sample_rate=output_sample_rate,
            output_audio_format=output_audio_format,
            compression=compression,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

    def with_overrides(self, session_config: VoiceLLMSessionConfig) -> DoubaoSessionConfig:
        """Return a new config with session overrides applied.

        YAML defaults are preserved for any field not provided by the session.
        Welcome message is special-cased so an explicit empty string disables the
        default greeting for that session.
        """
        overrides: dict[str, object] = {}
        if session_config.voice:
            overrides["voice_type"] = session_config.voice
        if session_config.bot_name:
            overrides["bot_name"] = session_config.bot_name
        if session_config.system_prompt:
            overrides["system_prompt"] = session_config.system_prompt
        if session_config.speaking_style:
            overrides["speaking_style"] = session_config.speaking_style
        if session_config.welcome_message is not None:
            overrides["say_hello_content"] = session_config.welcome_message
        if session_config.input_mode:
            overrides["input_mod"] = session_config.input_mode
        if session_config.session_id:
            overrides["conversation_id"] = session_config.session_id
        if session_config.dialog_context:
            overrides["dialog_context"] = [
                {
                    "role": item.role,
                    "text": item.text,
                    "timestamp": item.timestamp,
                }
                for item in session_config.dialog_context
            ]
        if not overrides:
            return self
        result = replace(self, **overrides)
        logger.debug("DoubaoSessionConfig overrides applied: %s", overrides)
        return result

    @property
    def has_welcome_message(self) -> bool:
        return bool(self.say_hello_content.strip())

    def build_ws_headers(self, connect_id: str) -> dict[str, str]:
        """
        Build WebSocket connection headers.

        Args:
            connect_id: Unique connection identifier (UUID)

        Returns:
            Dict of HTTP headers for WebSocket connection
        """
        return {
            "X-Api-App-ID": self.app_id,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": "volc.speech.dialog",
            "X-Api-App-Key": "PlgvMymc7f3tQnJ6",
            "X-Api-Connect-Id": connect_id,
        }

    def build_start_session_payload(self, dialog_id: str | None = None) -> dict:
        """
        Build the start_session request payload.

        Returns:
            Dict containing asr, tts, and dialog configuration
        """
        speaker = SC20_VOICES.get(self.voice_type, self.voice_type)
        logger.debug(
            "Doubao TTS speaker resolved: voice_type=%r -> speaker=%r (in SC20_VOICES: %s)",
            self.voice_type,
            speaker,
            self.voice_type in SC20_VOICES,
        )
        dialog: dict = {
            "character_manifest": self.build_character_manifest(),
            "extra": {
                "strict_audit": False,
                "recv_timeout": self.recv_timeout,
                "input_mod": self.input_mod,
                "model": self.model,
            },
        }
        if dialog_id:
            dialog["dialog_id"] = dialog_id
        if self.dialog_context:
            dialog["dialog_context"] = self.dialog_context
        return {
            "asr": {
                "extra": {
                    "end_smooth_window_ms": self.end_smooth_window_ms,
                },
            },
            "tts": {
                "speaker": speaker,
                "audio_config": {
                    "channel": 1,
                    "format": self.output_audio_format,
                    "sample_rate": self.output_sample_rate,
                },
            },
            "dialog": dialog,
        }

    def build_say_hello_payload(self) -> dict:
        """
        Build the say_hello request payload.

        Returns:
            Dict with greeting content
        """
        return {
            "content": self.say_hello_content,
        }

    def build_character_manifest(self) -> str:
        """
        Build character_manifest for SC2.0 from the three O-version persona fields.

        Combines bot_name, system_prompt (role background), and speaking_style
        into a single free-form description string that SC2.0 accepts.

        Returns:
            Formatted character manifest string
        """
        parts: list[str] = []
        if self.bot_name:
            parts.append(f"名字：{self.bot_name}")
        if self.system_prompt:
            parts.append(self.system_prompt)
        if self.speaking_style:
            parts.append(f"说话风格：{self.speaking_style}")
        return "\n".join(parts)


    @property
    def compression_bits(self) -> int:
        """
        Get compression bits for protocol header.

        Returns:
            COMPRESSION_GZIP or COMPRESSION_NONE
        """
        from inference.plugins.voice_llm.doubao_protocol import COMPRESSION_GZIP, COMPRESSION_NONE
        return COMPRESSION_GZIP if self.compression == "gzip" else COMPRESSION_NONE
