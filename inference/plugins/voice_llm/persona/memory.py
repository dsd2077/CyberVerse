from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_ENV_PLACEHOLDER_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")
_RECALL_QUERY_PUNCTUATION_RE = re.compile(r"""[\u3000\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65？。、《》“”‘’、；：！…（）【】?!,;:()[\]{}"']+""")


def _clean_config_string(value: Any) -> str:
    text = str(value or "").strip()
    if _ENV_PLACEHOLDER_RE.match(text):
        return ""
    return text


def _optional_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _optional_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.1, parsed)


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit]


def _normalize_recall_query(value: Any) -> str:
    text = str(value or "").strip()
    text = _RECALL_QUERY_PUNCTUATION_RE.sub(" ", text)
    return " ".join(text.split())


def _fallback_memory_path(value: Any) -> str:
    configured = _clean_config_string(value)
    if configured:
        return configured
    return os.path.join(os.getcwd(), "data", "persona_memory_shadow.jsonl")


def _keyword_terms(value: str) -> set[str]:
    text = _normalize_recall_query(value).lower()
    terms = set(re.findall(r"[a-z0-9_-]{3,}", text))
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.add(run)
        for size in (2, 3, 4):
            if len(run) >= size:
                terms.update(run[index : index + size] for index in range(0, len(run) - size + 1))
    return {term for term in terms if term.strip()}


def _explicit_memory_write(value: str) -> bool:
    text = str(value or "").strip().lower()
    return (
        "用户: 请记住" in text
        or "用户: 记住" in text
        or "user: remember" in text
        or "user: please remember" in text
    )


def _persona_hindsight_params(runtime_config: dict[str, Any] | None) -> dict[str, Any]:
    inference = runtime_config.get("inference", {}) if isinstance(runtime_config, dict) else {}
    inference = inference if isinstance(inference, dict) else {}
    persona_agent = inference.get("persona_agent", {})
    persona_agent = persona_agent if isinstance(persona_agent, dict) else {}
    persona_section = inference.get("persona", {})
    persona_section = persona_section if isinstance(persona_section, dict) else {}
    persona_plugin = persona_section.get("persona", {})
    persona_plugin = persona_plugin if isinstance(persona_plugin, dict) else {}

    for owner in (persona_agent, persona_plugin):
        memory = owner.get("memory", {}) if isinstance(owner, dict) else {}
        memory = memory if isinstance(memory, dict) else {}
        hindsight = memory.get("hindsight", {})
        if isinstance(hindsight, dict) and hindsight:
            return hindsight
    return {}


@dataclass(frozen=True)
class HindsightMemoryConfig:
    enabled: bool = True
    base_url: str = "https://hindsight.lucky.jmsu.top"
    api_key: str = ""
    bank_id: str = "openclaw"
    user_tag: str = ""
    timeout_seconds: float = 30.0
    recall_max_results: int = 5
    recall_max_tokens: int = 4096
    retain_max_chars: int = 6000
    local_fallback_enabled: bool = True
    local_fallback_path: str = ""


def hindsight_config_from_runtime_config(runtime_config: dict[str, Any] | None = None) -> HindsightMemoryConfig:
    params = _persona_hindsight_params(runtime_config)
    enabled_value = params.get("enabled") if "enabled" in params else os.getenv("HINDSIGHT_ENABLED")
    timeout_value = params.get("timeout_seconds") if "timeout_seconds" in params else os.getenv("HINDSIGHT_TIMEOUT_SECONDS")
    max_results_value = (
        params.get("recall_max_results")
        if "recall_max_results" in params
        else os.getenv("HINDSIGHT_RECALL_MAX_RESULTS")
    )
    max_tokens_value = (
        params.get("recall_max_tokens")
        if "recall_max_tokens" in params
        else os.getenv("HINDSIGHT_RECALL_MAX_TOKENS")
    )
    max_chars_value = (
        params.get("retain_max_chars")
        if "retain_max_chars" in params
        else os.getenv("HINDSIGHT_RETAIN_MAX_CHARS")
    )
    fallback_enabled_value = (
        params.get("local_fallback_enabled")
        if "local_fallback_enabled" in params
        else os.getenv("HINDSIGHT_LOCAL_FALLBACK_ENABLED")
    )
    fallback_path_value = (
        params.get("local_fallback_path")
        if "local_fallback_path" in params
        else os.getenv("HINDSIGHT_LOCAL_FALLBACK_PATH")
    )

    return HindsightMemoryConfig(
        enabled=_optional_bool(_clean_config_string(enabled_value), True),
        base_url=(
            _clean_config_string(params.get("base_url"))
            or _clean_config_string(os.getenv("HINDSIGHT_BASE_URL"))
            or "https://hindsight.lucky.jmsu.top"
        ).rstrip("/"),
        api_key=_clean_config_string(params.get("api_key")) or _clean_config_string(os.getenv("HINDSIGHT_API_KEY")),
        bank_id=(
            _clean_config_string(params.get("bank_id"))
            or _clean_config_string(os.getenv("HINDSIGHT_BANK_ID"))
            or "openclaw"
        ),
        user_tag=_clean_config_string(params.get("user_tag")) or _clean_config_string(os.getenv("HINDSIGHT_USER_TAG")),
        timeout_seconds=_optional_float(timeout_value, 30.0),
        recall_max_results=_bounded_int(max_results_value, 5, 1, 20),
        recall_max_tokens=_bounded_int(max_tokens_value, 4096, 256, 32768),
        retain_max_chars=_bounded_int(max_chars_value, 6000, 500, 50000),
        local_fallback_enabled=_optional_bool(_clean_config_string(fallback_enabled_value), True),
        local_fallback_path=_fallback_memory_path(fallback_path_value),
    )


def _memory_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    for key in ("text", "content"):
        text = _clean_config_string(value.get(key))
        if text:
            return text
    for key in ("memory", "item", "document"):
        nested = value.get(key)
        text = _memory_text(nested)
        if text:
            return text
    return ""


def format_memories(memories: list[dict[str, Any]]) -> str:
    lines = []
    for memory in memories:
        text = _memory_text(memory)
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


class HindsightMemoryClient:
    def __init__(self, config: HindsightMemoryConfig, http_client: Any | None = None) -> None:
        self.config = config
        self.http_client = http_client

    @property
    def enabled(self) -> bool:
        return bool(
            self.config.enabled
            and self.config.base_url
            and self.config.api_key
            and self.config.bank_id
            and self.config.user_tag
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, suffix: str) -> str:
        return f"{self.config.base_url}/v1/default/banks/{self.config.bank_id}/memories{suffix}"

    def _local_fallback_enabled(self) -> bool:
        return bool(
            self.config.local_fallback_enabled
            and self.config.local_fallback_path
            and self.http_client is None
        )

    def _append_local_memory(self, content: str, context: str) -> None:
        if not self._local_fallback_enabled():
            return
        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "tags": [self.config.user_tag] if self.config.user_tag else [],
            "content": content,
        }
        path = self.config.local_fallback_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Hindsight local fallback retain failed: %s", exc)

    def _recall_local_memory(self, query: str) -> list[dict[str, Any]]:
        if not self._local_fallback_enabled():
            return []
        path = self.config.local_fallback_path
        if not os.path.exists(path):
            return []

        query_terms = _keyword_terms(query)
        if not query_terms:
            return []

        matches: list[tuple[int, int, str]] = []
        explicit_matches: list[tuple[int, str]] = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()[-500:]
        except Exception as exc:
            logger.warning("Hindsight local fallback recall failed: %s", exc)
            return []

        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = _memory_text(record.get("content"))
            if not content:
                continue
            terms = _keyword_terms(content)
            score = len(query_terms & terms)
            if not score:
                continue
            is_explicit = _explicit_memory_write(content)
            if is_explicit:
                score += 100
            matches.append((score, index, content))
            if is_explicit:
                explicit_matches.append((index, content))

        if explicit_matches:
            _index, content = sorted(explicit_matches, reverse=True)[0]
            return [{"text": _clip_text(content, 1200)}]
        memories: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _score, _index, content in sorted(matches, reverse=True):
            text = _clip_text(content, 1200)
            if text in seen:
                continue
            seen.add(text)
            memories.append({"text": text})
            if len(memories) >= self.config.recall_max_results:
                break
        return memories

    def _merge_memories(self, *groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        memories: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in groups:
            for memory in group:
                text = _memory_text(memory)
                if not text or text in seen:
                    continue
                seen.add(text)
                memories.append({"text": text})
                if len(memories) >= self.config.recall_max_results:
                    return memories
        return memories

    async def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        client = self.http_client
        if client is not None:
            response = await client.post(url, json=payload, headers=self._headers())
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}

        import httpx

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as http_client:
            response = await http_client.post(url, json=payload, headers=self._headers())
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}

    async def recall(self, query: str) -> list[dict[str, Any]]:
        query = _normalize_recall_query(query)
        if not self.enabled or not query:
            return []
        local_memories = self._recall_local_memory(query)
        if local_memories and _explicit_memory_write(local_memories[0].get("text", "")):
            return local_memories
        payload: dict[str, Any] = {
            "query": query,
            "types": ["world", "experience"],
            "budget": "mid",
            "max_tokens": self.config.recall_max_tokens,
            "tags": [self.config.user_tag],
            "tags_match": "any",
        }
        try:
            data = await self._post_json(self._url("/recall"), payload)
        except Exception as exc:
            logger.warning("Hindsight recall failed: %s", exc)
            return local_memories

        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            return local_memories
        memories: list[dict[str, Any]] = []
        for result in raw_results[: self.config.recall_max_results]:
            text = _memory_text(result)
            if text:
                memories.append({"text": text})
        return self._merge_memories(local_memories, memories)

    async def retain(self, content: str, context: str = "conversation") -> dict[str, Any]:
        content = _clip_text(content, self.config.retain_max_chars)
        context = str(context or "conversation").strip() or "conversation"
        if not self.enabled or not content:
            return {"ok": False}
        self._append_local_memory(content, context)
        item: dict[str, Any] = {
            "content": content,
            "context": context,
            "tags": [self.config.user_tag],
        }
        payload = {"items": [item], "async": True}
        try:
            return await self._post_json(self._url(""), payload)
        except Exception as exc:
            logger.warning("Hindsight retain failed: %s", exc)
            return {"ok": False}
