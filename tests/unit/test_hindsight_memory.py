import pytest

from inference.plugins.voice_llm.persona.memory import HindsightMemoryClient, hindsight_config_from_runtime_config


class FakeResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload or {}
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeHTTPClient:
    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.posts = []

    async def post(self, url, json, headers):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return self.response


class FakeRemoteHindsightClient(HindsightMemoryClient):
    def __init__(self, config, recall_payload=None, error=None):
        super().__init__(config)
        self.recall_payload = recall_payload or {"results": []}
        self.error = error
        self.posts = []

    async def _post_json(self, url, payload):
        self.posts.append({"url": url, "payload": payload})
        if self.error:
            raise self.error
        if url.endswith("/recall"):
            return self.recall_payload
        return {"ok": True}


def test_hindsight_config_reads_env_and_cleans_placeholders(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_KEY", "env-key")
    monkeypatch.setenv("HINDSIGHT_USER_TAG", "env-user")

    config = hindsight_config_from_runtime_config(
        {
            "inference": {
                "persona": {
                    "persona": {
                        "memory": {
                            "hindsight": {
                                "enabled": True,
                                "base_url": "${HINDSIGHT_BASE_URL}",
                                "api_key": "${HINDSIGHT_API_KEY}",
                                "bank_id": "${HINDSIGHT_BANK_ID}",
                                "user_tag": "${HINDSIGHT_USER_TAG}",
                            }
                        }
                    }
                }
            }
        }
    )

    assert config.enabled is True
    assert config.base_url == "https://hindsight.jmsu.top"
    assert config.api_key == "env-key"
    assert config.bank_id == ""
    assert config.bank_id_template == "cv:user:{user_id}:character:{character_id}"
    assert config.user_tag == "env-user"


@pytest.mark.asyncio
async def test_hindsight_recall_sends_expected_payload():
    http = FakeHTTPClient(
        FakeResponse(
            {
                "results": [
                    {"text": "用户喜欢 Pixi。"},
                    {"memory": {"content": "用户在 macOS 上开发。"}},
                ]
            }
        )
    )
    client = HindsightMemoryClient(
        hindsight_config_from_runtime_config(
            {
                "inference": {
                    "persona_agent": {
                        "memory": {
                            "hindsight": {
                                "api_key": "test-key",
                                "user_tag": "user-1",
                                "recall_max_results": 2,
                            }
                        }
                    }
                }
            }
        ),
        http_client=http,
    )

    memories = await client.recall(
        "我喜欢什么环境管理工具？",
        session_id="session-1",
        character_id="char-1",
        turn_id="turn-1",
    )

    assert memories == [
        {"text": "用户喜欢 Pixi。"},
        {"text": "用户在 macOS 上开发。"},
    ]
    assert http.posts[0]["url"] == (
        "https://hindsight.jmsu.top/v1/default/banks/cv:user:user-1:character:char-1/memories/recall"
    )
    assert http.posts[0]["headers"]["Authorization"] == "Bearer test-key"
    assert http.posts[0]["json"] == {
        "query": "我喜欢什么环境管理工具",
        "types": ["world", "experience"],
        "budget": "low",
        "max_results": 2,
        "max_tokens": 4096,
        "tags": [
            "source:cyberverse",
            "user:user-1",
            "character:char-1",
            "session:session-1",
            "source:voice",
            "user-1",
        ],
        "tags_match": "any",
    }


@pytest.mark.asyncio
async def test_hindsight_retain_sends_expected_payload():
    http = FakeHTTPClient(FakeResponse({"ok": True}))
    client = HindsightMemoryClient(
        hindsight_config_from_runtime_config(
            {
                "inference": {
                    "persona_agent": {
                        "memory": {
                            "hindsight": {
                                "api_key": "test-key",
                                "user_tag": "user-1",
                            }
                        }
                    }
                }
            }
        ),
        http_client=http,
    )

    result = await client.retain(
        "用户: 你好\n助手: 你好，我在。",
        session_id="session-1",
        character_id="char-1",
        turn_id="turn-1",
        metadata={"source": "test"},
    )

    assert result == {"ok": True}
    assert http.posts[0]["url"] == "https://hindsight.jmsu.top/v1/default/banks/cv:user:user-1:character:char-1/memories"
    assert http.posts[0]["json"] == {
        "items": [
            {
                "content": "用户: 你好\n助手: 你好，我在。",
                "context": "cyberverse realtime conversation",
                "tags": [
                    "source:cyberverse",
                    "user:user-1",
                    "character:char-1",
                    "session:session-1",
                    "source:voice",
                    "user-1",
                ],
                "metadata": {"source": "test"},
                "document_id": "session:session-1:turn:turn-1",
            }
        ],
        "async": True,
    }


@pytest.mark.asyncio
async def test_hindsight_http_errors_do_not_raise():
    http = FakeHTTPClient(FakeResponse(error=RuntimeError("service down")))
    client = HindsightMemoryClient(
        hindsight_config_from_runtime_config(
            {
                "inference": {
                    "persona_agent": {
                        "memory": {
                            "hindsight": {
                                "api_key": "test-key",
                                "user_tag": "user-1",
                            }
                        }
                    }
                }
            }
        ),
        http_client=http,
    )

    assert await client.recall("hello") == []
    assert await client.retain("hello") == {"ok": False}


@pytest.mark.asyncio
async def test_hindsight_recall_normalizes_query_punctuation():
    http = FakeHTTPClient(FakeResponse({"results": []}))
    client = HindsightMemoryClient(
        hindsight_config_from_runtime_config(
            {
                "inference": {
                    "persona_agent": {
                        "memory": {
                            "hindsight": {
                                "api_key": "test-key",
                                "user_tag": "user-1",
                            }
                        }
                    }
                }
            }
        ),
        http_client=http,
    )

    await client.recall("我喜欢 Pixi？项目暗号：蓝色海盐-bb98afe4。")

    assert http.posts[0]["json"]["query"] == "我喜欢 Pixi 项目暗号 蓝色海盐-bb98afe4"


@pytest.mark.asyncio
async def test_hindsight_local_fallback_recalls_retained_content(tmp_path):
    client = FakeRemoteHindsightClient(
        hindsight_config_from_runtime_config(
            {
                "inference": {
                    "persona_agent": {
                        "memory": {
                            "hindsight": {
                                "api_key": "test-key",
                                "user_tag": "user-1",
                                "local_fallback_path": str(tmp_path / "memory.jsonl"),
                            }
                        }
                    }
                }
            }
        ),
        recall_payload={"results": [{"text": "远端已有记忆。"}]},
    )

    await client.retain("用户: 请记住我最喜欢的环境管理工具是 Pixi。\n助手: 好的。")
    memories = await client.recall("我最喜欢的环境管理工具是什么？")

    assert memories[0]["text"] == "用户: 请记住我最喜欢的环境管理工具是 Pixi。\n助手: 好的。"
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_hindsight_local_fallback_survives_remote_recall_error(tmp_path):
    client = FakeRemoteHindsightClient(
        hindsight_config_from_runtime_config(
            {
                "inference": {
                    "persona_agent": {
                        "memory": {
                            "hindsight": {
                                "api_key": "test-key",
                                "user_tag": "user-1",
                                "local_fallback_path": str(tmp_path / "memory.jsonl"),
                            }
                        }
                    }
                }
            }
        ),
        error=RuntimeError("remote recall failed"),
    )

    await client.retain("用户: 项目暗号是 蓝色海盐-bb98afe4。\n助手: 已记录。")

    assert await client.recall("项目暗号是什么？") == [
        {"text": "用户: 项目暗号是 蓝色海盐-bb98afe4。\n助手: 已记录。"}
    ]


@pytest.mark.asyncio
async def test_hindsight_local_fallback_prefers_explicit_memory_writes(tmp_path):
    client = FakeRemoteHindsightClient(
        hindsight_config_from_runtime_config(
            {
                "inference": {
                    "persona_agent": {
                        "memory": {
                            "hindsight": {
                                "api_key": "test-key",
                                "user_tag": "user-1",
                                "local_fallback_path": str(tmp_path / "memory.jsonl"),
                            }
                        }
                    }
                }
            }
        ),
        error=RuntimeError("remote recall failed"),
    )

    await client.retain("用户: 我最喜欢的环境管理工具是什么？项目暗号是什么？\n助手: 项目暗号是 OldMarker。")
    await client.retain("用户: 请记住：我最喜欢的环境管理工具是 Pixi，项目暗号是 NewMarker。\n助手: 已记录。")

    memories = await client.recall("我最喜欢的环境管理工具是什么？项目暗号是什么？")

    assert memories[0] == {
        "text": "用户: 请记住：我最喜欢的环境管理工具是 Pixi，项目暗号是 NewMarker。\n助手: 已记录。"
    }
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_hindsight_local_fallback_ignores_unrelated_explicit_memory_writes(tmp_path):
    client = FakeRemoteHindsightClient(
        hindsight_config_from_runtime_config(
            {
                "inference": {
                    "persona_agent": {
                        "memory": {
                            "hindsight": {
                                "api_key": "test-key",
                                "user_tag": "user-1",
                                "local_fallback_path": str(tmp_path / "memory.jsonl"),
                            }
                        }
                    }
                }
            }
        ),
        error=RuntimeError("remote recall failed"),
    )

    await client.retain("用户: 请记住：我最喜欢的环境管理工具是 Pixi，项目暗号是 NewMarker。\n助手: 已记录。")

    assert await client.recall("今天晚饭吃什么？") == []
