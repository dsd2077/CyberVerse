package orchestrator

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLocalFallbackPrefersExplicitMemoryWrites(t *testing.T) {
	path := filepath.Join(t.TempDir(), "memory.jsonl")
	client := &hindsightConversationMemory{
		maxResults:        3,
		localFallback:     true,
		localFallbackPath: path,
	}

	client.appendLocal("用户: 我最喜欢的环境管理工具是什么？项目暗号是什么？\n助手: 项目暗号是 OldMarker。", "conversation")
	client.appendLocal("用户: 请记住：我最喜欢的环境管理工具是 Pixi，项目暗号是 NewMarker。\n助手: 已记录。", "conversation")

	memories := client.recallLocal("我最喜欢的环境管理工具是什么？项目暗号是什么？")
	if len(memories) == 0 {
		t.Fatal("expected local fallback memories")
	}
	if !strings.Contains(memories[0], "NewMarker") {
		t.Fatalf("expected explicit memory write first, got %+v", memories)
	}
	if len(memories) != 1 {
		t.Fatalf("expected only latest explicit memory write, got %+v", memories)
	}
}

func TestLocalFallbackIgnoresUnrelatedExplicitMemoryWrites(t *testing.T) {
	path := filepath.Join(t.TempDir(), "memory.jsonl")
	client := &hindsightConversationMemory{
		maxResults:        3,
		localFallback:     true,
		localFallbackPath: path,
	}

	client.appendLocal("用户: 请记住：我最喜欢的环境管理工具是 Pixi，项目暗号是 NewMarker。\n助手: 已记录。", "conversation")

	memories := client.recallLocal("今天晚饭吃什么？")
	if len(memories) != 0 {
		t.Fatalf("expected unrelated explicit memory to be ignored, got %+v", memories)
	}
}

func TestHindsightConversationMemoryRetainWritesLocalFallback(t *testing.T) {
	path := filepath.Join(t.TempDir(), "memory.jsonl")
	client := &hindsightConversationMemory{
		baseURL:           "http://127.0.0.1:1",
		apiKey:            "test",
		bankIDTemplate:    "cv:user:{user_id}:character:{character_id}",
		userTag:           "user-1",
		maxResults:        3,
		maxTokens:         1024,
		retainMaxChars:    1000,
		localFallback:     true,
		localFallbackPath: path,
		httpClient:        &http.Client{Timeout: 10 * time.Millisecond},
	}

	_ = client.Retain(
		context.Background(),
		"用户: 请记住项目暗号是 LocalOnly。\n助手: 好。",
		"conversation",
		conversationMemoryScope{SessionID: "session-1", CharacterID: "char-1", Source: "standard", TurnID: "1"},
	)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !containsText([]string{string(data)}, "LocalOnly") {
		t.Fatalf("local fallback did not persist retained content: %s", data)
	}
}

func TestHindsightConversationMemoryResolvesCyberVerseScope(t *testing.T) {
	client := &hindsightConversationMemory{
		bankIDTemplate:     "cv:user:{user_id}:character:{character_id}",
		userTag:            "user-1",
		documentIDTemplate: "session:{session_id}:turn:{turn_id}",
		defaultTags:        []string{"source:cyberverse"},
	}
	scope := conversationMemoryScope{
		SessionID:   "session-1",
		CharacterID: "char-1",
		Source:      "standard",
		TurnID:      "7",
	}

	if got := client.resolveBankID(scope); got != "cv:user:user-1:character:char-1" {
		t.Fatalf("unexpected bank id: %s", got)
	}
	tags := client.tags(scope)
	for _, want := range []string{"source:cyberverse", "user:user-1", "character:char-1", "session:session-1", "source:standard", "user-1"} {
		if !containsText(tags, want) {
			t.Fatalf("expected tag %q in %+v", want, tags)
		}
	}
	if got := client.documentID(scope); got != "session:session-1:turn:7" {
		t.Fatalf("unexpected document id: %s", got)
	}
}

func containsText(values []string, needle string) bool {
	for _, value := range values {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}
