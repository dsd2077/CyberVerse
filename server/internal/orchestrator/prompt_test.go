package orchestrator

import (
	"strings"
	"testing"

	"github.com/cyberverse/server/internal/character"
)

func TestComposeSystemPromptSeparatesGlobalAndRole(t *testing.T) {
	got := composeSystemPrompt("全局规则", "角色设定")

	if !strings.Contains(got, "【全局输出规范】\n全局规则") {
		t.Fatalf("expected global section, got %q", got)
	}
	if !strings.Contains(got, "【角色设定】\n角色设定") {
		t.Fatalf("expected role section, got %q", got)
	}
}

func TestStandardSystemPromptUsesGlobalWithoutCharacter(t *testing.T) {
	orch := New(nil, nil, nil, nil, nil)
	session := NewSession("s1", ModeStandard, "")

	got := orch.standardSystemPrompt(session)
	if got != composeSystemPrompt(standardGlobalSystemPrompt, "") {
		t.Fatalf("unexpected system prompt: %q", got)
	}
}

func TestBuildVoiceLLMSessionConfigUsesOnlyVoiceLLMRolePrompt(t *testing.T) {
	store, err := character.NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	char, err := store.Create(&character.Character{
		Name:          "晴天",
		Description:   "日常聊天伙伴",
		VoiceProvider: "qwen_omni",
		VoiceType:     "Tina",
		SpeakingStyle: "自然、简洁",
		Personality:   "温和真诚",
		SystemPrompt:  "你和用户像熟悉的朋友。",
	})
	if err != nil {
		t.Fatal(err)
	}

	orch := New(nil, nil, nil, nil, store)
	session := NewSession("s1", ModeVoiceLLM, char.ID)

	got := orch.buildVoiceLLMSessionConfig(session, "s1")
	if got.Provider != "qwen_omni" {
		t.Fatalf("expected qwen_omni provider, got %q", got.Provider)
	}
	if got.BotName != "晴天" || got.SpeakingStyle != "自然、简洁" {
		t.Fatalf("expected voice-specific fields to stay separate, got %+v", got)
	}
	if got.SystemPrompt != "你和用户像熟悉的朋友。" {
		t.Fatalf("expected VoiceLLM to keep only the character prompt, got %q", got.SystemPrompt)
	}
	for _, unexpected := range []string{"【全局输出规范】", "默认简短", "角色描述：", "角色性格：", "说话风格："} {
		if strings.Contains(got.SystemPrompt, unexpected) {
			t.Fatalf("VoiceLLM prompt should not contain %q: %q", unexpected, got.SystemPrompt)
		}
	}
}
