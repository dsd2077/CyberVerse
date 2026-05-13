package orchestrator

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cyberverse/server/internal/agenttask"
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

func TestBuildVoiceLLMSessionConfigUsesPersonaAndOnlyOmniRolePrompt(t *testing.T) {
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
	session := NewSession("s1", ModeOmni, char.ID)

	got := orch.buildVoiceLLMSessionConfig(session, "s1")
	if got.Provider != "persona" {
		t.Fatalf("expected persona provider, got %q", got.Provider)
	}
	if got.BotName != "晴天" || got.SpeakingStyle != "自然、简洁" {
		t.Fatalf("expected voice-specific fields to stay separate, got %+v", got)
	}
	if got.SystemPrompt != "你和用户像熟悉的朋友。" {
		t.Fatalf("expected omni mode to keep only the character prompt, got %q", got.SystemPrompt)
	}
	for _, unexpected := range []string{"【全局输出规范】", "默认简短", "角色描述：", "角色性格：", "说话风格："} {
		if strings.Contains(got.SystemPrompt, unexpected) {
			t.Fatalf("omni prompt should not contain %q: %q", unexpected, got.SystemPrompt)
		}
	}
}

func TestVoiceStartupGreetingUsesUnderlyingProviderAndNoFixedWelcome(t *testing.T) {
	store, err := character.NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	char, err := store.Create(&character.Character{
		Name:           "晴天",
		VoiceProvider:  "qwen_omni",
		VoiceType:      "Tina",
		SpeakingStyle:  "自然、简洁",
		SystemPrompt:   "你和用户像熟悉的朋友。",
		WelcomeMessage: "欢迎回来，我们继续聊。",
	})
	if err != nil {
		t.Fatal(err)
	}

	orch := New(nil, nil, nil, nil, store)
	session := NewSession("s1", ModeOmni, char.ID)

	normal := orch.buildVoiceLLMSessionConfig(session, "s1")
	if normal.Provider != "persona" {
		t.Fatalf("expected normal omni config to use persona, got %q", normal.Provider)
	}
	if normal.WelcomeMessage != "" {
		t.Fatalf("expected fixed welcome to be withheld from normal voice config, got %q", normal.WelcomeMessage)
	}

	greeting := orch.buildVoiceStartupGreetingSessionConfig(session, "s1")
	if greeting.Provider != "qwen_omni" {
		t.Fatalf("expected startup greeting to use underlying provider, got %q", greeting.Provider)
	}
	if greeting.WelcomeMessage != "" {
		t.Fatalf("expected startup greeting to disable provider SayHello, got %q", greeting.WelcomeMessage)
	}
}

func TestBuildVoiceStartupGreetingPromptUsesHistory(t *testing.T) {
	store, err := character.NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	char, err := store.Create(&character.Character{
		Name:           "晴天",
		SpeakingStyle:  "自然、简洁",
		SystemPrompt:   "你和用户像熟悉的朋友。",
		WelcomeMessage: "见到你很高兴。",
	})
	if err != nil {
		t.Fatal(err)
	}

	orch := New(nil, nil, nil, nil, store)
	session := NewSession("s1", ModeOmni, char.ID)
	session.SetDialogContext([]DialogContextItem{
		{Role: "user", Text: "上次我们在聊出差计划。", Timestamp: 1},
		{Role: "assistant", Text: "我帮你整理了行程重点。", Timestamp: 2},
	})

	got := orch.buildVoiceStartupGreetingPrompt(session)
	for _, want := range []string{
		"你的名字：晴天",
		"可参考的开场偏好：见到你很高兴。",
		"用户：上次我们在聊出差计划。",
		"你：我帮你整理了行程重点。",
		"默认不要回顾、总结、复述或主动延续这些内容",
		"不要主动提及取消、失败、争执、情绪化表达、敏感内容或具体历史细节",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("expected prompt to contain %q, got %q", want, got)
		}
	}
	if strings.Contains(got, "可以自然提到上次关注的话题") {
		t.Fatalf("expected startup greeting not to encourage routine history recall: %q", got)
	}
	if strings.Contains(got, "当前没有可用的历史对话") {
		t.Fatalf("expected history prompt, got no-history branch: %q", got)
	}
}

func TestBuildVoiceStartupGreetingPromptIntroducesSelfWithoutHistory(t *testing.T) {
	store, err := character.NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	char, err := store.Create(&character.Character{Name: "晴天"})
	if err != nil {
		t.Fatal(err)
	}

	orch := New(nil, nil, nil, nil, store)
	session := NewSession("s1", ModeOmni, char.ID)

	got := orch.buildVoiceStartupGreetingPrompt(session)
	for _, want := range []string{"你的名字：晴天", "当前没有可用的历史对话", "实时语音视频聊天", "查询、调研、整理资料"} {
		if !strings.Contains(got, want) {
			t.Fatalf("expected prompt to contain %q, got %q", want, got)
		}
	}
}

func TestBuildVoiceLLMSessionConfigUsesPersonaWhenAgentEnabled(t *testing.T) {
	store, err := character.NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	char, err := store.Create(&character.Character{
		Name:          "晴天",
		VoiceProvider: "qwen_omni",
		VoiceType:     "Tina",
		SystemPrompt:  "你和用户像熟悉的朋友。",
	})
	if err != nil {
		t.Fatal(err)
	}

	taskRoot := t.TempDir()
	taskStore, err := agenttask.OpenStore(filepath.Join(taskRoot, "tasks.db"), filepath.Join(taskRoot, "artifacts"))
	if err != nil {
		t.Fatal(err)
	}
	defer taskStore.Close()

	orch := New(nil, nil, nil, nil, store)
	orch.SetTaskService(agenttask.NewService(taskStore, nil))
	session := NewSession("s1", ModeOmni, char.ID)

	got := orch.buildVoiceLLMSessionConfig(session, "s1")
	if got.Provider != "persona" {
		t.Fatalf("expected persona provider, got %q", got.Provider)
	}
	if !orch.sessionSupportsVisualInput(session) {
		t.Fatal("expected qwen_omni visual support to use the underlying character provider")
	}
}
