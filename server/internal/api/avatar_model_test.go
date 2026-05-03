package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/config"
	"github.com/cyberverse/server/internal/inference"
	"github.com/cyberverse/server/internal/orchestrator"
	pb "github.com/cyberverse/server/internal/pb"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type fakeInferenceService struct {
	avatarInfo              *pb.AvatarInfo
	infoErr                 error
	setAvatarErr            error
	setAvatarErrs           []error
	setAvatarCalls          int
	setAvatarSizes          []int
	setAvatarFormats        []string
	checkVoiceProviderError string
	checkVoiceErr           error
	voiceConfigs            chan inference.VoiceLLMSessionConfig
}

func (f *fakeInferenceService) HealthCheck(ctx context.Context) error {
	_, err := f.AvatarInfo(ctx)
	return err
}

func (f *fakeInferenceService) AvatarInfo(context.Context) (*pb.AvatarInfo, error) {
	if f.infoErr != nil {
		return nil, f.infoErr
	}
	return f.avatarInfo, nil
}

func (f *fakeInferenceService) SetAvatar(_ context.Context, _ string, imageData []byte, format string) error {
	f.setAvatarCalls++
	f.setAvatarSizes = append(f.setAvatarSizes, len(imageData))
	f.setAvatarFormats = append(f.setAvatarFormats, format)
	if len(f.setAvatarErrs) >= f.setAvatarCalls {
		return f.setAvatarErrs[f.setAvatarCalls-1]
	}
	return f.setAvatarErr
}
func (f *fakeInferenceService) GenerateAvatarStream(context.Context, <-chan *pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	videoCh := make(chan *pb.VideoChunk)
	errCh := make(chan error)
	close(videoCh)
	close(errCh)
	return videoCh, errCh
}
func (f *fakeInferenceService) GenerateAvatar(context.Context, []*pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	videoCh := make(chan *pb.VideoChunk)
	errCh := make(chan error)
	close(videoCh)
	close(errCh)
	return videoCh, errCh
}
func (f *fakeInferenceService) GenerateLLMStream(context.Context, string, []inference.ChatMessage, inference.LLMConfig) (<-chan *pb.LLMChunk, <-chan error) {
	ch := make(chan *pb.LLMChunk)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}
func (f *fakeInferenceService) SynthesizeSpeechStream(context.Context, <-chan string, inference.TTSConfig) (<-chan *pb.AudioChunk, <-chan error) {
	ch := make(chan *pb.AudioChunk)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}
func (f *fakeInferenceService) TranscribeStream(context.Context, <-chan []byte, inference.ASRConfig) (<-chan *pb.TranscriptEvent, <-chan error) {
	ch := make(chan *pb.TranscriptEvent)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}
func (f *fakeInferenceService) CheckVoice(context.Context, inference.VoiceLLMSessionConfig) (string, error) {
	if f.checkVoiceErr != nil {
		return "", f.checkVoiceErr
	}
	return f.checkVoiceProviderError, nil
}
func (f *fakeInferenceService) ConverseStream(_ context.Context, _ <-chan inference.VoiceLLMInputEvent, config inference.VoiceLLMSessionConfig) (<-chan *pb.VoiceLLMOutput, <-chan error) {
	if f.voiceConfigs != nil {
		select {
		case f.voiceConfigs <- config:
		default:
		}
	}
	ch := make(chan *pb.VoiceLLMOutput)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}
func (f *fakeInferenceService) Interrupt(context.Context, string) error { return nil }
func (f *fakeInferenceService) Close() error                            { return nil }

func newAvatarModelTestRouter(t *testing.T, activeModel string) (*Router, *character.Store) {
	t.Helper()

	root := t.TempDir()
	configPath := filepath.Join(root, "cyberverse_config.yaml")
	configYAML := `server:
  host: "0.0.0.0"
  http_port: 8080
  grpc_port: 50051
inference:
  avatar:
    default: "flash_head"
    runtime:
      cuda_visible_devices: "0,1"
      world_size: 2
    flash_head:
      plugin_class: "inference.plugins.avatar.flash_head_plugin.FlashHeadAvatarPlugin"
      checkpoint_dir: "/tmp/flash"
      wav2vec_dir: "/tmp/wav2vec"
      model_type: "pro"
      compile_model: true
      compile_vae: true
      dist_worker_main_thread: true
      infer_params:
        tgt_fps: 25
        frame_num: 33
    live_act:
      plugin_class: "inference.plugins.avatar.live_act_plugin.LiveActAvatarPlugin"
      ckpt_dir: "/tmp/live_act"
      wav2vec_dir: "/tmp/live_wav2vec"
      seed: 42
      t5_cpu: false
      fp8_kv_cache: false
      offload_cache: false
      block_offload: false
      mean_memory: false
      compile_wan_model: true
      compile_vae_decode: true
      dist_worker_main_thread: true
      default_prompt: "一个人在说话"
      infer_params:
        size: "320*480"
        fps: 24
        audio_cfg: 1.0
`
	if err := os.WriteFile(configPath, []byte(configYAML), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "models", "live_act"), 0755); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.Load(configPath)
	if err != nil {
		t.Fatal(err)
	}
	charStore, err := character.NewStore(filepath.Join(root, "characters"))
	if err != nil {
		t.Fatal(err)
	}
	inf := &fakeInferenceService{
		avatarInfo: &pb.AvatarInfo{ModelName: "avatar." + activeModel, OutputFps: 24, OutputWidth: 320, OutputHeight: 480},
	}
	orch := orchestrator.New(inf, nil, orchestrator.NewSessionManager(4), nil, charStore)
	return NewRouter(orchestrator.NewSessionManager(4), orch, nil, nil, cfg, charStore, "", configPath), charStore
}

func TestGetAvatarModelInfoUsesRuntimeModel(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "live_act")

	req := httptest.NewRequest("GET", "/api/v1/config/avatar-model", nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp avatarModelInfoResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.ActiveModel != "live_act" {
		t.Fatalf("expected active_model live_act, got %q", resp.ActiveModel)
	}
	if resp.ConfiguredDefaultModel != "flash_head" {
		t.Fatalf("expected configured_default_model flash_head, got %q", resp.ConfiguredDefaultModel)
	}
	for _, model := range resp.Models {
		if model.Name == "runtime" {
			t.Fatalf("did not expect runtime helper node to appear as an avatar model")
		}
	}
	if !resp.ConfigStatus.HasInferParams {
		t.Fatalf("expected live_act infer params to be present")
	}
}

func TestGetLaunchConfigKeepsLiveActModelParamsOutOfGPUSection(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "live_act")

	req := httptest.NewRequest("GET", "/api/v1/config/launch?model=flash_head", nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp launchConfigResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.ActiveModel != "live_act" {
		t.Fatalf("expected active_model live_act, got %q", resp.ActiveModel)
	}
	foundAvatarSection := false
	foundVideoSection := false
	foundGPUSection := false
	for _, section := range resp.Sections {
		if section.Title == "头像模型 (Avatar)" {
			foundAvatarSection = true
			paths := map[string]any{}
			for _, param := range section.Params {
				paths[param.Path] = param.Value
			}
			if got := fmt.Sprint(paths["inference.avatar.live_act.t5_cpu"]); got != "false" {
				t.Fatalf("expected live_act t5_cpu in avatar section, got %#v", got)
			}
			if got := fmt.Sprint(paths["inference.avatar.live_act.compile_wan_model"]); got != "true" {
				t.Fatalf("expected live_act compile_wan_model in avatar section, got %#v", got)
			}
			if got := fmt.Sprint(paths["inference.avatar.live_act.dist_worker_main_thread"]); got != "true" {
				t.Fatalf("expected live_act dist_worker_main_thread in avatar section, got %#v", got)
			}
			if got := fmt.Sprint(paths["inference.avatar.live_act.default_prompt"]); got != "一个人在说话" {
				t.Fatalf("expected live_act default_prompt in avatar section, got %#v", got)
			}
			continue
		}
		if section.Title == "视频输出" {
			foundVideoSection = true
			paths := map[string]any{}
			for _, param := range section.Params {
				paths[param.Path] = param.Value
			}
			if got := fmt.Sprint(paths["inference.avatar.live_act.infer_params.size"]); got != "320*480" {
				t.Fatalf("expected live_act infer_params.size from main config, got %#v", got)
			}
			if got := fmt.Sprint(paths["inference.avatar.live_act.infer_params.fps"]); got != "24" {
				t.Fatalf("expected live_act infer_params.fps from main config, got %#v", got)
			}
			continue
		}
		if section.Title != "GPU 配置" {
			continue
		}
		foundGPUSection = true
		paths := map[string]bool{}
		for _, param := range section.Params {
			paths[param.Path] = true
		}
		if !paths["inference.avatar.runtime.cuda_visible_devices"] {
			t.Fatalf("expected shared avatar runtime cuda_visible_devices in GPU section")
		}
		if !paths["inference.avatar.runtime.world_size"] {
			t.Fatalf("expected shared avatar runtime world_size in GPU section")
		}
		if paths["inference.avatar.live_act.compile_wan_model"] {
			t.Fatalf("did not expect live_act compile_wan_model in GPU section")
		}
		if paths["inference.avatar.live_act.t5_cpu"] {
			t.Fatalf("did not expect live_act t5_cpu in GPU section")
		}
	}
	if !foundAvatarSection {
		t.Fatalf("expected 头像模型 (Avatar) section for live_act")
	}
	if !foundVideoSection {
		t.Fatalf("expected 视频输出 section for live_act")
	}
	if !foundGPUSection {
		t.Fatalf("expected GPU 配置 section for live_act")
	}
}

func TestGetLaunchConfigReadsVideoSectionFromMainConfig(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "flash_head")

	req := httptest.NewRequest("GET", "/api/v1/config/launch", nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp launchConfigResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	foundVideoSection := false
	foundAvatarSection := false
	for _, section := range resp.Sections {
		if section.Title == "头像模型 (Avatar)" {
			foundAvatarSection = true
			paths := map[string]any{}
			for _, param := range section.Params {
				paths[param.Path] = param.Value
			}
			if got := fmt.Sprint(paths["inference.avatar.flash_head.compile_model"]); got != "true" {
				t.Fatalf("expected flash_head compile_model from main config, got %#v", got)
			}
			if got := fmt.Sprint(paths["inference.avatar.flash_head.compile_vae"]); got != "true" {
				t.Fatalf("expected flash_head compile_vae from main config, got %#v", got)
			}
			continue
		}
		if section.Title != "视频输出" {
			continue
		}
		foundVideoSection = true
		paths := map[string]any{}
		for _, param := range section.Params {
			paths[param.Path] = param.Value
		}
		if got := fmt.Sprint(paths["inference.avatar.flash_head.infer_params.tgt_fps"]); got != "25" {
			t.Fatalf("expected tgt_fps from main config, got %#v", got)
		}
		if got := fmt.Sprint(paths["inference.avatar.flash_head.infer_params.frame_num"]); got != "33" {
			t.Fatalf("expected frame_num from main config, got %#v", got)
		}
	}
	if !foundVideoSection {
		t.Fatalf("expected 视频输出 section for flash_head")
	}
	if !foundAvatarSection {
		t.Fatalf("expected 头像模型 (Avatar) section for flash_head")
	}
}

func TestUpdateLaunchConfigRejectsNonActiveModel(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "live_act")

	body := `{"model":"flash_head","params":[{"path":"inference.avatar.flash_head.world_size","value":1}]}`
	req := httptest.NewRequest("PUT", "/api/v1/config/launch", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

func TestUpdateLaunchConfigAllowsSharedAvatarRuntimeUpdates(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "live_act")

	body := `{"model":"live_act","params":[{"path":"inference.avatar.runtime.world_size","value":1}]}`
	req := httptest.NewRequest("PUT", "/api/v1/config/launch", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	doc, err := config.ReadYAMLNode(r.configPath)
	if err != nil {
		t.Fatal(err)
	}
	node, err := config.GetNodeAtPath(doc, "inference.avatar.runtime.world_size")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(config.NodeValue(node, true)); got != "1" {
		t.Fatalf("expected shared world_size to be updated to 1, got %#v", got)
	}
}

func TestUpdateLaunchConfigWritesInferParamsToMainConfig(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "flash_head")

	body := `{"model":"flash_head","params":[{"path":"inference.avatar.flash_head.infer_params.frame_num","value":29}]}`
	req := httptest.NewRequest("PUT", "/api/v1/config/launch", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	doc, err := config.ReadYAMLNode(r.configPath)
	if err != nil {
		t.Fatal(err)
	}
	node, err := config.GetNodeAtPath(doc, "inference.avatar.flash_head.infer_params.frame_num")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(config.NodeValue(node, true)); got != "29" {
		t.Fatalf("expected frame_num to be updated to 29, got %#v", got)
	}
}

func TestUpdateLaunchConfigWritesFlashHeadRootParamsToMainConfig(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "flash_head")

	body := `{"model":"flash_head","params":[{"path":"inference.avatar.flash_head.compile_model","value":false}]}`
	req := httptest.NewRequest("PUT", "/api/v1/config/launch", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	doc, err := config.ReadYAMLNode(r.configPath)
	if err != nil {
		t.Fatal(err)
	}
	node, err := config.GetNodeAtPath(doc, "inference.avatar.flash_head.compile_model")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(config.NodeValue(node, true)); got != "false" {
		t.Fatalf("expected compile_model to be updated to false, got %#v", got)
	}
}

func TestUpdateLaunchConfigWritesLiveActInferParamsToMainConfig(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "live_act")

	body := `{"model":"live_act","params":[{"path":"inference.avatar.live_act.infer_params.fps","value":20}]}`
	req := httptest.NewRequest("PUT", "/api/v1/config/launch", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	doc, err := config.ReadYAMLNode(r.configPath)
	if err != nil {
		t.Fatal(err)
	}
	node, err := config.GetNodeAtPath(doc, "inference.avatar.live_act.infer_params.fps")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(config.NodeValue(node, true)); got != "20" {
		t.Fatalf("expected fps to be updated to 20, got %#v", got)
	}
}

func TestUpdateLaunchConfigWritesLiveActRootParamsToMainConfig(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "live_act")

	body := `{"model":"live_act","params":[{"path":"inference.avatar.live_act.t5_cpu","value":true}]}`
	req := httptest.NewRequest("PUT", "/api/v1/config/launch", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	doc, err := config.ReadYAMLNode(r.configPath)
	if err != nil {
		t.Fatal(err)
	}
	node, err := config.GetNodeAtPath(doc, "inference.avatar.live_act.t5_cpu")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(config.NodeValue(node, true)); got != "true" {
		t.Fatalf("expected t5_cpu to be updated to true, got %#v", got)
	}
}

func TestCreateSessionWithCharacterUsesActiveRuntimeModelOnly(t *testing.T) {
	r, charStore := newAvatarModelTestRouter(t, "live_act")
	char, err := charStore.Create(&character.Character{
		Name:      "Character Session",
		VoiceType: "温柔文雅",
	})
	if err != nil {
		t.Fatal(err)
	}

	body := `{"mode":"voice_llm","character_id":"` + char.ID + `"}`
	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d", w.Code)
	}
}

func TestCreateSessionReturnsAvatarWarningWhenImageExceedsGRPCLimit(t *testing.T) {
	charStore, err := character.NewStore(filepath.Join(t.TempDir(), "characters"))
	if err != nil {
		t.Fatal(err)
	}
	char, err := charStore.Create(&character.Character{
		Name:      "Large Avatar",
		VoiceType: "温柔文雅",
	})
	if err != nil {
		t.Fatal(err)
	}

	image := character.ImageInfo{
		Filename: "avatar.png",
		OrigName: "avatar.png",
	}
	if err := os.WriteFile(filepath.Join(charStore.ImagesDir(char.ID), image.Filename), []byte("avatar-bytes"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := charStore.AddImage(char.ID, image); err != nil {
		t.Fatal(err)
	}
	idleDir := charStore.IdleVideosForSizeDir(char.ID, image.Filename, 512, 512)
	if err := os.MkdirAll(idleDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(idleDir, "cached.mp4"), []byte("cached"), 0644); err != nil {
		t.Fatal(err)
	}

	mgr := orchestrator.NewSessionManager(4)
	tooLargeErr := status.Error(codes.ResourceExhausted, "trying to send message larger than max (22347880 vs. 10485760)")
	inf := &fakeInferenceService{
		avatarInfo:    &pb.AvatarInfo{ModelName: "avatar.flash_head", OutputFps: 25, OutputWidth: 512, OutputHeight: 512},
		setAvatarErrs: []error{tooLargeErr, nil},
	}
	orch := orchestrator.New(inf, nil, mgr, nil, charStore)
	r := NewRouter(mgr, orch, nil, nil, nil, charStore, "", "")

	body := `{"mode":"voice_llm","character_id":"` + char.ID + `"}`
	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d", w.Code)
	}

	var resp CreateSessionResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Warnings) != 1 {
		t.Fatalf("expected one warning, got %v", resp.Warnings)
	}
	if !strings.Contains(resp.Warnings[0], "10MB") {
		t.Fatalf("expected warning to mention 10MB upload limit, got %q", resp.Warnings[0])
	}
	if inf.setAvatarCalls != 2 {
		t.Fatalf("expected failed character SetAvatar plus default reset SetAvatar, got %d calls", inf.setAvatarCalls)
	}
	if got := inf.setAvatarFormats[1]; got != "png" {
		t.Fatalf("expected default reset to use png, got %q", got)
	}
	if size := inf.setAvatarSizes[1]; size <= 0 || size >= 10*1024*1024 {
		t.Fatalf("expected compact default avatar image, got %d bytes", size)
	}
}

func TestCreateSessionRejectsWhenActiveRuntimeModelUnavailable(t *testing.T) {
	charStore, err := character.NewStore(filepath.Join(t.TempDir(), "characters"))
	if err != nil {
		t.Fatal(err)
	}
	char, err := charStore.Create(&character.Character{
		Name:      "Unavailable",
		VoiceType: "温柔文雅",
	})
	if err != nil {
		t.Fatal(err)
	}

	mgr := orchestrator.NewSessionManager(4)
	orch := orchestrator.New(&fakeInferenceService{
		infoErr: errors.New("inference unavailable"),
	}, nil, mgr, nil, charStore)
	r := NewRouter(mgr, orch, nil, nil, nil, charStore, "", "")

	body := `{"mode":"voice_llm","character_id":"` + char.ID + `"}`
	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}
