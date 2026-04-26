package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/cyberverse/server/internal/config"
	"gopkg.in/yaml.v3"
)

var errInferenceUnavailable = errors.New("inference service is unavailable")

// SettingsResponse is the JSON shape exchanged with the frontend Settings page.
type SettingsResponse struct {
	Doubao    DoubaoSettings    `json:"doubao"`
	LiveKit   LiveKitSettings   `json:"livekit"`
	LLM       LLMSettings       `json:"llm"`
	TTS       TTSSettings       `json:"tts"`
	ASR       ASRSettings       `json:"asr"`
	Inference InferenceSettings `json:"inference"`
}

type DoubaoSettings struct {
	AccessToken string `json:"access_token"`
	AppID       string `json:"app_id"`
	WsURL       string `json:"ws_url"`
}

type LiveKitSettings struct {
	URL       string `json:"url"`
	APIKey    string `json:"api_key"`
	APISecret string `json:"api_secret"`
}

type LLMSettings struct {
	APIKey      string  `json:"api_key"`
	Model       string  `json:"model"`
	Temperature float64 `json:"temperature"`
}

type TTSSettings struct {
	Model string `json:"model"`
	Voice string `json:"voice"`
}

type ASRSettings struct {
	ModelSize string `json:"model_size"`
	Language  string `json:"language"`
	Device    string `json:"device"`
}

type InferenceSettings struct {
	GRPCAddr string `json:"grpc_addr"`
}

type launchConfigParamJSON struct {
	Name            string   `json:"name"`
	Path            string   `json:"path"`
	Value           any      `json:"value"`
	Readonly        bool     `json:"readonly"`
	RequiresRestart bool     `json:"requires_restart"`
	Options         []string `json:"options,omitempty"`
}

type launchConfigSectionJSON struct {
	Title  string                  `json:"title"`
	Badge  string                  `json:"badge"`
	Params []launchConfigParamJSON `json:"params"`
}

type avatarModelConfigStatus struct {
	HasInferParams          bool     `json:"has_infer_params"`
	ConfigSectionsAvailable []string `json:"config_sections_available"`
}

type avatarModelDescriptor struct {
	Name                string                  `json:"name"`
	DisplayName         string                  `json:"display_name"`
	IsActive            bool                    `json:"is_active"`
	IsConfiguredDefault bool                    `json:"is_configured_default"`
	ConfigStatus        avatarModelConfigStatus `json:"config_status"`
}

type avatarModelInfoResponse struct {
	ActiveModel            string                  `json:"active_model"`
	ConfiguredDefaultModel string                  `json:"configured_default_model"`
	Models                 []avatarModelDescriptor `json:"models"`
	ConfigStatus           avatarModelConfigStatus `json:"config_status"`
}

type launchConfigResponse struct {
	ActiveModel            string                    `json:"active_model"`
	ConfiguredDefaultModel string                    `json:"configured_default_model"`
	ConfigStatus           avatarModelConfigStatus   `json:"config_status"`
	Sections               []launchConfigSectionJSON `json:"sections"`
}

// settingsField maps a UI field to an environment variable.
type settingsField struct {
	envKey   string
	getValue func(*SettingsResponse) string
}

var settingsFields = []settingsField{
	{"DOUBAO_ACCESS_TOKEN", func(s *SettingsResponse) string { return s.Doubao.AccessToken }},
	{"DOUBAO_APP_ID", func(s *SettingsResponse) string { return s.Doubao.AppID }},
	{"DOUBAO_WS_URL", func(s *SettingsResponse) string { return s.Doubao.WsURL }},
	{"LIVEKIT_URL", func(s *SettingsResponse) string { return s.LiveKit.URL }},
	{"LIVEKIT_API_KEY", func(s *SettingsResponse) string { return s.LiveKit.APIKey }},
	{"LIVEKIT_API_SECRET", func(s *SettingsResponse) string { return s.LiveKit.APISecret }},
	{"OPENAI_API_KEY", func(s *SettingsResponse) string { return s.LLM.APIKey }},
	{"LLM_MODEL", func(s *SettingsResponse) string { return s.LLM.Model }},
	{"LLM_TEMPERATURE", func(s *SettingsResponse) string {
		return strconv.FormatFloat(s.LLM.Temperature, 'f', -1, 64)
	}},
	{"TTS_MODEL", func(s *SettingsResponse) string { return s.TTS.Model }},
	{"TTS_VOICE", func(s *SettingsResponse) string { return s.TTS.Voice }},
	{"ASR_MODEL_SIZE", func(s *SettingsResponse) string { return s.ASR.ModelSize }},
	{"ASR_LANGUAGE", func(s *SettingsResponse) string { return s.ASR.Language }},
	{"ASR_DEVICE", func(s *SettingsResponse) string { return s.ASR.Device }},
	{"GRPC_INFERENCE_ADDR", func(s *SettingsResponse) string { return s.Inference.GRPCAddr }},
}

func (r *Router) handleGetSettings(w http.ResponseWriter, req *http.Request) {
	resp := SettingsResponse{
		Doubao: DoubaoSettings{
			AccessToken: os.Getenv("DOUBAO_ACCESS_TOKEN"),
			AppID:       os.Getenv("DOUBAO_APP_ID"),
			WsURL:       envOrDefault("DOUBAO_WS_URL", "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"),
		},
		LiveKit: LiveKitSettings{
			URL:       envOrDefault("LIVEKIT_URL", r.cfg.LiveKit.URL),
			APIKey:    envOrDefault("LIVEKIT_API_KEY", r.cfg.LiveKit.APIKey),
			APISecret: envOrDefault("LIVEKIT_API_SECRET", r.cfg.LiveKit.APISecret),
		},
		LLM: LLMSettings{
			APIKey:      os.Getenv("OPENAI_API_KEY"),
			Model:       envOrDefault("LLM_MODEL", "gpt-4o"),
			Temperature: envOrDefaultFloat("LLM_TEMPERATURE", 0.7),
		},
		TTS: TTSSettings{
			Model: envOrDefault("TTS_MODEL", "tts-1"),
			Voice: envOrDefault("TTS_VOICE", "nova"),
		},
		ASR: ASRSettings{
			ModelSize: envOrDefault("ASR_MODEL_SIZE", "base"),
			Language:  envOrDefault("ASR_LANGUAGE", "auto"),
			Device:    envOrDefault("ASR_DEVICE", "cpu"),
		},
		Inference: InferenceSettings{
			GRPCAddr: envOrDefault("GRPC_INFERENCE_ADDR", r.cfg.Inference.Addr),
		},
	}
	writeJSON(w, http.StatusOK, resp)
}

func (r *Router) handleUpdateSettings(w http.ResponseWriter, req *http.Request) {
	var body SettingsResponse
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON"})
		return
	}

	updates := make(map[string]string)
	for _, f := range settingsFields {
		val := f.getValue(&body)
		// Skip empty values to avoid blanking existing config.
		if val == "" {
			continue
		}
		updates[f.envKey] = val
	}

	if len(updates) > 0 {
		// Persist to .env file.
		if r.envPath != "" {
			if err := config.SaveDotenv(r.envPath, updates); err != nil {
				writeJSON(w, http.StatusInternalServerError, ErrorResponse{
					Error: fmt.Sprintf("failed to save .env: %v", err),
				})
				return
			}
		}

		// Update process environment so subsequent GET reflects new values.
		for k, v := range updates {
			os.Setenv(k, v)
		}

		// Sync in-memory config for fields the Go struct captures.
		if v, ok := updates["LIVEKIT_URL"]; ok {
			r.cfg.LiveKit.URL = v
		}
		if v, ok := updates["LIVEKIT_API_KEY"]; ok {
			r.cfg.LiveKit.APIKey = v
		}
		if v, ok := updates["LIVEKIT_API_SECRET"]; ok {
			r.cfg.LiveKit.APISecret = v
		}
		if v, ok := updates["GRPC_INFERENCE_ADDR"]; ok {
			r.cfg.Inference.Addr = v
		}
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "saved"})
}

func (r *Router) handleTestConnection(w http.ResponseWriter, req *http.Request) {
	if err := r.inferenceHealthError(req.Context()); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"status": "error",
			"error":  err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// paramMeta defines special attributes for known parameter keys.
// Keys not listed here default to readonly=false, hidden=false.
var paramMeta = map[string]struct {
	Readonly bool
	Hidden   bool
	Options  []string
}{
	"plugin_class": {Readonly: true, Hidden: true},
	"models_dir":   {Readonly: true, Hidden: true},
	"model_type":   {Options: []string{"pro", "lite"}},
}

// Shared avatar runtime GPU-related keys are shown in a separate section.
var runtimeGPUKeys = map[string]bool{
	"cuda_visible_devices":      true,
	"world_size":                true,
	"dist_keepalive_interval_s": true,
	"dist_keepalive_idle_s":     true,
}

var runtimeGPUKeyOrder = []string{
	"cuda_visible_devices",
	"world_size",
	"dist_keepalive_interval_s",
	"dist_keepalive_idle_s",
}

// Some models keep a small subset of GPU-related controls outside infer_params.
// These remain in the GPU section only when explicitly allowlisted here.
var modelGPUKeys = map[string]map[string]bool{
	"flash_head": {
		"device": true,
	},
}

var modelGPUKeyOrder = map[string][]string{
	"flash_head": {"device"},
}

// Readonly keys in flash_head infer_params.
var inferParamsReadonly = map[string]bool{}

func normalizeAvatarModelName(name string) string {
	name = strings.TrimSpace(name)
	if strings.HasPrefix(name, "avatar.") {
		return strings.TrimPrefix(name, "avatar.")
	}
	return name
}

func displayAvatarModelName(name string) string {
	switch name {
	case "flash_head":
		return "FlashHead"
	case "live_act":
		return "LiveAct"
	default:
		parts := strings.Split(name, "_")
		for i, p := range parts {
			if p == "" {
				continue
			}
			parts[i] = strings.ToUpper(p[:1]) + p[1:]
		}
		return strings.Join(parts, " ")
	}
}

func isModelGPUKey(modelName, key string) bool {
	keys, ok := modelGPUKeys[modelName]
	return ok && keys[key]
}

func orderedGPUKeys(modelName string) []string {
	ordered := append([]string{}, runtimeGPUKeyOrder...)
	if extraOrder, ok := modelGPUKeyOrder[modelName]; ok {
		ordered = append(ordered, extraOrder...)
	}
	return ordered
}

func (r *Router) configuredDefaultAvatarModel() string {
	if r.configPath == "" {
		return ""
	}
	doc, err := config.ReadYAMLNode(r.configPath)
	if err != nil {
		return ""
	}
	node, err := config.GetNodeAtPath(doc, "inference.avatar.default")
	if err != nil {
		return ""
	}
	if v, ok := config.NodeValue(node, true).(string); ok {
		return strings.TrimSpace(v)
	}
	return ""
}

func (r *Router) configuredAvatarModels() []string {
	if r.configPath == "" {
		return nil
	}
	doc, err := config.ReadYAMLNode(r.configPath)
	if err != nil {
		return nil
	}
	keys, err := config.GetMappingKeys(doc, "inference.avatar")
	if err != nil {
		return nil
	}
	models := make([]string, 0, len(keys))
	for _, key := range keys {
		if key == "default" || key == "runtime" {
			continue
		}
		models = append(models, key)
	}
	sort.Strings(models)
	return models
}

func inferParamsConfigPath(modelName string) string {
	return "inference.avatar." + modelName + ".infer_params"
}

func (r *Router) inferParamsExists(modelName string) bool {
	if modelName == "" || r.configPath == "" {
		return false
	}
	doc, err := config.ReadYAMLNode(r.configPath)
	if err != nil {
		return false
	}
	_, err = config.GetMappingKeys(doc, inferParamsConfigPath(modelName))
	return err == nil
}

func (r *Router) configStatusForModel(modelName string) avatarModelConfigStatus {
	sections := make([]string, 0, 3)
	if modelName != "" {
		sections = append(sections, "avatar", "gpu")
	}
	hasInfer := r.inferParamsExists(modelName)
	if hasInfer {
		sections = append(sections, "video_output")
	}
	return avatarModelConfigStatus{
		HasInferParams:          hasInfer,
		ConfigSectionsAvailable: sections,
	}
}

func (r *Router) activeAvatarModel(ctx context.Context) (string, error) {
	if r.orch == nil {
		return "", errInferenceUnavailable
	}
	info, err := r.orch.AvatarInfo(ctx)
	if err != nil {
		return "", err
	}
	model := normalizeAvatarModelName(info.GetModelName())
	if model == "" {
		return "", errors.New("avatar model name is empty")
	}
	return model, nil
}

func (r *Router) buildAvatarModelInfo(ctx context.Context) (*avatarModelInfoResponse, error) {
	activeModel, err := r.activeAvatarModel(ctx)
	if err != nil {
		return nil, err
	}
	configuredDefault := r.configuredDefaultAvatarModel()
	configuredModels := r.configuredAvatarModels()
	if len(configuredModels) == 0 && activeModel != "" {
		configuredModels = []string{activeModel}
	}
	seen := map[string]bool{}
	models := make([]avatarModelDescriptor, 0, len(configuredModels)+1)
	for _, model := range append(configuredModels, activeModel) {
		if model == "" || seen[model] {
			continue
		}
		seen[model] = true
		models = append(models, avatarModelDescriptor{
			Name:                model,
			DisplayName:         displayAvatarModelName(model),
			IsActive:            model == activeModel,
			IsConfiguredDefault: model == configuredDefault,
			ConfigStatus:        r.configStatusForModel(model),
		})
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	return &avatarModelInfoResponse{
		ActiveModel:            activeModel,
		ConfiguredDefaultModel: configuredDefault,
		Models:                 models,
		ConfigStatus:           r.configStatusForModel(activeModel),
	}, nil
}

func (r *Router) buildLaunchSections(modelName string) []launchConfigSectionJSON {
	var sections []launchConfigSectionJSON
	avatarSection := launchConfigSectionJSON{Title: "头像模型 (Avatar)", Badge: "restart"}
	gpuSection := launchConfigSectionJSON{Title: "GPU 配置", Badge: "restart"}

	if r.configPath != "" {
		doc, err := config.ReadYAMLNode(r.configPath)
		if err == nil {
			modelPath := "inference.avatar." + modelName
			modelGPUParams := map[string]launchConfigParamJSON{}
			keys, err := config.GetMappingKeys(doc, modelPath)
			if err == nil {
				for _, key := range keys {
					meta, hasMeta := paramMeta[key]
					if hasMeta && meta.Hidden {
						continue
					}
					node, err := config.GetNodeAtPath(doc, modelPath+"."+key)
					if err != nil {
						continue
					}
					if node.Kind != yaml.ScalarNode {
						continue
					}
					p := launchConfigParamJSON{
						Name:            key,
						Path:            modelPath + "." + key,
						Value:           config.NodeValue(node, true),
						Readonly:        hasMeta && meta.Readonly,
						RequiresRestart: true,
					}
					if hasMeta && len(meta.Options) > 0 {
						p.Options = meta.Options
					}
					if isModelGPUKey(modelName, key) {
						modelGPUParams[key] = p
					} else {
						avatarSection.Params = append(avatarSection.Params, p)
					}
				}
			}

			runtimeGPUParams := map[string]launchConfigParamJSON{}
			runtimePath := "inference.avatar.runtime"
			runtimeKeys, err := config.GetMappingKeys(doc, runtimePath)
			if err == nil {
				for _, key := range runtimeKeys {
					if !runtimeGPUKeys[key] {
						continue
					}
					node, err := config.GetNodeAtPath(doc, runtimePath+"."+key)
					if err != nil {
						continue
					}
					runtimeGPUParams[key] = launchConfigParamJSON{
						Name:            key,
						Path:            runtimePath + "." + key,
						Value:           config.NodeValue(node, true),
						RequiresRestart: true,
					}
				}
			}

			usedGPUKeys := map[string]bool{}
			for _, key := range orderedGPUKeys(modelName) {
				if p, ok := modelGPUParams[key]; ok {
					gpuSection.Params = append(gpuSection.Params, p)
					usedGPUKeys[key] = true
					continue
				}
				if p, ok := runtimeGPUParams[key]; ok {
					gpuSection.Params = append(gpuSection.Params, p)
					usedGPUKeys[key] = true
				}
			}

			extraGPUKeySet := map[string]bool{}
			for key := range modelGPUParams {
				if !usedGPUKeys[key] {
					extraGPUKeySet[key] = true
				}
			}
			for key := range runtimeGPUParams {
				if !usedGPUKeys[key] {
					extraGPUKeySet[key] = true
				}
			}
			extraGPUKeys := make([]string, 0, len(extraGPUKeySet))
			for key := range extraGPUKeySet {
				extraGPUKeys = append(extraGPUKeys, key)
			}
			sort.Strings(extraGPUKeys)
			for _, key := range extraGPUKeys {
				if p, ok := modelGPUParams[key]; ok {
					gpuSection.Params = append(gpuSection.Params, p)
					usedGPUKeys[key] = true
					continue
				}
				if p, ok := runtimeGPUParams[key]; ok {
					gpuSection.Params = append(gpuSection.Params, p)
					usedGPUKeys[key] = true
				}
			}
		}
	}

	if len(avatarSection.Params) > 0 {
		sections = append(sections, avatarSection)
	}

	if r.inferParamsExists(modelName) {
		videoSection := launchConfigSectionJSON{Title: "视频输出", Badge: "restart"}
		inferPath := inferParamsConfigPath(modelName)
		if doc, err := config.ReadYAMLNode(r.configPath); err == nil {
			keys, err := config.GetMappingKeys(doc, inferPath)
			if err == nil {
				for _, key := range keys {
					node, err := config.GetNodeAtPath(doc, inferPath+"."+key)
					if err != nil {
						continue
					}
					videoSection.Params = append(videoSection.Params, launchConfigParamJSON{
						Name:            key,
						Path:            inferPath + "." + key,
						Value:           config.NodeValue(node, false),
						Readonly:        inferParamsReadonly[key],
						RequiresRestart: true,
					})
				}
			}
		}
		if len(videoSection.Params) > 0 {
			sections = append(sections, videoSection)
		}
	}

	if len(gpuSection.Params) > 0 {
		sections = append(sections, gpuSection)
	}
	return sections
}

func (r *Router) handleGetAvatarModelInfo(w http.ResponseWriter, req *http.Request) {
	info, err := r.buildAvatarModelInfo(req.Context())
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, info)
}

func (r *Router) handleGetLaunchConfig(w http.ResponseWriter, req *http.Request) {
	activeModel, err := r.activeAvatarModel(req.Context())
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, launchConfigResponse{
		ActiveModel:            activeModel,
		ConfiguredDefaultModel: r.configuredDefaultAvatarModel(),
		ConfigStatus:           r.configStatusForModel(activeModel),
		Sections:               r.buildLaunchSections(activeModel),
	})
}

func (r *Router) handleUpdateLaunchConfig(w http.ResponseWriter, req *http.Request) {
	var body struct {
		Model  string `json:"model"`
		Params []struct {
			Path  string `json:"path"`
			Value any    `json:"value"`
		} `json:"params"`
	}
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON"})
		return
	}
	activeModel, err := r.activeAvatarModel(req.Context())
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: err.Error()})
		return
	}
	if body.Model == "" {
		body.Model = activeModel
	}
	if body.Model != activeModel {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{
			Error: fmt.Sprintf("model %q is not active; current active model is %q", body.Model, activeModel),
		})
		return
	}

	// Group updates by config path.
	mainUpdates := map[string]string{} // dot-path -> value

	modelPrefix := "inference.avatar." + body.Model + "."
	runtimePrefix := "inference.avatar.runtime."
	inferParamsPrefix := inferParamsConfigPath(body.Model) + "."

	for _, p := range body.Params {
		// Determine source and validate.
		if strings.HasPrefix(p.Path, inferParamsPrefix) {
			key := strings.TrimPrefix(p.Path, inferParamsPrefix)
			if inferParamsReadonly[key] {
				writeJSON(w, http.StatusBadRequest, ErrorResponse{
					Error: fmt.Sprintf("parameter %q is readonly", p.Path),
				})
				return
			}
			mainUpdates[p.Path] = fmt.Sprintf("%v", p.Value)
		} else if strings.HasPrefix(p.Path, runtimePrefix) {
			key := strings.TrimPrefix(p.Path, runtimePrefix)
			if !runtimeGPUKeys[key] {
				writeJSON(w, http.StatusBadRequest, ErrorResponse{
					Error: fmt.Sprintf("parameter %q is not a shared runtime parameter", p.Path),
				})
				return
			}
			mainUpdates[p.Path] = fmt.Sprintf("%v", p.Value)
		} else if strings.HasPrefix(p.Path, modelPrefix) {
			key := strings.TrimPrefix(p.Path, modelPrefix)
			meta, hasMeta := paramMeta[key]
			if hasMeta && meta.Readonly {
				writeJSON(w, http.StatusBadRequest, ErrorResponse{
					Error: fmt.Sprintf("parameter %q is readonly", p.Path),
				})
				return
			}
			mainUpdates[p.Path] = fmt.Sprintf("%v", p.Value)
		} else {
			writeJSON(w, http.StatusBadRequest, ErrorResponse{
				Error: fmt.Sprintf("parameter %q is not in scope for model %q", p.Path, body.Model),
			})
			return
		}
	}

	if len(mainUpdates) > 0 && r.configPath != "" {
		doc, err := config.ReadYAMLNode(r.configPath)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: err.Error()})
			return
		}
		for path, val := range mainUpdates {
			if err := config.SetNodeAtPath(doc, path, val); err != nil {
				writeJSON(w, http.StatusInternalServerError, ErrorResponse{
					Error: fmt.Sprintf("set %s: %v", path, err),
				})
				return
			}
		}
		if err := config.WriteYAMLNode(r.configPath, doc); err != nil {
			writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: err.Error()})
			return
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"status":           "saved",
		"requires_restart": true,
	})
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envOrDefaultFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}
