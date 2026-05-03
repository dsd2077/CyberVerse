package api

import (
	"net/http"
	"sort"
	"strings"

	"github.com/cyberverse/server/internal/config"
	"gopkg.in/yaml.v3"
)

type componentOption struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Model     string `json:"model"`
	Default   bool   `json:"default"`
	Available bool   `json:"available"`
}

type componentsResponse struct {
	LLM []componentOption `json:"llm"`
	ASR []componentOption `json:"asr"`
	TTS []componentOption `json:"tts"`
}

func (r *Router) handleListComponents(w http.ResponseWriter, req *http.Request) {
	if r.configPath == "" {
		writeJSON(w, http.StatusOK, r.defaultComponentsResponse())
		return
	}

	doc, err := config.ReadYAMLNode(r.configPath)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, componentsResponse{
		LLM: r.componentOptions(doc, "llm", r.pipelineDefault("llm")),
		ASR: r.componentOptions(doc, "asr", r.pipelineDefault("asr")),
		TTS: r.componentOptions(doc, "tts", r.pipelineDefault("tts")),
	})
}

func (r *Router) defaultComponentsResponse() componentsResponse {
	return componentsResponse{
		LLM: []componentOption{{ID: r.pipelineDefault("llm"), Name: displayComponentName(r.pipelineDefault("llm")), Default: true, Available: true}},
		ASR: []componentOption{{ID: r.pipelineDefault("asr"), Name: displayComponentName(r.pipelineDefault("asr")), Default: true, Available: true}},
		TTS: []componentOption{{ID: r.pipelineDefault("tts"), Name: displayComponentName(r.pipelineDefault("tts")), Default: true, Available: true}},
	}
}

func (r *Router) pipelineDefault(category string) string {
	if r != nil && r.cfg != nil {
		switch category {
		case "llm":
			if r.cfg.Pipeline.DefaultLLM != "" {
				return r.cfg.Pipeline.DefaultLLM
			}
		case "asr":
			if r.cfg.Pipeline.DefaultASR != "" {
				return r.cfg.Pipeline.DefaultASR
			}
		case "tts":
			if r.cfg.Pipeline.DefaultTTS != "" {
				return r.cfg.Pipeline.DefaultTTS
			}
		}
	}
	return "qwen"
}

func (r *Router) componentOptions(doc *yaml.Node, category string, fallbackDefault string) []componentOption {
	section, err := config.GetNodeAtPath(doc, "inference."+category)
	if err != nil || section.Kind != yaml.MappingNode {
		return []componentOption{{
			ID:        fallbackDefault,
			Name:      displayComponentName(fallbackDefault),
			Default:   true,
			Available: true,
		}}
	}

	defaultID := fallbackDefault
	if n := mappingValue(section, "default"); n != nil {
		if v := strings.TrimSpace(config.NodeScalarValue(n, true)); v != "" {
			defaultID = v
		}
	}

	options := make([]componentOption, 0)
	for i := 0; i < len(section.Content)-1; i += 2 {
		id := section.Content[i].Value
		if id == "default" {
			continue
		}
		node := section.Content[i+1]
		if node.Kind != yaml.MappingNode {
			continue
		}
		pluginClass := scalarAt(node, "plugin_class")
		if pluginClass == "" {
			continue
		}
		options = append(options, componentOption{
			ID:        id,
			Name:      displayComponentName(id),
			Model:     componentModel(node, category),
			Default:   id == defaultID,
			Available: pluginClass != "",
		})
	}

	sort.SliceStable(options, func(i, j int) bool {
		if options[i].Default != options[j].Default {
			return options[i].Default
		}
		return options[i].ID < options[j].ID
	})
	return options
}

func mappingValue(node *yaml.Node, key string) *yaml.Node {
	if node == nil || node.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i < len(node.Content)-1; i += 2 {
		if node.Content[i].Value == key {
			return node.Content[i+1]
		}
	}
	return nil
}

func scalarAt(node *yaml.Node, key string) string {
	child := mappingValue(node, key)
	if child == nil {
		return ""
	}
	return strings.TrimSpace(config.NodeScalarValue(child, true))
}

func componentModel(node *yaml.Node, category string) string {
	for _, key := range []string{"model", "model_size"} {
		if value := scalarAt(node, key); value != "" {
			return value
		}
	}
	return category
}

func displayComponentName(id string) string {
	switch strings.ToLower(strings.TrimSpace(id)) {
	case "qwen":
		return "Qwen"
	case "openai":
		return "OpenAI"
	case "whisper":
		return "Whisper"
	default:
		if id == "" {
			return "Qwen"
		}
		return strings.ToUpper(id[:1]) + id[1:]
	}
}
