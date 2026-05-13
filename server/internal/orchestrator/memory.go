package orchestrator

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

type conversationMemoryClient interface {
	Recall(ctx context.Context, query string) ([]string, error)
	Retain(ctx context.Context, content string, contextLabel string) error
}

type hindsightConversationMemory struct {
	baseURL           string
	apiKey            string
	bankID            string
	userTag           string
	timeout           time.Duration
	maxResults        int
	maxTokens         int
	retainMaxChars    int
	localFallback     bool
	localFallbackPath string
	httpClient        *http.Client
}

type localMemoryRecord struct {
	CreatedAt string   `json:"created_at"`
	Context   string   `json:"context"`
	Tags      []string `json:"tags,omitempty"`
	Content   string   `json:"content"`
}

var asciiRecallTermRE = regexp.MustCompile(`[a-z0-9_-]{3,}`)

func newHindsightConversationMemoryFromEnv() conversationMemoryClient {
	if !envBool("HINDSIGHT_ENABLED", true) {
		return nil
	}
	apiKey := cleanEnvValue(os.Getenv("HINDSIGHT_API_KEY"))
	userTag := cleanEnvValue(os.Getenv("HINDSIGHT_USER_TAG"))
	if apiKey == "" || userTag == "" {
		return nil
	}
	baseURL := strings.TrimRight(cleanEnvValue(os.Getenv("HINDSIGHT_BASE_URL")), "/")
	if baseURL == "" {
		baseURL = "https://hindsight.lucky.jmsu.top"
	}
	bankID := cleanEnvValue(os.Getenv("HINDSIGHT_BANK_ID"))
	if bankID == "" {
		bankID = "openclaw"
	}
	timeout := time.Duration(envInt("HINDSIGHT_TIMEOUT_SECONDS", 30, 1, 120)) * time.Second
	fallbackPath := cleanEnvValue(os.Getenv("HINDSIGHT_LOCAL_FALLBACK_PATH"))
	if fallbackPath == "" {
		fallbackPath = filepath.Join(".", "data", "persona_memory_shadow.jsonl")
	}
	return &hindsightConversationMemory{
		baseURL:           baseURL,
		apiKey:            apiKey,
		bankID:            bankID,
		userTag:           userTag,
		timeout:           timeout,
		maxResults:        envInt("HINDSIGHT_RECALL_MAX_RESULTS", 5, 1, 20),
		maxTokens:         envInt("HINDSIGHT_RECALL_MAX_TOKENS", 4096, 256, 32768),
		retainMaxChars:    envInt("HINDSIGHT_RETAIN_MAX_CHARS", 6000, 500, 50000),
		localFallback:     envBool("HINDSIGHT_LOCAL_FALLBACK_ENABLED", true),
		localFallbackPath: fallbackPath,
		httpClient:        &http.Client{Timeout: timeout},
	}
}

func cleanEnvValue(value string) string {
	value = strings.TrimSpace(value)
	if strings.HasPrefix(value, "${") && strings.HasSuffix(value, "}") {
		return ""
	}
	return value
}

func envBool(name string, fallback bool) bool {
	value := strings.ToLower(cleanEnvValue(os.Getenv(name)))
	switch value {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

func envInt(name string, fallback int, minValue int, maxValue int) int {
	value := cleanEnvValue(os.Getenv(name))
	if value == "" {
		return fallback
	}
	var parsed int
	if _, err := fmt.Sscanf(value, "%d", &parsed); err != nil {
		return fallback
	}
	if parsed < minValue {
		return minValue
	}
	if parsed > maxValue {
		return maxValue
	}
	return parsed
}

func normalizeMemoryQuery(value string) string {
	fields := strings.FieldsFunc(strings.ToLower(strings.TrimSpace(value)), func(r rune) bool {
		return unicode.IsPunct(r) || unicode.IsSpace(r) || strings.ContainsRune("？。、《》“”‘’、；：！…（）【】", r)
	})
	return strings.Join(fields, " ")
}

func memoryTerms(value string) map[string]struct{} {
	normalized := normalizeMemoryQuery(value)
	terms := make(map[string]struct{})
	for _, term := range asciiRecallTermRE.FindAllString(normalized, -1) {
		terms[term] = struct{}{}
	}

	var runes []rune
	flushCJK := func() {
		if len(runes) >= 2 {
			text := string(runes)
			terms[text] = struct{}{}
			for size := 2; size <= 4; size++ {
				if len(runes) < size {
					continue
				}
				for index := 0; index <= len(runes)-size; index++ {
					terms[string(runes[index:index+size])] = struct{}{}
				}
			}
		}
		runes = nil
	}
	for _, r := range normalized {
		if unicode.Is(unicode.Han, r) {
			runes = append(runes, r)
			continue
		}
		flushCJK()
	}
	flushCJK()
	return terms
}

func clipMemoryContent(value string, limit int) string {
	value = strings.TrimSpace(value)
	if limit <= 0 || utf8.RuneCountInString(value) <= limit {
		return value
	}
	runes := []rune(value)
	return string(runes[:limit])
}

func (h *hindsightConversationMemory) memoryURL(suffix string) string {
	return h.baseURL + "/v1/default/banks/" + h.bankID + "/memories" + suffix
}

func (h *hindsightConversationMemory) Recall(ctx context.Context, query string) ([]string, error) {
	query = normalizeMemoryQuery(query)
	if query == "" {
		return nil, nil
	}
	local := h.recallLocal(query)
	if len(local) > 0 && explicitMemoryWrite(local[0]) {
		return local, nil
	}

	body := map[string]any{
		"query":      query,
		"types":      []string{"world", "experience"},
		"budget":     "mid",
		"max_tokens": h.maxTokens,
		"tags":       []string{h.userTag},
		"tags_match": "any",
	}
	data, err := json.Marshal(body)
	if err != nil {
		return local, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.memoryURL("/recall"), bytes.NewReader(data))
	if err != nil {
		return local, err
	}
	req.Header.Set("Authorization", "Bearer "+h.apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := h.httpClient.Do(req)
	if err != nil {
		return local, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return local, fmt.Errorf("hindsight recall status %d", resp.StatusCode)
	}
	var payload struct {
		Results []map[string]any `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return local, err
	}
	remote := make([]string, 0, len(payload.Results))
	for _, result := range payload.Results {
		if text := memoryText(result); text != "" {
			remote = append(remote, text)
		}
	}
	return mergeMemoryTexts(h.maxResults, local, remote), nil
}

func (h *hindsightConversationMemory) Retain(ctx context.Context, content string, contextLabel string) error {
	content = clipMemoryContent(content, h.retainMaxChars)
	if content == "" {
		return nil
	}
	if contextLabel == "" {
		contextLabel = "conversation"
	}
	h.appendLocal(content, contextLabel)

	body := map[string]any{
		"items": []map[string]any{
			{
				"content": content,
				"context": contextLabel,
				"tags":    []string{h.userTag},
			},
		},
		"async": true,
	}
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.memoryURL(""), bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+h.apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := h.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("hindsight retain status %d", resp.StatusCode)
	}
	return nil
}

func memoryText(value any) string {
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed)
	case map[string]any:
		for _, key := range []string{"text", "content"} {
			if text, ok := typed[key].(string); ok && strings.TrimSpace(text) != "" {
				return strings.TrimSpace(text)
			}
		}
		for _, key := range []string{"memory", "item", "document"} {
			if text := memoryText(typed[key]); text != "" {
				return text
			}
		}
	}
	return ""
}

func (h *hindsightConversationMemory) appendLocal(content string, contextLabel string) {
	if !h.localFallback || h.localFallbackPath == "" {
		return
	}
	record := localMemoryRecord{
		CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Context:   contextLabel,
		Tags:      []string{h.userTag},
		Content:   content,
	}
	if err := os.MkdirAll(filepath.Dir(h.localFallbackPath), 0o755); err != nil {
		log.Printf("hindsight local fallback mkdir failed: %v", err)
		return
	}
	file, err := os.OpenFile(h.localFallbackPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		log.Printf("hindsight local fallback retain failed: %v", err)
		return
	}
	defer file.Close()
	if err := json.NewEncoder(file).Encode(record); err != nil {
		log.Printf("hindsight local fallback encode failed: %v", err)
	}
}

func (h *hindsightConversationMemory) recallLocal(query string) []string {
	if !h.localFallback || h.localFallbackPath == "" {
		return nil
	}
	file, err := os.Open(h.localFallbackPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		log.Printf("hindsight local fallback recall failed: %v", err)
		return nil
	}
	defer file.Close()

	queryTerms := memoryTerms(query)
	type scoredMemory struct {
		score    int
		index    int
		text     string
		explicit bool
	}
	var scored []scoredMemory
	var explicit []scoredMemory
	scanner := bufio.NewScanner(file)
	index := 0
	for scanner.Scan() {
		var record localMemoryRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			index++
			continue
		}
		text := strings.TrimSpace(record.Content)
		score := overlapCount(queryTerms, memoryTerms(text))
		if score == 0 {
			index++
			continue
		}
		isExplicit := explicitMemoryWrite(text)
		if isExplicit {
			score += 100
		}
		item := scoredMemory{score: score, index: index, text: clipMemoryContent(text, 1200), explicit: isExplicit}
		scored = append(scored, item)
		if isExplicit {
			explicit = append(explicit, item)
		}
		index++
	}
	if err := scanner.Err(); err != nil {
		log.Printf("hindsight local fallback scan failed: %v", err)
	}
	if len(explicit) > 0 {
		sort.Slice(explicit, func(i, j int) bool {
			return explicit[i].index > explicit[j].index
		})
		return []string{explicit[0].text}
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].score == scored[j].score {
			return scored[i].index > scored[j].index
		}
		return scored[i].score > scored[j].score
	})
	memories := make([]string, 0, min(h.maxResults, len(scored)))
	seen := make(map[string]struct{})
	for _, item := range scored {
		if _, ok := seen[item.text]; ok {
			continue
		}
		seen[item.text] = struct{}{}
		memories = append(memories, item.text)
		if len(memories) >= h.maxResults {
			break
		}
	}
	return memories
}

func explicitMemoryWrite(text string) bool {
	normalized := strings.ToLower(strings.TrimSpace(text))
	return strings.Contains(normalized, "用户: 请记住") ||
		strings.Contains(normalized, "用户: 记住") ||
		strings.Contains(normalized, "user: remember") ||
		strings.Contains(normalized, "user: please remember")
}

func overlapCount(left map[string]struct{}, right map[string]struct{}) int {
	count := 0
	for term := range left {
		if _, ok := right[term]; ok {
			count++
		}
	}
	return count
}

func mergeMemoryTexts(limit int, groups ...[]string) []string {
	if limit <= 0 {
		limit = 5
	}
	merged := make([]string, 0, limit)
	seen := make(map[string]struct{})
	for _, group := range groups {
		for _, item := range group {
			text := strings.TrimSpace(item)
			if text == "" {
				continue
			}
			if _, ok := seen[text]; ok {
				continue
			}
			seen[text] = struct{}{}
			merged = append(merged, text)
			if len(merged) >= limit {
				return merged
			}
		}
	}
	return merged
}

func formatConversationMemory(memories []string) string {
	if len(memories) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("相关长期记忆：\n")
	for _, memory := range memories {
		if text := strings.TrimSpace(memory); text != "" {
			b.WriteString("- ")
			b.WriteString(text)
			b.WriteString("\n")
		}
	}
	return strings.TrimSpace(b.String())
}
