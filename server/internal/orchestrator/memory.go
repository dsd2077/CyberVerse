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
	Recall(ctx context.Context, query string, scope conversationMemoryScope) ([]string, error)
	Retain(ctx context.Context, content string, contextLabel string, scope conversationMemoryScope) error
}

type conversationMemoryScope struct {
	UserID      string
	CharacterID string
	SessionID   string
	Source      string
	TurnID      string
}

type hindsightConversationMemory struct {
	baseURL            string
	apiKey             string
	bankID             string
	bankIDTemplate     string
	userID             string
	userTag            string
	timeout            time.Duration
	recallBudget       string
	maxResults         int
	maxTokens          int
	retainContext      string
	documentIDTemplate string
	defaultTags        []string
	retainMaxChars     int
	localFallback      bool
	localFallbackPath  string
	httpClient         *http.Client
}

type localMemoryRecord struct {
	CreatedAt string   `json:"created_at"`
	Context   string   `json:"context"`
	Tags      []string `json:"tags,omitempty"`
	Content   string   `json:"content"`
}

var asciiRecallTermRE = regexp.MustCompile(`[a-z0-9_-]{3,}`)
var unsafeMemoryIDRE = regexp.MustCompile(`[^a-zA-Z0-9_.:-]+`)

func newHindsightConversationMemoryFromEnv() conversationMemoryClient {
	if !envBool("HINDSIGHT_ENABLED", true) {
		return nil
	}
	apiKey := cleanEnvValue(os.Getenv("HINDSIGHT_API_KEY"))
	userID := cleanEnvValue(os.Getenv("HINDSIGHT_USER_ID"))
	userTag := cleanEnvValue(os.Getenv("HINDSIGHT_USER_TAG"))
	baseURL := strings.TrimRight(cleanEnvValue(os.Getenv("HINDSIGHT_BASE_URL")), "/")
	if baseURL == "" {
		baseURL = "https://hindsight.jmsu.top"
	}
	bankIDTemplate := cleanEnvValue(os.Getenv("HINDSIGHT_BANK_ID_TEMPLATE"))
	bankID := cleanEnvValue(os.Getenv("HINDSIGHT_BANK_ID"))
	if bankIDTemplate != "" {
		bankID = ""
	} else if bankID == "" {
		bankIDTemplate = "cv:user:{user_id}:character:{character_id}"
	}
	timeout := time.Duration(envInt("HINDSIGHT_TIMEOUT_SECONDS", 30, 1, 120)) * time.Second
	fallbackPath := cleanEnvValue(os.Getenv("HINDSIGHT_LOCAL_FALLBACK_PATH"))
	if fallbackPath == "" {
		fallbackPath = filepath.Join(".", "data", "persona_memory_shadow.jsonl")
	}
	tags := configTags(os.Getenv("HINDSIGHT_TAGS"))
	return &hindsightConversationMemory{
		baseURL:            baseURL,
		apiKey:             apiKey,
		bankID:             bankID,
		bankIDTemplate:     bankIDTemplate,
		userID:             userID,
		userTag:            userTag,
		timeout:            timeout,
		recallBudget:       cleanEnvValue(os.Getenv("HINDSIGHT_RECALL_BUDGET")),
		maxResults:         envInt("HINDSIGHT_RECALL_MAX_RESULTS", 5, 1, 20),
		maxTokens:          envInt("HINDSIGHT_RECALL_MAX_TOKENS", 4096, 256, 32768),
		retainContext:      cleanEnvValue(os.Getenv("HINDSIGHT_RETAIN_CONTEXT")),
		documentIDTemplate: cleanEnvValue(os.Getenv("HINDSIGHT_DOCUMENT_ID_TEMPLATE")),
		defaultTags:        tags,
		retainMaxChars:     envInt("HINDSIGHT_RETAIN_MAX_CHARS", 6000, 500, 50000),
		localFallback:      envBool("HINDSIGHT_LOCAL_FALLBACK_ENABLED", true),
		localFallbackPath:  fallbackPath,
		httpClient:         &http.Client{Timeout: timeout},
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

func configTags(value string) []string {
	parts := strings.Split(value, ",")
	tags := []string{"source:cyberverse"}
	seen := map[string]struct{}{"source:cyberverse": {}}
	for _, part := range parts {
		tag := cleanEnvValue(part)
		if tag == "" {
			continue
		}
		if _, ok := seen[tag]; ok {
			continue
		}
		seen[tag] = struct{}{}
		tags = append(tags, tag)
	}
	return tags
}

func safeMemoryID(value string, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		value = fallback
	}
	value = unsafeMemoryIDRE.ReplaceAllString(value, "-")
	if value == "" {
		value = fallback
	}
	if len(value) > 128 {
		return value[:128]
	}
	return value
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
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

func (h *hindsightConversationMemory) scopeValues(scope conversationMemoryScope) map[string]string {
	userID := scope.UserID
	if userID == "" {
		userID = h.userID
	}
	if userID == "" {
		userID = h.userTag
	}
	if userID == "" {
		userID = scope.SessionID
	}
	source := scope.Source
	if source == "" {
		source = "standard"
	}
	return map[string]string{
		"user_id":      safeMemoryID(userID, "anonymous"),
		"character_id": safeMemoryID(scope.CharacterID, "default-character"),
		"session_id":   safeMemoryID(scope.SessionID, "unknown-session"),
		"source":       safeMemoryID(source, "standard"),
		"turn_id":      safeMemoryID(scope.TurnID, "unknown-turn"),
	}
}

func formatMemoryTemplate(template string, values map[string]string) string {
	replacer := strings.NewReplacer(
		"{user_id}", values["user_id"],
		"{character_id}", values["character_id"],
		"{session_id}", values["session_id"],
		"{source}", values["source"],
		"{turn_id}", values["turn_id"],
	)
	return replacer.Replace(template)
}

func (h *hindsightConversationMemory) resolveBankID(scope conversationMemoryScope) string {
	if h.bankID != "" {
		return h.bankID
	}
	template := h.bankIDTemplate
	if template == "" {
		template = "cv:user:{user_id}:character:{character_id}"
	}
	return formatMemoryTemplate(template, h.scopeValues(scope))
}

func (h *hindsightConversationMemory) tags(scope conversationMemoryScope) []string {
	values := h.scopeValues(scope)
	tags := append([]string(nil), h.defaultTags...)
	tags = append(tags,
		"user:"+values["user_id"],
		"character:"+values["character_id"],
		"session:"+values["session_id"],
		"source:"+values["source"],
	)
	if h.userTag != "" {
		tags = append(tags, h.userTag)
	}
	seen := make(map[string]struct{}, len(tags))
	deduped := make([]string, 0, len(tags))
	for _, tag := range tags {
		tag = strings.TrimSpace(tag)
		if tag == "" {
			continue
		}
		if _, ok := seen[tag]; ok {
			continue
		}
		seen[tag] = struct{}{}
		deduped = append(deduped, tag)
	}
	return deduped
}

func (h *hindsightConversationMemory) documentID(scope conversationMemoryScope) string {
	template := h.documentIDTemplate
	if template == "" {
		template = "session:{session_id}:turn:{turn_id}"
	}
	return formatMemoryTemplate(template, h.scopeValues(scope))
}

func (h *hindsightConversationMemory) memoryURL(suffix string, scope conversationMemoryScope) string {
	return h.baseURL + "/v1/default/banks/" + h.resolveBankID(scope) + "/memories" + suffix
}

func (h *hindsightConversationMemory) Recall(ctx context.Context, query string, scope conversationMemoryScope) ([]string, error) {
	query = normalizeMemoryQuery(query)
	if query == "" {
		return nil, nil
	}
	local := h.recallLocal(query)
	if len(local) > 0 && explicitMemoryWrite(local[0]) {
		return local, nil
	}

	body := map[string]any{
		"query":       query,
		"types":       []string{"world", "experience"},
		"budget":      firstNonEmpty(h.recallBudget, "low"),
		"max_results": h.maxResults,
		"max_tokens":  h.maxTokens,
		"tags":        h.tags(scope),
		"tags_match":  "any",
	}
	data, err := json.Marshal(body)
	if err != nil {
		return local, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.memoryURL("/recall", scope), bytes.NewReader(data))
	if err != nil {
		return local, err
	}
	if h.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+h.apiKey)
	}
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

func (h *hindsightConversationMemory) Retain(ctx context.Context, content string, contextLabel string, scope conversationMemoryScope) error {
	content = clipMemoryContent(content, h.retainMaxChars)
	if content == "" {
		return nil
	}
	if contextLabel == "" {
		contextLabel = firstNonEmpty(h.retainContext, "cyberverse realtime conversation")
	}
	h.appendLocal(content, contextLabel)

	body := map[string]any{
		"items": []map[string]any{
			{
				"content":     content,
				"context":     contextLabel,
				"tags":        h.tags(scope),
				"metadata":    map[string]string{"session_id": scope.SessionID, "character_id": scope.CharacterID, "source": firstNonEmpty(scope.Source, "standard")},
				"document_id": h.documentID(scope),
			},
		},
		"async": true,
	}
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.memoryURL("", scope), bytes.NewReader(data))
	if err != nil {
		return err
	}
	if h.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+h.apiKey)
	}
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
