package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/inference"
	ragstore "github.com/cyberverse/server/internal/rag"
)

func createKnowledgeTestCharacter(t *testing.T, r *Router) *character.Character {
	t.Helper()
	char, err := r.charStore.Create(&character.Character{Name: "知识角色"})
	if err != nil {
		t.Fatalf("Create character: %v", err)
	}
	return char
}

func decodeKnowledgeUploadResponse(t *testing.T, w *httptest.ResponseRecorder) uploadKnowledgeFilesResponse {
	t.Helper()
	var resp uploadKnowledgeFilesResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode knowledge upload response: %v", err)
	}
	return resp
}

func waitForKnowledgeSourceStatus(t *testing.T, r *Router, characterID, sourceID string, status ragstore.SourceStatus) ragstore.Source {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		source, err := r.ragStore.Get(characterID, sourceID)
		if err == nil && source.Status == status {
			return *source
		}
		time.Sleep(10 * time.Millisecond)
	}
	source, err := r.ragStore.Get(characterID, sourceID)
	if err != nil {
		t.Fatalf("Get knowledge source: %v", err)
	}
	t.Fatalf("expected source status %q, got %+v", status, source)
	return ragstore.Source{}
}

func addMultipartFile(t *testing.T, writer *multipart.Writer, fieldName, relativePath, content string) {
	t.Helper()
	addMultipartBytes(t, writer, fieldName, relativePath, []byte(content))
}

func addMultipartBytes(t *testing.T, writer *multipart.Writer, fieldName, relativePath string, content []byte) {
	t.Helper()
	filename := relativePath
	if idx := strings.LastIndex(relativePath, "/"); idx >= 0 {
		filename = relativePath[idx+1:]
	}
	part, err := writer.CreateFormFile(fieldName, filename)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(content); err != nil {
		t.Fatal(err)
	}
	if err := writer.WriteField("relative_paths", relativePath); err != nil {
		t.Fatal(err)
	}
}

func TestKnowledgeTextImportRouteIsDisabled(t *testing.T) {
	r := newTestRouter()
	char := createKnowledgeTestCharacter(t, r)

	req := httptest.NewRequest(
		"POST",
		"/api/v1/characters/"+char.ID+"/knowledge",
		strings.NewReader(`{"content":"不再支持文本导入"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	if w.Code == http.StatusCreated {
		t.Fatalf("expected text import to be disabled, got %d: %s", w.Code, w.Body.String())
	}
}

func TestKnowledgeFolderUploadPreservesStructureAndIndexesOnlyIndexable(t *testing.T) {
	inf := &fakeInferenceService{
		ragIndexRequests:   make(chan inference.RAGIndexSourceRequest, 2),
		ragIndexChunkCount: 3,
	}
	r := newTestRouterWithInference(inf)
	char := createKnowledgeTestCharacter(t, r)

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	addMultipartFile(t, writer, "files", "角色资料/docs/notes.md", "# 设定\n角色喜欢研究星图。")
	addMultipartFile(t, writer, "files", "角色资料/profile.txt", "角色出生在海边。")
	addMultipartFile(t, writer, "files", "角色资料/images/avatar.png", "png")
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest("POST", "/api/v1/characters/"+char.ID+"/knowledge/files", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", w.Code, w.Body.String())
	}
	resp := decodeKnowledgeUploadResponse(t, w)
	if len(resp.Sources) != 3 {
		t.Fatalf("expected 3 uploaded sources, got %+v", resp.Sources)
	}
	if len(resp.Skipped) != 0 {
		t.Fatalf("expected no skipped files, got %+v", resp.Skipped)
	}
	if resp.Sources[0].Type != "" || resp.Sources[0].Title == "" || resp.Sources[0].RelativePath == "" || resp.Sources[0].StoredPath == "" {
		t.Fatalf("expected unified source metadata, got %+v", resp.Sources[0])
	}

	seen := map[string]bool{}
	for i := 0; i < 2; i++ {
		select {
		case indexReq := <-inf.ragIndexRequests:
			seen[indexReq.Filename] = true
			if indexReq.SourceType != "" {
				t.Fatalf("expected empty source type in index request, got %+v", indexReq)
			}
			data, err := os.ReadFile(indexReq.SourcePath)
			if err != nil {
				t.Fatalf("read uploaded source: %v", err)
			}
			if len(data) == 0 {
				t.Fatal("expected uploaded source file to contain data")
			}
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for RAG index request")
		}
	}
	if !seen["notes.md"] || !seen["profile.txt"] {
		t.Fatalf("missing index requests for uploaded files: %+v", seen)
	}
	imageSaved := false
	for _, source := range resp.Sources {
		ready := waitForKnowledgeSourceStatus(t, r, char.ID, source.ID, ragstore.SourceStatusReady)
		if source.Indexable && ready.ChunkCount != 3 {
			t.Fatalf("unexpected indexed source: %+v", ready)
		}
		if !source.Indexable && ready.ChunkCount != 0 {
			t.Fatalf("unexpected stored-only source: %+v", ready)
		}
		if source.RelativePath == "角色资料/images/avatar.png" {
			imageSaved = true
			path, err := r.ragStore.SourcePath(char.ID, &source)
			if err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if string(data) != "png" || source.StoredPath != "sources/角色资料/images/avatar.png" {
				t.Fatalf("unexpected saved image source: path=%s source=%+v data=%q", path, source, string(data))
			}
		}
	}
	if !imageSaved {
		t.Fatalf("expected nested image to be saved, got %+v", resp.Sources)
	}
	select {
	case extra := <-inf.ragIndexRequests:
		t.Fatalf("expected image to skip RAG indexing, got %+v", extra)
	case <-time.After(50 * time.Millisecond):
	}
}

func TestKnowledgeUploadDoesNotEnforceOldFileSizeLimit(t *testing.T) {
	r := newTestRouter()
	char := createKnowledgeTestCharacter(t, r)

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	content := bytes.Repeat([]byte("x"), 21*1024*1024)
	addMultipartBytes(t, writer, "files", "large/blob.bin", content)
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest("POST", "/api/v1/characters/"+char.ID+"/knowledge/files", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", w.Code, w.Body.String())
	}
	resp := decodeKnowledgeUploadResponse(t, w)
	if len(resp.Sources) != 1 || resp.Sources[0].Status != ragstore.SourceStatusReady || resp.Sources[0].Indexable {
		t.Fatalf("expected one stored-only ready source, got %+v", resp.Sources)
	}
	path, err := r.ragStore.SourcePath(char.ID, &resp.Sources[0])
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() != int64(len(content)) {
		t.Fatalf("expected saved file size %d, got %d", len(content), info.Size())
	}
}

func TestKnowledgeUploadDoesNotUseParseMultipartFormPartLimit(t *testing.T) {
	r := newTestRouter()
	char := createKnowledgeTestCharacter(t, r)

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	const fileCount = 505
	for i := 0; i < fileCount; i++ {
		addMultipartFile(t, writer, "files", fmt.Sprintf("folder/file-%03d.bin", i), "x")
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest("POST", "/api/v1/characters/"+char.ID+"/knowledge/files", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", w.Code, w.Body.String())
	}
	resp := decodeKnowledgeUploadResponse(t, w)
	if len(resp.Sources) != fileCount || len(resp.Skipped) != 0 {
		t.Fatalf("expected %d uploaded sources without skipped files, got sources=%d skipped=%+v", fileCount, len(resp.Sources), resp.Skipped)
	}
}

func TestKnowledgeUploadUnsafeRelativePathReturnsBadRequestWithSkippedFiles(t *testing.T) {
	r := newTestRouter()
	char := createKnowledgeTestCharacter(t, r)

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	addMultipartFile(t, writer, "files", "../script.exe", "nope")
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest("POST", "/api/v1/characters/"+char.ID+"/knowledge/files", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
	var resp struct {
		Error   string                 `json:"error"`
		Skipped []skippedKnowledgeFile `json:"skipped"`
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.Error == "" || len(resp.Skipped) != 1 {
		t.Fatalf("expected skipped files in error response, got %+v", resp)
	}
}

func TestKnowledgeUploadDuplicateRelativePathReusesSource(t *testing.T) {
	inf := &fakeInferenceService{
		ragIndexRequests:   make(chan inference.RAGIndexSourceRequest, 2),
		ragIndexChunkCount: 2,
	}
	r := newTestRouterWithInference(inf)
	char := createKnowledgeTestCharacter(t, r)

	upload := func(content string) ragstore.Source {
		var body bytes.Buffer
		writer := multipart.NewWriter(&body)
		addMultipartFile(t, writer, "files", "docs/profile.md", content)
		if err := writer.Close(); err != nil {
			t.Fatal(err)
		}
		req := httptest.NewRequest("POST", "/api/v1/characters/"+char.ID+"/knowledge/files", &body)
		req.Header.Set("Content-Type", writer.FormDataContentType())
		w := httptest.NewRecorder()
		r.Handler().ServeHTTP(w, req)
		if w.Code != http.StatusCreated {
			t.Fatalf("expected 201, got %d: %s", w.Code, w.Body.String())
		}
		resp := decodeKnowledgeUploadResponse(t, w)
		if len(resp.Sources) != 1 {
			t.Fatalf("expected one source, got %+v", resp.Sources)
		}
		select {
		case <-inf.ragIndexRequests:
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for RAG index request")
		}
		waitForKnowledgeSourceStatus(t, r, char.ID, resp.Sources[0].ID, ragstore.SourceStatusReady)
		return resp.Sources[0]
	}

	first := upload("first")
	second := upload("second")
	if first.ID != second.ID {
		t.Fatalf("expected duplicate relative path to reuse source ID, first=%s second=%s", first.ID, second.ID)
	}
	path, err := r.ragStore.SourcePath(char.ID, &second)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "second" {
		t.Fatalf("expected overwritten file content, got %q", string(data))
	}
}

func TestKnowledgeReindexAndDelete(t *testing.T) {
	inf := &fakeInferenceService{
		ragIndexRequests:   make(chan inference.RAGIndexSourceRequest, 2),
		ragDeleteRequests:  make(chan string, 1),
		ragIndexChunkCount: 4,
	}
	r := newTestRouterWithInference(inf)
	char := createKnowledgeTestCharacter(t, r)

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	addMultipartFile(t, writer, "file", "folder/rules.txt", "只回答导入事实。")
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest("POST", "/api/v1/characters/"+char.ID+"/knowledge/files", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", w.Code, w.Body.String())
	}
	resp := decodeKnowledgeUploadResponse(t, w)
	if len(resp.Sources) != 1 {
		t.Fatalf("expected one source, got %+v", resp.Sources)
	}
	created := resp.Sources[0]
	select {
	case <-inf.ragIndexRequests:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for initial RAG index request")
	}
	waitForKnowledgeSourceStatus(t, r, char.ID, created.ID, ragstore.SourceStatusReady)

	req = httptest.NewRequest("POST", "/api/v1/characters/"+char.ID+"/knowledge/"+created.ID+"/reindex", nil)
	w = httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusAccepted {
		t.Fatalf("expected 202, got %d: %s", w.Code, w.Body.String())
	}
	var reindexed ragstore.Source
	if err := json.NewDecoder(w.Body).Decode(&reindexed); err != nil {
		t.Fatal(err)
	}
	if reindexed.Status != ragstore.SourceStatusIndexing {
		t.Fatalf("expected indexing status after reindex, got %+v", reindexed)
	}
	select {
	case indexReq := <-inf.ragIndexRequests:
		if indexReq.SourceID != created.ID {
			t.Fatalf("unexpected reindex request: %+v", indexReq)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for reindex request")
	}
	waitForKnowledgeSourceStatus(t, r, char.ID, created.ID, ragstore.SourceStatusReady)

	req = httptest.NewRequest("DELETE", "/api/v1/characters/"+char.ID+"/knowledge/"+created.ID, nil)
	w = httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusNoContent {
		t.Fatalf("expected 204, got %d: %s", w.Code, w.Body.String())
	}
	select {
	case sourceID := <-inf.ragDeleteRequests:
		if sourceID != created.ID {
			t.Fatalf("unexpected delete request source ID: %s", sourceID)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for RAG delete request")
	}
	sources, err := r.ragStore.List(char.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) != 0 {
		t.Fatalf("expected deleted source to be removed, got %+v", sources)
	}
	path := r.charStore.CharDir(char.ID) + "/knowledge/sources/folder/rules.txt"
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("expected deleted file to be removed, stat err=%v", err)
	}
}
