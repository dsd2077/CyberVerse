package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

const testZhihuRedirectURI = "http://localhost:5173/kanshan/oauth/callback"

func setTestZhihuEnv(t *testing.T) {
	t.Helper()
	t.Setenv("ZHIHU_APP_ID", "test-app-id")
	t.Setenv("ZHIHU_APP_KEY", "test-app-key")
	t.Setenv("ZHIHU_REDIRECT_URI", testZhihuRedirectURI)
}

func createZhihuOAuthState(t *testing.T, r *Router) string {
	t.Helper()
	req := httptest.NewRequest("GET", "/api/v1/auth/zhihu/url?redirect_uri="+url.QueryEscape(testZhihuRedirectURI), nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp zhihuAuthorizeURLResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.State == "" {
		t.Fatal("expected non-empty OAuth state")
	}
	return resp.State
}

func TestZhihuAuthURLRequiresConfig(t *testing.T) {
	t.Setenv("ZHIHU_APP_ID", "")
	t.Setenv("ZHIHU_APP_KEY", "")
	t.Setenv("ZHIHU_OAUTH_APP_ID", "")
	t.Setenv("ZHIHU_OAUTH_APP_KEY", "")
	t.Setenv("ZHIHU_REDIRECT_URI", "")

	r := newTestRouter()
	req := httptest.NewRequest("GET", "/api/v1/auth/zhihu/url?redirect_uri="+url.QueryEscape(testZhihuRedirectURI), nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}

func TestZhihuAuthURLAcceptsOAuthEnvAliases(t *testing.T) {
	t.Setenv("ZHIHU_APP_ID", "")
	t.Setenv("ZHIHU_APP_KEY", "")
	t.Setenv("ZHIHU_OAUTH_APP_ID", "alias-app-id")
	t.Setenv("ZHIHU_OAUTH_APP_KEY", "alias-app-key")
	t.Setenv("ZHIHU_REDIRECT_URI", testZhihuRedirectURI)

	r := newTestRouter()
	req := httptest.NewRequest("GET", "/api/v1/auth/zhihu/url?redirect_uri="+url.QueryEscape(testZhihuRedirectURI), nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp zhihuAuthorizeURLResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	parsed, err := url.Parse(resp.AuthorizeURL)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Query().Get("app_id") != "alias-app-id" {
		t.Fatalf("expected alias app id, got %q", parsed.Query().Get("app_id"))
	}
}

func TestZhihuAuthURLBuildsAuthorizeURL(t *testing.T) {
	setTestZhihuEnv(t)

	r := newTestRouter()
	req := httptest.NewRequest("GET", "/api/v1/auth/zhihu/url?redirect_uri="+url.QueryEscape(testZhihuRedirectURI), nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp zhihuAuthorizeURLResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	parsed, err := url.Parse(resp.AuthorizeURL)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Scheme != "https" || parsed.Host != "openapi.zhihu.com" || parsed.Path != "/authorize" {
		t.Fatalf("unexpected authorize URL: %s", resp.AuthorizeURL)
	}
	q := parsed.Query()
	if q.Get("app_id") != "test-app-id" {
		t.Fatalf("unexpected app_id: %q", q.Get("app_id"))
	}
	if q.Get("redirect_uri") != testZhihuRedirectURI {
		t.Fatalf("unexpected redirect_uri: %q", q.Get("redirect_uri"))
	}
	if q.Get("response_type") != "code" {
		t.Fatalf("unexpected response_type: %q", q.Get("response_type"))
	}
	if q.Get("state") == "" || q.Get("state") != resp.State {
		t.Fatalf("state mismatch: query=%q body=%q", q.Get("state"), resp.State)
	}
}

func TestZhihuAuthURLRejectsUnexpectedRedirectURI(t *testing.T) {
	setTestZhihuEnv(t)

	r := newTestRouter()
	req := httptest.NewRequest("GET", "/api/v1/auth/zhihu/url?redirect_uri="+url.QueryEscape("http://evil.example/callback"), nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

func TestZhihuCallbackRejectsInvalidState(t *testing.T) {
	setTestZhihuEnv(t)

	r := newTestRouter()
	body := `{"code":"code-1","state":"missing","redirect_uri":"` + testZhihuRedirectURI + `"}`
	req := httptest.NewRequest("POST", "/api/v1/auth/zhihu/callback", strings.NewReader(body))
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
}

func TestZhihuCallbackRejectsExpiredState(t *testing.T) {
	setTestZhihuEnv(t)

	r := newTestRouter()
	state := createZhihuOAuthState(t, r)

	r.zhihuAuth.mu.Lock()
	r.zhihuAuth.states[state] = zhihuOAuthState{RedirectURI: testZhihuRedirectURI, ExpiresAt: time.Now().Add(-time.Second)}
	r.zhihuAuth.mu.Unlock()

	body := `{"code":"code-1","state":"` + state + `","redirect_uri":"` + testZhihuRedirectURI + `"}`
	req := httptest.NewRequest("POST", "/api/v1/auth/zhihu/callback", strings.NewReader(body))
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
}

func TestZhihuCallbackCreatesSessionAndMeReturnsUser(t *testing.T) {
	setTestZhihuEnv(t)

	zhihuAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		switch req.URL.Path {
		case "/access_token":
			if req.Method != http.MethodPost {
				t.Fatalf("expected POST token request, got %s", req.Method)
			}
			if got := req.Header.Get("Content-Type"); got != "application/x-www-form-urlencoded" {
				t.Fatalf("unexpected Content-Type: %q", got)
			}
			if err := req.ParseForm(); err != nil {
				t.Fatal(err)
			}
			if req.Form.Get("app_id") != "test-app-id" || req.Form.Get("app_key") != "test-app-key" || req.Form.Get("code") != "code-1" {
				t.Fatalf("unexpected token form: %+v", req.Form)
			}
			if req.Form.Get("redirect_uri") != testZhihuRedirectURI || req.Form.Get("grant_type") != "authorization_code" {
				t.Fatalf("unexpected token form: %+v", req.Form)
			}
			writeJSON(w, http.StatusOK, map[string]any{
				"access_token": "token-1",
				"token_type":   "bearer",
				"code":         20000,
				"expires_in":   2592000,
			})
		case "/user":
			if req.Header.Get("Authorization") != "Bearer token-1" {
				t.Fatalf("unexpected Authorization header: %q", req.Header.Get("Authorization"))
			}
			writeJSON(w, http.StatusOK, zhihuUser{
				UID:        23456789876,
				Fullname:   "用户昵称",
				Headline:   "个人简介",
				AvatarPath: "https://picl.zhimg.com/avatar.jpg",
				Email:      "user@example.com",
			})
		default:
			http.NotFound(w, req)
		}
	}))
	defer zhihuAPI.Close()

	r := newTestRouter()
	r.zhihuAuth.client = zhihuAPI.Client()
	r.zhihuAuth.tokenEndpoint = zhihuAPI.URL + "/access_token"
	r.zhihuAuth.userEndpoint = zhihuAPI.URL + "/user"
	state := createZhihuOAuthState(t, r)

	body := `{"code":"code-1","state":"` + state + `","redirect_uri":"` + testZhihuRedirectURI + `"}`
	req := httptest.NewRequest("POST", "/api/v1/auth/zhihu/callback", strings.NewReader(body))
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	cookies := w.Result().Cookies()
	if len(cookies) != 1 || cookies[0].Name != zhihuSessionCookie || !cookies[0].HttpOnly {
		t.Fatalf("expected HttpOnly Zhihu session cookie, got %+v", cookies)
	}

	meReq := httptest.NewRequest("GET", "/api/v1/auth/zhihu/me", nil)
	meReq.AddCookie(cookies[0])
	meW := httptest.NewRecorder()
	r.Handler().ServeHTTP(meW, meReq)

	if meW.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", meW.Code, meW.Body.String())
	}
	var me zhihuMeResponse
	if err := json.NewDecoder(meW.Body).Decode(&me); err != nil {
		t.Fatal(err)
	}
	if !me.Authenticated || me.User.Fullname != "用户昵称" || me.User.UID != 23456789876 {
		t.Fatalf("unexpected me response: %+v", me)
	}
}

func TestZhihuAPIErrorTreatsCode20000AsSuccess(t *testing.T) {
	raw := json.RawMessage(`{"access_token":"token-1","token_type":"bearer","code":20000,"expires_in":2592000}`)
	if err := zhihuAPIError(raw); err != nil {
		t.Fatalf("expected code 20000 to be success, got %v", err)
	}
}

func TestZhihuAPIErrorDoesNotLeakRawTokenBody(t *testing.T) {
	raw := json.RawMessage(`{"access_token":"secret-token","code":401,"data":{"message":"Access token is not valid"}}`)
	err := zhihuAPIError(raw)
	if err == nil {
		t.Fatal("expected error")
	}
	if strings.Contains(err.Error(), "secret-token") {
		t.Fatalf("error leaked token body: %v", err)
	}
	if !strings.Contains(err.Error(), "Access token is not valid") {
		t.Fatalf("expected safe error message, got %v", err)
	}
}

func TestZhihuLogoutClearsSession(t *testing.T) {
	setTestZhihuEnv(t)

	r := newTestRouter()
	expiresAt := time.Now().Add(time.Hour)
	r.zhihuAuth.storeSession("session-1", zhihuSession{
		AccessToken: "token-1",
		User:        zhihuUser{UID: 1, Fullname: "用户昵称"},
		ExpiresAt:   expiresAt,
	})

	req := httptest.NewRequest("POST", "/api/v1/auth/zhihu/logout", nil)
	req.AddCookie(&http.Cookie{Name: zhihuSessionCookie, Value: "session-1"})
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	if _, ok := r.zhihuAuth.sessions["session-1"]; ok {
		t.Fatal("expected session to be deleted")
	}
	cookies := w.Result().Cookies()
	if len(cookies) != 1 || cookies[0].MaxAge >= 0 {
		t.Fatalf("expected expired cookie, got %+v", cookies)
	}
}
