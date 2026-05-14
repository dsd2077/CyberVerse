package api

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	zhihuAuthorizeEndpoint = "https://openapi.zhihu.com/authorize"
	zhihuTokenEndpoint     = "https://openapi.zhihu.com/access_token"
	zhihuUserEndpoint      = "https://openapi.zhihu.com/user"

	zhihuOAuthStateTTL = 10 * time.Minute
	zhihuSessionCookie = "cyberverse_zhihu_session"
	zhihuDefaultExpiry = 30 * 24 * time.Hour
)

type zhihuAuthConfig struct {
	AppID       string
	AppKey      string
	RedirectURI string
}

type zhihuOAuthState struct {
	RedirectURI string
	ExpiresAt   time.Time
}

type zhihuSession struct {
	AccessToken string
	User        zhihuUser `json:"user"`
	ExpiresAt   time.Time
}

type zhihuUser struct {
	UID         int64  `json:"uid,omitempty"`
	HashID      string `json:"hash_id,omitempty"`
	Fullname    string `json:"fullname,omitempty"`
	Gender      string `json:"gender,omitempty"`
	Headline    string `json:"headline,omitempty"`
	Description string `json:"description,omitempty"`
	AvatarPath  string `json:"avatar_path,omitempty"`
	URL         string `json:"url,omitempty"`
	PhoneNo     string `json:"phone_no,omitempty"`
	Email       string `json:"email,omitempty"`
}

type zhihuTokenResponse struct {
	AccessToken string `json:"access_token"`
	TokenType   string `json:"token_type"`
	ExpiresIn   int64  `json:"expires_in"`
}

type zhihuAuthService struct {
	mu                sync.Mutex
	states            map[string]zhihuOAuthState
	sessions          map[string]zhihuSession
	client            *http.Client
	authorizeEndpoint string
	tokenEndpoint     string
	userEndpoint      string
	now               func() time.Time
}

func newZhihuAuthService() *zhihuAuthService {
	return &zhihuAuthService{
		states:            make(map[string]zhihuOAuthState),
		sessions:          make(map[string]zhihuSession),
		client:            &http.Client{Timeout: 10 * time.Second},
		authorizeEndpoint: zhihuAuthorizeEndpoint,
		tokenEndpoint:     zhihuTokenEndpoint,
		userEndpoint:      zhihuUserEndpoint,
		now:               time.Now,
	}
}

func loadZhihuAuthConfig() (zhihuAuthConfig, error) {
	cfg := zhihuAuthConfig{
		AppID:       firstEnv("ZHIHU_APP_ID", "ZHIHU_OAUTH_APP_ID"),
		AppKey:      firstEnv("ZHIHU_APP_KEY", "ZHIHU_OAUTH_APP_KEY"),
		RedirectURI: strings.TrimSpace(os.Getenv("ZHIHU_REDIRECT_URI")),
	}
	if cfg.AppID == "" || cfg.AppKey == "" || cfg.RedirectURI == "" {
		return cfg, errors.New("Zhihu OAuth is not configured")
	}
	return cfg, nil
}

func firstEnv(keys ...string) string {
	for _, key := range keys {
		if value := strings.TrimSpace(os.Getenv(key)); value != "" {
			return value
		}
	}
	return ""
}

type zhihuAuthorizeURLResponse struct {
	AuthorizeURL string `json:"authorize_url"`
	State        string `json:"state"`
}

type zhihuCallbackRequest struct {
	Code        string `json:"code"`
	State       string `json:"state"`
	RedirectURI string `json:"redirect_uri"`
}

type zhihuCallbackResponse struct {
	Authenticated bool      `json:"authenticated"`
	ExpiresIn     int64     `json:"expires_in"`
	User          zhihuUser `json:"user"`
}

type zhihuMeResponse struct {
	Authenticated bool      `json:"authenticated"`
	User          zhihuUser `json:"user"`
}

func (r *Router) handleZhihuAuthURL(w http.ResponseWriter, req *http.Request) {
	cfg, err := loadZhihuAuthConfig()
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: err.Error()})
		return
	}

	redirectURI := strings.TrimSpace(req.URL.Query().Get("redirect_uri"))
	if redirectURI == "" {
		redirectURI = cfg.RedirectURI
	}
	if redirectURI != cfg.RedirectURI {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "redirect_uri does not match configured Zhihu redirect URI"})
		return
	}

	state, err := randomURLToken(32)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "failed to generate OAuth state"})
		return
	}
	r.zhihuAuth.storeState(state, redirectURI)

	authorizeURL, err := r.zhihuAuth.authorizeURL(cfg.AppID, redirectURI, state)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, zhihuAuthorizeURLResponse{AuthorizeURL: authorizeURL, State: state})
}

func (r *Router) handleZhihuAuthCallback(w http.ResponseWriter, req *http.Request) {
	cfg, err := loadZhihuAuthConfig()
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: err.Error()})
		return
	}

	var body zhihuCallbackRequest
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON: " + err.Error()})
		return
	}
	body.Code = strings.TrimSpace(body.Code)
	body.State = strings.TrimSpace(body.State)
	body.RedirectURI = strings.TrimSpace(body.RedirectURI)
	if body.Code == "" {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "code is required"})
		return
	}
	if body.State == "" {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "state is required"})
		return
	}
	if body.RedirectURI == "" {
		body.RedirectURI = cfg.RedirectURI
	}
	if body.RedirectURI != cfg.RedirectURI {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "redirect_uri does not match configured Zhihu redirect URI"})
		return
	}
	if !r.zhihuAuth.consumeState(body.State, body.RedirectURI) {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid or expired OAuth state"})
		return
	}

	token, err := r.zhihuAuth.exchangeToken(req.Context(), cfg, body.Code, body.RedirectURI)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, ErrorResponse{Error: err.Error()})
		return
	}
	user, err := r.zhihuAuth.fetchUser(req.Context(), token.AccessToken)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, ErrorResponse{Error: err.Error()})
		return
	}

	expiresIn := token.ExpiresIn
	if expiresIn <= 0 {
		expiresIn = int64(zhihuDefaultExpiry.Seconds())
	}
	expiresAt := r.zhihuAuth.now().Add(time.Duration(expiresIn) * time.Second)
	sessionID, err := randomURLToken(32)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "failed to create Zhihu session"})
		return
	}
	r.zhihuAuth.storeSession(sessionID, zhihuSession{
		AccessToken: token.AccessToken,
		User:        user,
		ExpiresAt:   expiresAt,
	})
	http.SetCookie(w, r.zhihuAuth.cookie(req, sessionID, expiresAt, int(expiresIn)))

	writeJSON(w, http.StatusOK, zhihuCallbackResponse{
		Authenticated: true,
		ExpiresIn:     expiresIn,
		User:          user,
	})
}

func (r *Router) handleZhihuMe(w http.ResponseWriter, req *http.Request) {
	session, ok := r.zhihuAuth.sessionFromRequest(req)
	if !ok {
		writeJSON(w, http.StatusUnauthorized, ErrorResponse{Error: "not authenticated with Zhihu"})
		return
	}
	writeJSON(w, http.StatusOK, zhihuMeResponse{Authenticated: true, User: session.User})
}

func (r *Router) handleZhihuLogout(w http.ResponseWriter, req *http.Request) {
	if cookie, err := req.Cookie(zhihuSessionCookie); err == nil {
		r.zhihuAuth.deleteSession(cookie.Value)
	}
	http.SetCookie(w, r.zhihuAuth.expiredCookie(req))
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *zhihuAuthService) authorizeURL(appID, redirectURI, state string) (string, error) {
	u, err := url.Parse(s.authorizeEndpoint)
	if err != nil {
		return "", fmt.Errorf("invalid Zhihu authorize endpoint: %w", err)
	}
	q := u.Query()
	q.Set("redirect_uri", redirectURI)
	q.Set("app_id", appID)
	q.Set("response_type", "code")
	q.Set("state", state)
	u.RawQuery = q.Encode()
	return u.String(), nil
}

func (s *zhihuAuthService) storeState(state, redirectURI string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupLocked()
	s.states[state] = zhihuOAuthState{
		RedirectURI: redirectURI,
		ExpiresAt:   s.now().Add(zhihuOAuthStateTTL),
	}
}

func (s *zhihuAuthService) consumeState(state, redirectURI string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupLocked()
	stored, ok := s.states[state]
	if !ok || stored.RedirectURI != redirectURI || !stored.ExpiresAt.After(s.now()) {
		return false
	}
	delete(s.states, state)
	return true
}

func (s *zhihuAuthService) storeSession(id string, session zhihuSession) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupLocked()
	s.sessions[id] = session
}

func (s *zhihuAuthService) deleteSession(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.sessions, id)
}

func (s *zhihuAuthService) sessionFromRequest(req *http.Request) (zhihuSession, bool) {
	cookie, err := req.Cookie(zhihuSessionCookie)
	if err != nil || cookie.Value == "" {
		return zhihuSession{}, false
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	session, ok := s.sessions[cookie.Value]
	if !ok {
		return zhihuSession{}, false
	}
	if !session.ExpiresAt.After(s.now()) {
		delete(s.sessions, cookie.Value)
		return zhihuSession{}, false
	}
	return session, true
}

func (s *zhihuAuthService) cleanupLocked() {
	now := s.now()
	for state, stored := range s.states {
		if !stored.ExpiresAt.After(now) {
			delete(s.states, state)
		}
	}
	for id, session := range s.sessions {
		if !session.ExpiresAt.After(now) {
			delete(s.sessions, id)
		}
	}
}

func (s *zhihuAuthService) exchangeToken(ctx context.Context, cfg zhihuAuthConfig, code, redirectURI string) (zhihuTokenResponse, error) {
	form := url.Values{}
	form.Set("app_id", cfg.AppID)
	form.Set("app_key", cfg.AppKey)
	form.Set("grant_type", "authorization_code")
	form.Set("redirect_uri", redirectURI)
	form.Set("code", code)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.tokenEndpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return zhihuTokenResponse{}, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	var token zhihuTokenResponse
	if err := s.doJSON(req, &token); err != nil {
		return zhihuTokenResponse{}, fmt.Errorf("Zhihu token exchange failed: %w", err)
	}
	if token.AccessToken == "" {
		return zhihuTokenResponse{}, errors.New("Zhihu token exchange failed: missing access_token")
	}
	return token, nil
}

func (s *zhihuAuthService) fetchUser(ctx context.Context, accessToken string) (zhihuUser, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.userEndpoint, nil)
	if err != nil {
		return zhihuUser{}, err
	}
	req.Header.Set("Authorization", "Bearer "+accessToken)

	var user zhihuUser
	if err := s.doJSON(req, &user); err != nil {
		return zhihuUser{}, fmt.Errorf("Zhihu user fetch failed: %w", err)
	}
	if user.UID == 0 && user.Fullname == "" {
		return zhihuUser{}, errors.New("Zhihu user fetch failed: missing user profile")
	}
	return user, nil
}

func (s *zhihuAuthService) doJSON(req *http.Request, out any) error {
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	var raw json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	if err := zhihuAPIError(raw); err != nil {
		return err
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return err
	}
	return nil
}

func zhihuAPIError(raw json.RawMessage) error {
	var body struct {
		Code *int            `json:"code"`
		Data json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return err
	}
	if body.Code == nil || *body.Code == 0 || *body.Code == 20000 {
		return nil
	}
	message := decodeZhihuErrorData(body.Data)
	if message == "" || message == "null" {
		message = "request failed"
	}
	return fmt.Errorf("code %d: %s", *body.Code, message)
}

func decodeZhihuErrorData(raw json.RawMessage) string {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err == nil {
		for _, key := range []string{"message", "error", "error_description", "detail"} {
			if value, ok := obj[key].(string); ok {
				return value
			}
		}
	}
	return ""
}

func (s *zhihuAuthService) cookie(req *http.Request, value string, expires time.Time, maxAge int) *http.Cookie {
	return &http.Cookie{
		Name:     zhihuSessionCookie,
		Value:    value,
		Path:     "/",
		Expires:  expires,
		MaxAge:   maxAge,
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   requestIsHTTPS(req),
	}
}

func (s *zhihuAuthService) expiredCookie(req *http.Request) *http.Cookie {
	return &http.Cookie{
		Name:     zhihuSessionCookie,
		Value:    "",
		Path:     "/",
		Expires:  time.Unix(0, 0),
		MaxAge:   -1,
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   requestIsHTTPS(req),
	}
}

func requestIsHTTPS(req *http.Request) bool {
	if req.TLS != nil {
		return true
	}
	return strings.EqualFold(req.Header.Get("X-Forwarded-Proto"), "https")
}

func randomURLToken(size int) (string, error) {
	buf := make([]byte, size)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}
