package api

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/google/uuid"

	"github.com/cyberverse/server/internal/livekit"
	"github.com/cyberverse/server/internal/orchestrator"
	"github.com/cyberverse/server/internal/ws"
)

type CreateSessionRequest struct {
	Mode        string `json:"mode"`         // "voice_llm" or "standard"
	CharacterID string `json:"character_id"` // which character this session is for
}

type CreateSessionResponse struct {
	SessionID     string   `json:"session_id"`
	StreamingMode string   `json:"streaming_mode"`
	LiveKitURL    string   `json:"livekit_url,omitempty"`
	Token         string   `json:"livekit_token,omitempty"`
	IdleVideoURL  string   `json:"idle_video_url,omitempty"`
	IdleVideoURLs []string `json:"idle_video_urls,omitempty"`
	Warnings      []string `json:"warnings,omitempty"`
}

type SendMessageRequest struct {
	Text string `json:"text"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}

func (r *Router) handleHealth(w http.ResponseWriter, req *http.Request) {
	inferenceErr := r.inferenceHealthError(req.Context())
	connected := inferenceErr == nil
	status := "ok"
	errorMessage := ""
	if !connected {
		status = "error"
		errorMessage = inferenceErr.Error()
	}

	payload := map[string]any{
		"status":              status,
		"sessions":            r.sessionMgr.Count(),
		"inference_connected": connected,
		"error":               errorMessage,
	}
	if body, err := json.Marshal(payload); err == nil {
		log.Printf("[health] GET /api/v1/health response: %s", body)
	}

	writeJSON(w, http.StatusOK, payload)
}

func (r *Router) handleCreateSession(w http.ResponseWriter, req *http.Request) {
	var body CreateSessionRequest
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON: " + err.Error()})
		return
	}

	mode := orchestrator.ModeVoiceLLM
	if body.Mode == "standard" {
		mode = orchestrator.ModeStandard
	}

	if r.orch != nil && r.charStore != nil && body.CharacterID != "" {
		if _, err := r.activeAvatarModel(req.Context()); err != nil {
			writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: err.Error()})
			return
		}
	}

	sessionID := uuid.New().String()
	session, err := r.sessionMgr.Create(sessionID, mode, body.CharacterID)
	if err != nil {
		status := http.StatusInternalServerError
		if err == orchestrator.ErrMaxSessions {
			status = http.StatusServiceUnavailable
		}
		writeJSON(w, status, ErrorResponse{Error: err.Error()})
		return
	}
	if r.orch != nil {
		if err := r.orch.HydrateVoiceDialogContext(session); err != nil {
			log.Printf("Failed to load dialog context for session %s character %s: %v", sessionID, body.CharacterID, err)
		}
	}

	// If character uses random image mode, pick a random image
	if body.CharacterID != "" {
		if ch, chErr := r.charStore.Get(body.CharacterID); chErr == nil {
			if ch.ImageMode == "random" && len(ch.Images) > 1 {
				if rErr := r.charStore.RandomizeImage(body.CharacterID); rErr != nil {
					log.Printf("Failed to randomize image for character %s: %v", body.CharacterID, rErr)
				}
			}
		}
	}

	resp := CreateSessionResponse{
		SessionID: sessionID,
	}

	if r.orch != nil && body.CharacterID != "" {
		target := r.currentIdleVideoTarget(req.Context())
		// Return any already-cached idle video URLs immediately; generation happens in background.
		if char, err := r.charStore.Get(body.CharacterID); err == nil {
			resp.IdleVideoURLs = r.idleVideoURLs(char.ID, char.ActiveImage, target)
			if len(resp.IdleVideoURLs) > 0 {
				resp.IdleVideoURL = resp.IdleVideoURLs[0]
			}
		}
		// Trigger background generation only if the current-resolution idle video does not exist yet.
		// Once ready, push the URLs to the frontend via WebSocket so the idle
		// videos can start playing without a page reload.
		if len(resp.IdleVideoURLs) == 0 {
			char, _ := r.charStore.Get(body.CharacterID)
			activeImage := ""
			if char != nil {
				activeImage = char.ActiveImage
			}
			go func(charID, sessID, img string) {
				bgCtx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
				defer cancel()
				if _, err := r.orch.EnsureIdleVideo(bgCtx, charID); err != nil {
					if warning, ok := orchestrator.AvatarImageTooLargeWarning(err); ok {
						log.Printf("Idle video background generation failed for character %s: %s details=%v", charID, warning, err)
						if r.wsHub != nil {
							r.wsHub.BroadcastJSON(sessID, map[string]any{
								"type":    "avatar_warning",
								"message": warning,
							})
						}
					} else {
						log.Printf("Idle video background generation failed for character %s: %v", charID, err)
					}
					return
				}
				urls := r.idleVideoURLs(charID, img, r.currentIdleVideoTarget(bgCtx))
				if len(urls) > 0 {
					r.wsHub.BroadcastJSON(sessID, map[string]any{
						"type": "idle_video_ready",
						"url":  urls[0],
						"urls": urls,
					})
				}
			}(body.CharacterID, sessionID, activeImage)
		}
	}

	// Set up media peer (DirectPeer or LiveKit Bot) if orchestrator is available
	if r.orch != nil {
		streamingMode := r.orch.StreamingMode()
		resp.StreamingMode = streamingMode

		// Generate LiveKit token only in livekit mode
		if streamingMode == "livekit" && r.roomMgr != nil && r.cfg != nil {
			roomName := livekit.RoomName(sessionID)
			token, err := livekit.GenerateToken(
				r.cfg.LiveKit.APIKey,
				r.cfg.LiveKit.APISecret,
				roomName,
				"user-"+sessionID,
				true,
			)
			if err != nil {
				log.Printf("Failed to generate LiveKit token: %v", err)
			} else {
				resp.LiveKitURL = r.cfg.LiveKit.URL
				resp.Token = token
			}
		}

		// Setup session with media peer.
		// Important: don't tie this lifecycle to req.Context(), because the browser client
		// may abort/cancel the HTTP request (navigation, rapid reconnect, etc.).
		setupCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		peer, warnings, err := r.orch.SetupSession(setupCtx, session, r.roomMgr)
		resp.Warnings = append(resp.Warnings, warnings...)
		if err != nil {
			log.Printf("Failed to setup session %s: %v", sessionID, err)
		} else if mode == orchestrator.ModeVoiceLLM {
			// Start listening for user audio in VoiceLLM mode
			go func() {
				if err := r.orch.HandleAudioStream(context.Background(), sessionID, peer.SubscribeUserAudio()); err != nil {
					log.Printf("Failed to start audio stream for session %s: %v", sessionID, err)
				}
			}()
		}
	}

	writeJSON(w, http.StatusCreated, resp)
}

func (r *Router) handleDeleteSession(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	if _, err := r.sessionMgr.Get(id); err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}

	// Teardown orchestrator resources
	if r.orch != nil {
		if err := r.orch.TeardownSession(id); err != nil {
			log.Printf("Failed to teardown session %s: %v", id, err)
		}
	}

	r.sessionMgr.Delete(id)
	w.WriteHeader(http.StatusNoContent)
}

func (r *Router) handleSendMessage(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	if _, err := r.sessionMgr.Get(id); err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}

	var body SendMessageRequest
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON: " + err.Error()})
		return
	}
	if body.Text == "" {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "text is required"})
		return
	}

	// Note: HandleTextInput already calls session.AddMessage for user role,
	// so we do NOT add it here to avoid duplicate messages.

	// Trigger the standard pipeline via orchestrator
	if r.orch != nil {
		if err := r.orch.HandleTextInput(context.Background(), id, body.Text); err != nil {
			log.Printf("Failed to handle text input for session %s: %v", id, err)
			writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "failed to process message"})
			return
		}
	}

	writeJSON(w, http.StatusAccepted, map[string]string{"status": "queued"})
}

func (r *Router) handleListSessions(w http.ResponseWriter, req *http.Request) {
	sessions := r.sessionMgr.List()
	type sessionInfo struct {
		ID    string `json:"id"`
		State string `json:"state"`
	}
	result := make([]sessionInfo, len(sessions))
	for i, s := range sessions {
		result[i] = sessionInfo{ID: s.ID, State: s.GetState().String()}
	}
	writeJSON(w, http.StatusOK, result)
}

func (r *Router) handleWebSocket(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	if _, err := r.sessionMgr.Get(id); err != nil {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}

	handler := ws.HandleWebSocket(
		r.wsHub,
		id,
		func(sessionID string, msg ws.WSMessage) {
			switch msg.Type {
			case "text_input":
				if r.orch != nil && msg.Text != "" {
					go func() {
						// Detach from request context to avoid cancelling an in-flight text turn.
						if err := r.orch.HandleTextInput(context.Background(), sessionID, msg.Text); err != nil {
							log.Printf("Failed to handle WS text input for session %s: %v", sessionID, err)
						}
					}()
				}
			case "interrupt":
				if r.orch != nil {
					if err := r.orch.Interrupt(sessionID); err != nil {
						log.Printf("Failed to interrupt session %s: %v", sessionID, err)
					}
				}
			case "webrtc_ready", "webrtc_answer", "ice_candidate":
				if r.orch != nil {
					r.orch.HandleSignaling(sessionID, msg)
				}
			}
		},
		func(sessionID string) {
			_ = r.sessionMgr.Touch(sessionID)
		},
	)
	handler(w, req)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("writeJSON encode error: %v", err)
	}
}

func (r *Router) inferenceHealthError(ctx context.Context) error {
	if r.orch == nil {
		return errInferenceUnavailable
	}
	return r.orch.HealthCheck(ctx)
}
