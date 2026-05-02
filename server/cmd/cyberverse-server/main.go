package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"path/filepath"

	"github.com/cyberverse/server/internal/api"
	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/config"
	"github.com/cyberverse/server/internal/direct"
	"github.com/cyberverse/server/internal/inference"
	"github.com/cyberverse/server/internal/livekit"
	"github.com/cyberverse/server/internal/orchestrator"
	"github.com/cyberverse/server/internal/recording"
	"github.com/cyberverse/server/internal/ws"
)

func main() {
	configPath := flag.String("config", "../../cyberverse_config.yaml", "path to config file")
	flag.Parse()

	// Load .env before config so ${VAR} placeholders in YAML expand correctly.
	envPath := filepath.Join(filepath.Dir(*configPath), ".env")
	if err := config.LoadDotenv(envPath); err != nil {
		log.Printf("Warning: failed to load .env: %v", err)
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Create session manager
	sessionMgr := orchestrator.NewSessionManager(cfg.Session.MaxConcurrent)

	// Create WebSocket hub
	wsHub := ws.NewHub()

	// Create inference gRPC client
	inferenceClient, err := inference.NewClient(cfg.Inference.Addr)
	if err != nil {
		log.Printf("Warning: failed to connect to inference server at %s: %v", cfg.Inference.Addr, err)
		log.Printf("Server will start but inference features will be unavailable")
	}

	// Create LiveKit room manager
	roomMgr := livekit.NewRoomManager(cfg.LiveKit.URL, cfg.LiveKit.APIKey, cfg.LiveKit.APISecret)

	// Create character store (directory-based, one dir per character)
	dataDir := filepath.Join(filepath.Dir(*configPath), "data")
	os.MkdirAll(dataDir, 0755)
	charStore, err := character.NewStore(filepath.Join(dataDir, "characters"))
	if err != nil {
		log.Fatalf("Failed to init character store: %v", err)
	}

	// Create orchestrator (needs charStore for recording paths)
	recorder := recording.NewVideoRecorder(cfg.Recording)
	orch := orchestrator.New(inferenceClient, wsHub, sessionMgr, recorder, charStore, cfg.Pipeline)

	// Embedded TURN-over-TCP server for NAT traversal (AutoDL, SSH tunnel, etc.)
	var turnServer *direct.TURNServer
	if cfg.Pipeline.TURNEnabled && cfg.Pipeline.TURNPort > 0 {
		publicIP := cfg.Pipeline.ICEPublicIP
		// Resolve hostname to IP if needed
		if publicIP != "" && net.ParseIP(publicIP) == nil {
			addrs, err := net.LookupHost(publicIP)
			if err != nil || len(addrs) == 0 {
				log.Fatalf("Cannot resolve ice_public_ip %q: %v", publicIP, err)
			}
			publicIP = addrs[0]
			log.Printf("Resolved ice_public_ip %q -> %s", cfg.Pipeline.ICEPublicIP, publicIP)
		}
		if publicIP == "" {
			publicIP = "127.0.0.1"
		}
		ts, err := direct.NewTURNServer(
			cfg.Pipeline.TURNPort, publicIP,
			cfg.Pipeline.TURNRealm,
			cfg.Pipeline.TURNUsername,
			cfg.Pipeline.TURNPassword,
		)
		if err != nil {
			log.Fatalf("TURN server setup failed: %v", err)
		}
		turnServer = ts
		orch.SetTURNServer(ts)
		log.Printf("TURN server enabled on TCP port %d (relay IP: %s)", cfg.Pipeline.TURNPort, publicIP)
	}

	// WebRTC API with interceptors (NACK, TWCC, GCC pacer) for direct streaming mode
	if cfg.Pipeline.StreamingMode == "direct" {
		api, estimatorCh, err := direct.NewWebRTCAPI(direct.WebRTCAPIConfig{
			InitialBitrate: 6_000_000,  // 6 Mbps — comfortably above VP8 actual ~4 Mbps to avoid pacer starvation at turn start
			MinBitrate:     5_000_000,  // 5 Mbps — floor above VP8 actual ~4 Mbps to prevent AV desync from pacer queuing
			MaxBitrate:     10_000_000, // 10 Mbps
		})
		if err != nil {
			log.Fatalf("WebRTC API setup failed: %v", err)
		}
		orch.SetWebRTCAPI(api, estimatorCh)
		log.Println("WebRTC API initialized with interceptors (NACK, TWCC, GCC)")
	}

	// Register session end callback to persist conversation history
	sessionMgr.OnSessionEnd = func(s *orchestrator.Session) {
		history := s.HistorySnapshot()
		log.Printf("OnSessionEnd: session=%s character=%s historyLen=%d", s.ID, s.CharacterID, len(history))
		if s.CharacterID == "" || len(history) == 0 {
			log.Printf("OnSessionEnd: skipping save — characterID=%q historyLen=%d", s.CharacterID, len(history))
			return
		}
		messages := make([]map[string]any, len(history))
		for i, m := range history {
			ts := m.Timestamp
			if ts.IsZero() {
				ts = s.CreatedAt
			}
			messages[i] = map[string]any{
				"role":      m.Role,
				"content":   m.Content,
				"timestamp": ts.UTC().Format(time.RFC3339Nano),
			}
			if m.TurnSeq > 0 {
				messages[i]["turn_seq"] = m.TurnSeq
			}
		}
		if err := charStore.SaveConversation(s.CharacterID, s.ID, s.CreatedAt, s.LastActiveAt, messages); err != nil {
			log.Printf("Failed to save conversation for session %s: %v", s.ID, err)
		} else {
			log.Printf("Conversation saved for session %s (character %s)", s.ID, s.CharacterID)
		}
	}

	// Create router with all dependencies
	router := api.NewRouter(sessionMgr, orch, wsHub, roomMgr, cfg, charStore, envPath, *configPath)

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.HTTPPort)
	srv := &http.Server{
		Addr:    addr,
		Handler: router.Handler(),
	}

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("Shutting down server...")

		// Teardown all orchestrator sessions
		orch.TeardownAll()

		// Close inference client
		if inferenceClient != nil {
			inferenceClient.Close()
		}

		// Close TURN server
		if turnServer != nil {
			turnServer.Close()
		}

		// Stop session manager cleanup
		sessionMgr.Stop()

		srv.Close()
	}()

	log.Printf("CyberVerse Server starting on %s", addr)
	log.Printf("Inference server: %s", cfg.Inference.Addr)
	log.Printf("LiveKit URL: %s", cfg.LiveKit.URL)
	log.Printf("Streaming mode: %s", cfg.Pipeline.StreamingMode)
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
}
