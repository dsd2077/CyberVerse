package ws

import (
	"log"
	"net/http"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true // CORS handled at router level
	},
}

// HandleWebSocket returns an HTTP handler that upgrades connections to WebSocket
// and dispatches incoming messages via onMessage.
func HandleWebSocket(
	hub *Hub,
	sessionID string,
	onMessage func(string, WSMessage),
	onActivity func(string),
) http.HandlerFunc {
	return HandleWebSocketWithReadLimit(hub, sessionID, 0, onMessage, onActivity)
}

func HandleWebSocketWithReadLimit(
	hub *Hub,
	sessionID string,
	maxMessageSize int64,
	onMessage func(string, WSMessage),
	onActivity func(string),
) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("ws: upgrade failed for session %s: %v", sessionID, err)
			return
		}

		client := &Client{
			SessionID:      sessionID,
			Conn:           conn,
			Send:           make(chan []byte, 64),
			MaxMessageSize: maxMessageSize,
			hub:            hub,
		}

		hub.Register(client)
		if onActivity != nil {
			onActivity(sessionID)
		}

		go client.WritePump()
		go client.ReadPump(onMessage, onActivity)
	}
}
