package ws

import (
	"encoding/json"
	"log"
	"sync"
)

// Hub manages WebSocket clients grouped by session ID.
type Hub struct {
	clients map[string]map[*Client]struct{} // sessionID → set of clients
	mu      sync.RWMutex
}

// NewHub creates a new WebSocket hub.
func NewHub() *Hub {
	return &Hub{
		clients: make(map[string]map[*Client]struct{}),
	}
}

// Register adds a client to the hub under its session ID.
func (h *Hub) Register(client *Client) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if _, ok := h.clients[client.SessionID]; !ok {
		h.clients[client.SessionID] = make(map[*Client]struct{})
	}
	h.clients[client.SessionID][client] = struct{}{}
}

// Unregister removes a client from the hub and closes its send channel.
// It returns the remaining client count for the session and whether the client
// was removed by this call.
func (h *Hub) Unregister(client *Client) (int, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if clients, ok := h.clients[client.SessionID]; ok {
		if _, exists := clients[client]; exists {
			delete(clients, client)
			close(client.Send)
			remaining := len(clients)
			if len(clients) == 0 {
				delete(h.clients, client.SessionID)
			}
			return remaining, true
		}
	}
	return 0, false
}

// Broadcast sends a message to all clients in a session.
func (h *Hub) Broadcast(sessionID string, msg []byte) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	clients, ok := h.clients[sessionID]
	if !ok {
		return
	}
	for client := range clients {
		select {
		case client.Send <- msg:
		default:
			log.Printf("ws: dropping message for slow client in session %s", sessionID)
		}
	}
}

// BroadcastJSON marshals v as JSON and broadcasts to the session.
func (h *Hub) BroadcastJSON(sessionID string, v any) {
	data, err := json.Marshal(v)
	if err != nil {
		log.Printf("ws: failed to marshal broadcast message: %v", err)
		return
	}
	h.Broadcast(sessionID, data)
}

// CloseSession disconnects all clients for a session.
func (h *Hub) CloseSession(sessionID string) {
	h.mu.Lock()
	clients, ok := h.clients[sessionID]
	if ok {
		delete(h.clients, sessionID)
	}
	h.mu.Unlock()

	if ok {
		for client := range clients {
			close(client.Send)
			client.Conn.Close()
		}
	}
}
