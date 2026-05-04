package ws

import (
	"encoding/json"
	"log"
	"time"

	"github.com/gorilla/websocket"
)

const (
	writeWait             = 10 * time.Second
	pongWait              = 60 * time.Second
	pingPeriod            = (pongWait * 9) / 10
	defaultMaxMessageSize = 1024 * 1024
)

// Client represents a single WebSocket connection.
type Client struct {
	SessionID      string
	Conn           *websocket.Conn
	Send           chan []byte
	MaxMessageSize int64
	hub            *Hub
}

// WSMessage represents a message received from a WebSocket client.
type WSMessage struct {
	Type        string  `json:"type"`
	Text        string  `json:"text,omitempty"`
	SDP         string  `json:"sdp,omitempty"`
	Candidate   string  `json:"candidate,omitempty"`
	SDPMid      string  `json:"sdp_mid,omitempty"`
	SDPMLine    *uint16 `json:"sdp_mline_index,omitempty"`
	Source      string  `json:"source,omitempty"`
	Mime        string  `json:"mime,omitempty"`
	Data        string  `json:"data,omitempty"`
	Width       int32   `json:"width,omitempty"`
	Height      int32   `json:"height,omitempty"`
	TimestampMS int64   `json:"timestamp_ms,omitempty"`
	FrameSeq    int64   `json:"frame_seq,omitempty"`
}

// ReadPump reads messages from the WebSocket and dispatches them via onMessage.
// Must be run as a goroutine. When it returns, the client is unregistered.
func (c *Client) ReadPump(
	onMessage func(sessionID string, msg WSMessage),
	onActivity func(sessionID string),
) {
	defer func() {
		c.hub.Unregister(c)
		c.Conn.Close()
	}()

	maxMessageSize := c.MaxMessageSize
	if maxMessageSize <= 0 {
		maxMessageSize = defaultMaxMessageSize
	}
	c.Conn.SetReadLimit(maxMessageSize)
	c.Conn.SetReadDeadline(time.Now().Add(pongWait))
	c.Conn.SetPongHandler(func(string) error {
		c.Conn.SetReadDeadline(time.Now().Add(pongWait))
		if onActivity != nil {
			onActivity(c.SessionID)
		}
		return nil
	})

	for {
		_, message, err := c.Conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseNormalClosure) {
				log.Printf("ws: read error for session %s: %v", c.SessionID, err)
			}
			return
		}
		if onActivity != nil {
			onActivity(c.SessionID)
		}

		var msg WSMessage
		if err := json.Unmarshal(message, &msg); err != nil {
			log.Printf("ws: invalid JSON from session %s: %v", c.SessionID, err)
			continue
		}

		onMessage(c.SessionID, msg)
	}
}

// WritePump writes messages from the Send channel to the WebSocket.
// Must be run as a goroutine.
func (c *Client) WritePump() {
	ticker := time.NewTicker(pingPeriod)
	defer func() {
		ticker.Stop()
		c.Conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.Send:
			c.Conn.SetWriteDeadline(time.Now().Add(writeWait))
			if !ok {
				c.Conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			if err := c.Conn.WriteMessage(websocket.TextMessage, message); err != nil {
				return
			}

		case <-ticker.C:
			c.Conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.Conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}
