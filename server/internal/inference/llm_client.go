package inference

import (
	"context"
	"io"

	pb "github.com/cyberverse/server/internal/pb"
)

// GenerateLLMStream sends a chat request and returns a channel of streaming LLM chunks.
func (c *Client) GenerateLLMStream(ctx context.Context, sessionID string, messages []ChatMessage, config LLMConfig) (<-chan *pb.LLMChunk, <-chan error) {
	chunkCh := make(chan *pb.LLMChunk, 16)
	errCh := make(chan error, 1)

	go func() {
		defer close(chunkCh)
		defer close(errCh)

		pbMessages := make([]*pb.ChatMessage, len(messages))
		for i, m := range messages {
			pbMessages[i] = &pb.ChatMessage{
				Role:    m.Role,
				Content: m.Content,
			}
		}

		stream, err := c.llm.GenerateStream(ctx, &pb.LLMRequest{
			SessionId: sessionID,
			Messages:  pbMessages,
			Config: &pb.LLMConfig{
				Model:       config.Model,
				Temperature: config.Temperature,
				MaxTokens:   config.MaxTokens,
				Provider:    config.Provider,
			},
		})
		if err != nil {
			errCh <- err
			return
		}

		for {
			chunk, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				errCh <- err
				return
			}
			select {
			case chunkCh <- chunk:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}
	}()

	return chunkCh, errCh
}
