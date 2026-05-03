package inference

import (
	"context"
	"io"

	pb "github.com/cyberverse/server/internal/pb"
)

// SynthesizeSpeechStream opens a bidirectional stream: sends text chunks,
// receives audio chunks.
func (c *Client) SynthesizeSpeechStream(ctx context.Context, textCh <-chan string, config TTSConfig) (<-chan *pb.AudioChunk, <-chan error) {
	audioCh := make(chan *pb.AudioChunk, 8)
	errCh := make(chan error, 1)

	go func() {
		defer close(audioCh)
		defer close(errCh)

		stream, err := c.tts.SynthesizeStream(ctx)
		if err != nil {
			errCh <- err
			return
		}

		// Sender goroutine
		sendDone := make(chan error, 1)
		go func() {
			defer func() { _ = stream.CloseSend() }()
			if err := stream.Send(&pb.TextChunk{
				Config: &pb.TTSConfig{
					Provider:      config.Provider,
					Voice:         config.Voice,
					SpeakingStyle: config.SpeakingStyle,
					Language:      config.Language,
					SessionId:     config.SessionID,
				},
			}); err != nil {
				sendDone <- err
				return
			}
			for {
				select {
				case <-ctx.Done():
					sendDone <- ctx.Err()
					return
				case text, ok := <-textCh:
					if !ok {
						sendDone <- nil
						return
					}
					err := stream.Send(&pb.TextChunk{
						Text:    text,
						IsFinal: false,
					})
					if err != nil {
						sendDone <- err
						return
					}
				}
			}
		}()

		// Receiver loop
		for {
			chunk, err := stream.Recv()
			if err == io.EOF {
				break
			}
			if err != nil {
				errCh <- err
				return
			}
			select {
			case audioCh <- chunk:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}

		if err := <-sendDone; err != nil {
			errCh <- err
		}
	}()

	return audioCh, errCh
}
