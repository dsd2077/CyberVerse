package inference

import (
	"context"
	"io"

	pb "github.com/cyberverse/server/internal/pb"
)

// TranscribeStream opens a bidirectional stream: sends audio chunks,
// receives transcript events.
func (c *Client) TranscribeStream(ctx context.Context, audioCh <-chan []byte, config ASRConfig) (<-chan *pb.TranscriptEvent, <-chan error) {
	transcriptCh := make(chan *pb.TranscriptEvent, 8)
	errCh := make(chan error, 1)

	go func() {
		defer close(transcriptCh)
		defer close(errCh)

		stream, err := c.asr.TranscribeStream(ctx)
		if err != nil {
			errCh <- err
			return
		}

		// Sender goroutine
		sendDone := make(chan error, 1)
		go func() {
			defer func() { _ = stream.CloseSend() }()
			if err := stream.Send(&pb.ASRInput{
				Input: &pb.ASRInput_Config{
					Config: &pb.ASRConfig{
						Provider:  config.Provider,
						Language:  config.Language,
						SessionId: config.SessionID,
					},
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
				case data, ok := <-audioCh:
					if !ok {
						sendDone <- nil
						return
					}
					err := stream.Send(&pb.ASRInput{
						Input: &pb.ASRInput_Audio{
							Audio: &pb.AudioChunk{
								Data:       data,
								SampleRate: 16000,
								Channels:   1,
								Format:     "pcm_s16le",
							},
						},
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
			event, err := stream.Recv()
			if err == io.EOF {
				break
			}
			if err != nil {
				errCh <- err
				return
			}
			select {
			case transcriptCh <- event:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}

		if err := <-sendDone; err != nil {
			errCh <- err
		}
	}()

	return transcriptCh, errCh
}
