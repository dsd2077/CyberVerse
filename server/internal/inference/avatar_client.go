package inference

import (
	"context"
	"io"
	"strconv"

	pb "github.com/cyberverse/server/internal/pb"
	"google.golang.org/grpc/metadata"
)

// SetAvatar sends an image to the inference server to configure the avatar.
func (c *Client) SetAvatar(ctx context.Context, sessionID string, imageData []byte, format string) error {
	_, err := c.avatar.SetAvatar(ctx, &pb.SetAvatarRequest{
		SessionId:   sessionID,
		ImageData:   imageData,
		ImageFormat: format,
		UseFaceCrop: false,
	})
	return err
}

// GenerateAvatarStream opens a bidirectional stream: sends audio chunks,
// receives video chunks. Returns output channel and error channel.
func (c *Client) GenerateAvatarStream(ctx context.Context, audioCh <-chan *pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	videoCh := make(chan *pb.VideoChunk, 4)
	errCh := make(chan error, 1)

	go func() {
		// Close videoCh before errCh so consumers can drain buffered VideoChunk
		// before seeing errCh close (avoids racing on a zero error receive).
		defer close(errCh)
		defer close(videoCh)

		if trace, ok := TraceContextFromContext(ctx); ok {
			ctx = metadata.AppendToOutgoingContext(
				ctx,
				"x-cyberverse-session-id", trace.SessionID,
				"x-cyberverse-question-id", trace.QuestionID,
				"x-cyberverse-reply-id", trace.ReplyID,
				"x-cyberverse-turn-seq", strconv.FormatUint(trace.TurnSeq, 10),
			)
		}

		stream, err := c.avatar.GenerateStream(ctx)
		if err != nil {
			errCh <- err
			return
		}

		sendDone := make(chan error, 1)
		go func() {
			defer func() { _ = stream.CloseSend() }()
			for {
				select {
				case <-ctx.Done():
					sendDone <- ctx.Err()
					return
				case chunk, ok := <-audioCh:
					if !ok {
						sendDone <- nil
						return
					}
					err := stream.Send(chunk)
					if err != nil {
						sendDone <- err
						return
					}
				}
			}
		}()

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
			case videoCh <- chunk:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}

		if err := <-sendDone; err != nil {
			errCh <- err
		}
	}()

	return videoCh, errCh
}

// GenerateAvatar sends pre-collected audio chunks to the avatar service
// and returns a channel of video chunks as they are generated.
func (c *Client) GenerateAvatar(ctx context.Context, audioChunks []*pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	videoCh := make(chan *pb.VideoChunk, 4)
	errCh := make(chan error, 1)

	go func() {
		defer close(errCh)
		defer close(videoCh)

		stream, err := c.avatar.GenerateStream(ctx)
		if err != nil {
			errCh <- err
			return
		}

		// Send all audio chunks then close the send side.
		for _, chunk := range audioChunks {
			if ctx.Err() != nil {
				errCh <- ctx.Err()
				return
			}
			if err := stream.Send(chunk); err != nil {
				errCh <- err
				return
			}
		}
		if err := stream.CloseSend(); err != nil {
			errCh <- err
			return
		}

		// Stream back video chunks as they are generated.
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
			case videoCh <- chunk:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}
	}()

	return videoCh, errCh
}
