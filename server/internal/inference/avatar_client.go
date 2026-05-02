package inference

import (
	"context"
	"fmt"
	"io"
	"log"
	"strconv"
	"strings"

	pb "github.com/cyberverse/server/internal/pb"
	"google.golang.org/grpc/metadata"
)

const flashheadGenerationStartedHeader = "x-cyberverse-trace-flashhead-generation-started-since-user-final-ms"

func safeTraceValue(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "-"
	}
	return value
}

func logVoiceTraceWithSinceValue(event string, trace TraceContext, since string, fields ...string) {
	if strings.TrimSpace(since) == "" {
		since = "-"
	}
	parts := []string{
		fmt.Sprintf("voice_trace event=%-30s", event),
		"sid=" + safeTraceValue(trace.SessionID),
		"turn=" + strconv.FormatUint(trace.TurnSeq, 10),
		"reply=" + safeTraceValue(trace.ReplyID),
		"qid=" + safeTraceValue(trace.QuestionID),
		"since_user_final_ms=" + since,
	}
	parts = append(parts, fields...)
	log.Print(strings.Join(parts, " "))
}

// SetAvatar sends an image to the inference server to configure the avatar.
func (c *Client) SetAvatar(ctx context.Context, sessionID string, imageData []byte, format string) error {
	resp, err := c.avatar.SetAvatar(ctx, &pb.SetAvatarRequest{
		SessionId:   sessionID,
		ImageData:   imageData,
		ImageFormat: format,
		UseFaceCrop: false,
	})
	if err != nil {
		return err
	}
	if resp != nil && !resp.GetSuccess() {
		return fmt.Errorf("set avatar rejected by inference server: %s", resp.GetMessage())
	}
	return nil
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

		var trace TraceContext
		var traceEnabled bool
		if attachedTrace, ok := TraceContextFromContext(ctx); ok {
			traceEnabled = true
			trace = attachedTrace
			mdPairs := []string{
				"x-cyberverse-session-id", trace.SessionID,
				"x-cyberverse-question-id", trace.QuestionID,
				"x-cyberverse-reply-id", trace.ReplyID,
				"x-cyberverse-turn-seq", strconv.FormatUint(trace.TurnSeq, 10),
			}
			if !trace.UserFinalAt.IsZero() {
				mdPairs = append(
					mdPairs,
					"x-cyberverse-user-final-unix-ms", strconv.FormatInt(trace.UserFinalAt.UnixMilli(), 10),
				)
			}
			ctx = metadata.AppendToOutgoingContext(ctx, mdPairs...)
		}

		stream, err := c.avatar.GenerateStream(ctx)
		if err != nil {
			errCh <- err
			return
		}

		if traceEnabled {
			go func(trace TraceContext) {
				header, headerErr := stream.Header()
				if headerErr != nil {
					return
				}
				values := header.Get(flashheadGenerationStartedHeader)
				if len(values) == 0 {
					return
				}
				logVoiceTraceWithSinceValue(
					"flashhead_generation_started",
					trace,
					strings.TrimSpace(values[0]),
				)
			}(trace)
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
