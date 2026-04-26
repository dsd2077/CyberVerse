package inference

import (
	"context"
	"errors"
	"io"

	pb "github.com/cyberverse/server/internal/pb"
	"google.golang.org/grpc/status"
)

func voiceLLMConfigPB(config VoiceLLMSessionConfig) *pb.VoiceLLMConfig {
	return &pb.VoiceLLMConfig{
		SessionId:      config.SessionID,
		SystemPrompt:   config.SystemPrompt,
		Voice:          config.Voice,
		BotName:        config.BotName,
		SpeakingStyle:  config.SpeakingStyle,
		WelcomeMessage: config.WelcomeMessage,
	}
}

func unwrapGRPCError(err error) error {
	if err == nil {
		return nil
	}
	if st, ok := status.FromError(err); ok {
		return errors.New(st.Message())
	}
	return err
}

func (c *Client) CheckVoice(ctx context.Context, config VoiceLLMSessionConfig) (string, error) {
	resp, err := c.voiceLLM.CheckVoice(ctx, &pb.CheckVoiceRequest{
		Config: voiceLLMConfigPB(config),
	})
	if err != nil {
		return "", unwrapGRPCError(err)
	}
	return resp.GetProviderError(), nil
}

// ConverseStream opens a bidirectional stream for a VoiceLLM conversation.
// Sends a config message first, then streams user audio/text input events.
func (c *Client) ConverseStream(ctx context.Context, inputCh <-chan VoiceLLMInputEvent, config VoiceLLMSessionConfig) (<-chan *pb.VoiceLLMOutput, <-chan error) {
	outputCh := make(chan *pb.VoiceLLMOutput, 8)
	errCh := make(chan error, 1)

	go func() {
		defer close(outputCh)
		defer close(errCh)

		stream, err := c.voiceLLM.Converse(ctx)
		if err != nil {
			errCh <- err
			return
		}

		// Send config message first
		err = stream.Send(&pb.VoiceLLMInput{
			Input: &pb.VoiceLLMInput_Config{
				Config: voiceLLMConfigPB(config),
			},
		})
		if err != nil {
			errCh <- err
			return
		}

		// Sender goroutine: stream user audio/text events.
		sendDone := make(chan error, 1)
		go func() {
			defer func() { _ = stream.CloseSend() }()
			for {
				select {
				case <-ctx.Done():
					sendDone <- ctx.Err()
					return
				case input, ok := <-inputCh:
					if !ok {
						sendDone <- nil
						return
					}
					var req *pb.VoiceLLMInput
					switch {
					case len(input.Audio) > 0:
						req = &pb.VoiceLLMInput{
							Input: &pb.VoiceLLMInput_Audio{
								Audio: &pb.AudioChunk{
									Data:       input.Audio,
									SampleRate: 16000,
									Channels:   1,
									Format:     "float32",
								},
							},
						}
					case input.Text != "":
						req = &pb.VoiceLLMInput{
							Input: &pb.VoiceLLMInput_Text{
								Text: input.Text,
							},
						}
					default:
						continue
					}
					err := stream.Send(req)
					if err != nil {
						sendDone <- err
						return
					}
				}
			}
		}()

		// Receiver loop
		for {
			output, err := stream.Recv()
			if err == io.EOF {
				break
			}
			if err != nil {
				errCh <- err
				return
			}
			select {
			case outputCh <- output:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}

		if err := <-sendDone; err != nil {
			errCh <- err
		}
	}()

	return outputCh, errCh
}

// Interrupt sends an interrupt request to stop the current VoiceLLM response.
func (c *Client) Interrupt(ctx context.Context, sessionID string) error {
	_, err := c.voiceLLM.Interrupt(ctx, &pb.InterruptRequest{
		SessionId: sessionID,
	})
	return err
}
