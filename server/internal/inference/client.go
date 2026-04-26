package inference

import (
	"context"
	"fmt"
	"time"

	pb "github.com/cyberverse/server/internal/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

// Client manages the gRPC connection to the Python inference server
// and provides typed access to all service clients.
type Client struct {
	conn     *grpc.ClientConn
	avatar   pb.AvatarServiceClient
	llm      pb.LLMServiceClient
	tts      pb.TTSServiceClient
	asr      pb.ASRServiceClient
	voiceLLM pb.VoiceLLMServiceClient
}

// Compile-time check that Client implements InferenceService.
var _ InferenceService = (*Client)(nil)

// NewClient creates a new gRPC client connected to the inference server.
func NewClient(addr string) (*Client, error) {
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                60 * time.Second,
			Timeout:             20 * time.Second,
			PermitWithoutStream: true,
		}),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(50*1024*1024), // 50MB for video frames
			grpc.MaxCallSendMsgSize(10*1024*1024),
		),
	}

	conn, err := grpc.NewClient(addr, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to inference server at %s: %w", addr, err)
	}

	return &Client{
		conn:     conn,
		avatar:   pb.NewAvatarServiceClient(conn),
		llm:      pb.NewLLMServiceClient(conn),
		tts:      pb.NewTTSServiceClient(conn),
		asr:      pb.NewASRServiceClient(conn),
		voiceLLM: pb.NewVoiceLLMServiceClient(conn),
	}, nil
}

// Close shuts down the gRPC connection.
func (c *Client) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// HealthCheck verifies the inference server is reachable.
func (c *Client) HealthCheck(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	_, err := c.AvatarInfo(ctx)
	return err
}

func (c *Client) AvatarInfo(ctx context.Context) (*pb.AvatarInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	return c.avatar.GetInfo(ctx, &pb.GetInfoRequest{})
}
