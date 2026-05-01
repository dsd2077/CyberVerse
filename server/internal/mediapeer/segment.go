package mediapeer

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os/exec"
	"time"

	"github.com/pion/webrtc/v4/pkg/media"
	"github.com/pion/webrtc/v4/pkg/media/ivfreader"
	opus "gopkg.in/hraban/opus.v2"
)

// RawAVSegment is an unencoded video+audio segment ready for VP8 encoding.
type RawAVSegment struct {
	TraceLabel  string
	Epoch       uint64
	RGB         []byte
	PCM         []byte
	UserFinalAt time.Time
	SampleRate  int
	Width       int
	Height      int
	FPS         int
	NumFrames   int
	QueuedAt    time.Time // set by SendAVSegment for pipeline latency tracking

	// Fence is a pipeline drain marker. When set (non-nil) and RGB is empty,
	// the encoder passes it through without encoding. The publisher closes
	// the channel after finishing all preceding segments, signalling the
	// producer that the pipeline has been fully drained.
	Fence chan struct{}
}

// AVSegment is a pre-encoded video+audio segment ready for paced publishing.
type AVSegment struct {
	TraceLabel  string
	Epoch       uint64
	VP8Samples  []media.Sample
	PCM         []byte
	UserFinalAt time.Time
	SampleRate  int
	Width       int
	Height      int
	FPS         int
	NumFrames   int
	QueuedAt    time.Time // carried from RawAVSegment for end-to-end latency
	Fence       chan struct{}
}

// EncodeRGBChunkToVP8Samples encodes a contiguous RGB24 buffer to VP8 samples.
func EncodeRGBChunkToVP8Samples(rgb []byte, width, height, numFrames, fps int) ([]media.Sample, error) {
	if numFrames <= 0 || width <= 0 || height <= 0 {
		return nil, nil
	}
	if fps <= 0 {
		fps = 25
	}
	want := width * height * 3 * numFrames
	if len(rgb) < want {
		return nil, fmt.Errorf("rgb buffer too small: got %d want >= %d", len(rgb), want)
	}
	rgb = rgb[:want]

	frameDur := time.Second / time.Duration(fps)
	if frameDur <= 0 {
		frameDur = time.Millisecond * 40
	}

	// IVF VP8 via libvpx; single-process encode per chunk (latency vs simplicity tradeoff).
	cmd := exec.Command(
		"ffmpeg",
		"-loglevel", "error",
		"-f", "rawvideo",
		"-pixel_format", "rgb24",
		"-video_size", fmt.Sprintf("%dx%d", width, height),
		"-framerate", fmt.Sprintf("%d", fps),
		"-i", "pipe:0",
		"-frames:v", fmt.Sprintf("%d", numFrames),
		"-c:v", "libvpx",
		"-deadline", "realtime",
		"-cpu-used", "8",
		"-row-mt", "1",
		"-b:v", "2M",
		"-g", "1", // All-Intra: every frame is a keyframe; eliminates reference-frame corruption
		"-an",
		"-f", "ivf",
		"pipe:1",
	)
	cmd.Stdin = bytes.NewReader(rgb)
	var ivfOut bytes.Buffer
	cmd.Stdout = &ivfOut
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg vp8: %w: %s", err, stderr.String())
	}

	reader, fh, err := ivfreader.NewWith(&ivfOut)
	if err != nil {
		return nil, fmt.Errorf("ivf header: %w", err)
	}

	n := int(fh.NumFrames)
	if n <= 0 {
		n = numFrames
	}

	var samples []media.Sample
	for i := 0; i < n; i++ {
		payload, _, err := reader.ParseNextFrame()
		if err != nil {
			break
		}
		if len(payload) == 0 {
			continue
		}
		frame := make([]byte, len(payload))
		copy(frame, payload)
		samples = append(samples, media.Sample{
			Data:     frame,
			Duration: frameDur,
		})
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("ffmpeg produced zero vp8 frames")
	}
	return samples, nil
}

// EncodePCMToOpusSamples encodes an entire 16-bit mono PCM buffer into
// 20ms Opus frames returned as media.Samples. Encoding the full buffer
// at once avoids the sample-loss caused by splitting PCM into small
// slices and encoding each independently.
func EncodePCMToOpusSamples(pcm []byte, sampleRate int) ([]media.Sample, error) {
	if len(pcm) == 0 || sampleRate <= 0 {
		return nil, nil
	}

	enc, err := opus.NewEncoder(sampleRate, 1, opus.AppVoIP)
	if err != nil {
		return nil, fmt.Errorf("create opus encoder: %w", err)
	}

	// 20ms frame parameters
	samplesPerFrame := sampleRate / 50   // 20ms = 1/50 second
	bytesPerFrame := samplesPerFrame * 2 // 16-bit mono
	opusBuf := make([]byte, 4000)        // max opus frame size

	estimatedFrames := len(pcm) / bytesPerFrame
	samples := make([]media.Sample, 0, estimatedFrames)

	for offset := 0; offset+bytesPerFrame <= len(pcm); offset += bytesPerFrame {
		pcmSamples := make([]int16, samplesPerFrame)
		for i := 0; i < samplesPerFrame; i++ {
			pcmSamples[i] = int16(binary.LittleEndian.Uint16(pcm[offset+i*2:]))
		}

		n, err := enc.Encode(pcmSamples, opusBuf)
		if err != nil {
			return nil, fmt.Errorf("opus encode: %w", err)
		}
		if n == 0 {
			continue
		}

		frame := make([]byte, n)
		copy(frame, opusBuf[:n])
		samples = append(samples, media.Sample{
			Data:     frame,
			Duration: 20 * time.Millisecond,
		})
	}
	return samples, nil
}
