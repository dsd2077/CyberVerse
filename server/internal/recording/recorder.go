package recording

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"

	"github.com/cyberverse/server/internal/config"
)

// VideoRecorder is held by the Orchestrator and creates per-turn recordings.
type VideoRecorder struct {
	cfg config.RecordingConfig
}

// NewVideoRecorder creates a new VideoRecorder.
func NewVideoRecorder(cfg config.RecordingConfig) *VideoRecorder {
	return &VideoRecorder{cfg: cfg}
}

func (r *VideoRecorder) resolveOutputDir(sessionDir string) string {
	if filepath.IsAbs(sessionDir) {
		return sessionDir
	}
	return filepath.Join(r.cfg.OutputDir, sessionDir)
}

// EncodeRGB24ToMP4 writes raw RGB24 frames plus optional mono PCM16 audio into an MP4.
func EncodeRGB24ToMP4(outPath string, width, height, fps int, rgbChunks [][]byte, pcm []byte, sampleRate int, crf int) error {
	if width <= 0 || height <= 0 || fps <= 0 {
		return fmt.Errorf("recording: invalid video params width=%d height=%d fps=%d", width, height, fps)
	}
	if len(rgbChunks) == 0 {
		return fmt.Errorf("recording: no rgb chunks to encode")
	}
	if crf <= 0 {
		crf = 23
	}
	if sampleRate <= 0 {
		sampleRate = 16000
	}

	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return fmt.Errorf("recording: ffmpeg not found in PATH")
	}
	if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
		return fmt.Errorf("recording: create output dir: %w", err)
	}

	videoSize := fmt.Sprintf("%dx%d", width, height)
	args := []string{
		"-hide_banner",
		"-loglevel", "error",
		"-y",
		"-f", "rawvideo",
		"-pixel_format", "rgb24",
		"-video_size", videoSize,
		"-framerate", fmt.Sprintf("%d", fps),
		"-i", "pipe:0",
	}

	var audioTmpPath string
	if len(pcm) > 0 {
		// Use os.CreateTemp to avoid naming conflicts when multiple goroutines
		// encode idle videos concurrently.
		tmpFile, err := os.CreateTemp("", "cyberverse-idle-audio-*.pcm")
		if err != nil {
			return fmt.Errorf("recording: create temp pcm file: %w", err)
		}
		audioTmpPath = tmpFile.Name()
		if _, err := tmpFile.Write(pcm); err != nil {
			tmpFile.Close()
			os.Remove(audioTmpPath)
			return fmt.Errorf("recording: write temp pcm: %w", err)
		}
		tmpFile.Close()
		defer os.Remove(audioTmpPath)
		args = append(args,
			"-f", "s16le",
			"-ac", "1",
			"-ar", fmt.Sprintf("%d", sampleRate),
			"-i", audioTmpPath,
		)
	}

	args = append(args,
		"-c:v", "libx264",
		"-preset", "fast",
		"-crf", fmt.Sprintf("%d", crf),
		"-pix_fmt", "yuv420p",
	)
	if len(pcm) > 0 {
		args = append(args,
			"-c:a", "aac",
			"-b:a", "96k",
			"-shortest",
		)
	} else {
		args = append(args, "-an")
	}
	args = append(args, outPath)

	cmd := exec.Command("ffmpeg", args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("recording: ffmpeg stdin pipe: %w", err)
	}

	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("recording: ffmpeg start: %w", err)
	}

	writeErr := func() error {
		for _, chunk := range rgbChunks {
			if len(chunk) == 0 {
				continue
			}
			if _, err := stdin.Write(chunk); err != nil {
				return err
			}
		}
		return nil
	}()
	closeErr := stdin.Close()
	waitErr := cmd.Wait()

	if writeErr != nil {
		return fmt.Errorf("recording: ffmpeg stdin write: %w", writeErr)
	}
	if closeErr != nil {
		return fmt.Errorf("recording: ffmpeg stdin close: %w", closeErr)
	}
	if waitErr != nil {
		return fmt.Errorf("recording: ffmpeg encode failed: %w: %s", waitErr, stderr.String())
	}
	return nil
}

// TurnRecording manages one conversation turn: video encoded via ffmpeg stdin pipe,
// audio (silence-padded segPCM from flushVoiceSeg) accumulated in memory.
// Finish() muxes them into a single MP4.
type TurnRecording struct {
	videoCmd     *exec.Cmd
	videoWriter  io.WriteCloser
	videoDone    chan error
	videoTmpPath string

	mu         sync.Mutex
	audioBuf   bytes.Buffer
	sampleRate int

	outPath string
	cfg     config.RecordingConfig
}

// BeginTurn starts a new turn recording. Returns nil if recording is disabled.
// sessionDir can be an absolute path (used when recording into a character's data space)
// or a relative name (legacy behavior, joined with cfg.OutputDir).
func (r *VideoRecorder) BeginTurn(sessionDir, turnID string, width, height, fps int) *TurnRecording {
	if !r.cfg.Enabled {
		return nil
	}

	outDir := r.resolveOutputDir(sessionDir)
	if err := os.MkdirAll(outDir, 0755); err != nil {
		log.Printf("recording: failed to create dir %s: %v", outDir, err)
		return nil
	}

	outPath := filepath.Join(outDir, turnID+".mp4")
	videoTmpPath := filepath.Join(outDir, turnID+"-video.mp4")
	videoSize := fmt.Sprintf("%dx%d", width, height)

	cmd := exec.Command("ffmpeg", "-y",
		"-f", "rawvideo",
		"-pixel_format", "rgb24",
		"-video_size", videoSize,
		"-framerate", fmt.Sprintf("%d", fps),
		"-i", "pipe:0",
		"-c:v", "libx264",
		"-preset", "fast",
		"-crf", fmt.Sprintf("%d", r.cfg.CRF),
		"-pix_fmt", "yuv420p",
		videoTmpPath,
	)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		log.Printf("recording: stdin pipe error: %v", err)
		return nil
	}
	if err := cmd.Start(); err != nil {
		log.Printf("recording: ffmpeg start error: %v", err)
		return nil
	}

	t := &TurnRecording{
		videoCmd:     cmd,
		videoWriter:  stdin,
		videoDone:    make(chan error, 1),
		videoTmpPath: videoTmpPath,
		outPath:      outPath,
		cfg:          r.cfg,
	}
	go func() { t.videoDone <- cmd.Wait() }()
	return t
}

// WriteVideoChunk writes raw RGB24 bytes into the ffmpeg encoder.
func (t *TurnRecording) WriteVideoChunk(rgbBytes []byte) {
	if t == nil || len(rgbBytes) == 0 {
		return
	}
	if _, err := t.videoWriter.Write(rgbBytes); err != nil {
		log.Printf("recording: video write error: %v", err)
	}
}

// WriteAudioChunk accumulates silence-padded PCM that is already aligned to
// the video segment duration (segPCM from flushVoiceSeg).
func (t *TurnRecording) WriteAudioChunk(pcmBytes []byte, sampleRate int) {
	if t == nil || len(pcmBytes) == 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.sampleRate == 0 {
		t.sampleRate = sampleRate
	}
	t.audioBuf.Write(pcmBytes)
}

// Finish closes the video pipe, writes a WAV tmp file, muxes both into the
// final MP4, then removes the temporary files.
func (t *TurnRecording) Finish() error {
	if t == nil {
		return nil
	}

	// Step 1: finalize video encoding.
	t.videoWriter.Close()
	if err := <-t.videoDone; err != nil {
		log.Printf("recording: video ffmpeg error: %v", err)
		os.Remove(t.videoTmpPath)
		return fmt.Errorf("recording: finalize video failed: %w", err)
	}

	// Step 2: write audio WAV temp file.
	t.mu.Lock()
	pcmData := t.audioBuf.Bytes()
	sr := t.sampleRate
	t.mu.Unlock()

	audioTmpPath := t.videoTmpPath[:len(t.videoTmpPath)-len("-video.mp4")] + "-audio.wav"
	f, err := os.Create(audioTmpPath)
	if err != nil {
		return fmt.Errorf("recording: create audio tmp: %w", err)
	}
	writeWAVHeader(f, len(pcmData), sr)
	f.Write(pcmData)
	f.Close()

	// Step 3: mux video + audio.
	muxCmd := exec.Command("ffmpeg", "-y",
		"-i", t.videoTmpPath,
		"-i", audioTmpPath,
		"-c:v", "copy",
		"-c:a", "aac",
		"-movflags", "+faststart",
		t.outPath,
	)
	if out, err := muxCmd.CombinedOutput(); err != nil {
		log.Printf("recording: mux error: %v\n%s", err, string(out))
		os.Remove(t.videoTmpPath)
		os.Remove(audioTmpPath)
		return fmt.Errorf("recording: mux failed: %w", err)
	}

	// Step 4: cleanup.
	os.Remove(t.videoTmpPath)
	os.Remove(audioTmpPath)
	log.Printf("recording saved: %s", t.outPath)
	return nil
}

// SaveRawAudio saves Doubao's raw PCM output for a turn as a WAV file.
// sessionDir can be an absolute path or a relative name (joined with cfg.OutputDir).
func (r *VideoRecorder) SaveRawAudio(sessionDir, turnID string, pcm []byte, sampleRate int) error {
	if !r.cfg.Enabled || len(pcm) == 0 {
		return nil
	}
	outDir := r.resolveOutputDir(sessionDir)
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return fmt.Errorf("recording: create raw audio dir: %w", err)
	}
	path := filepath.Join(outDir, turnID+"-raw.wav")
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("recording: create raw audio: %w", err)
	}
	defer f.Close()
	writeWAVHeader(f, len(pcm), sampleRate)
	f.Write(pcm)
	log.Printf("recording raw audio saved: %s", path)
	return nil
}

// SaveTranscript saves a turn's assistant reply as a UTF-8 text file.
// sessionDir can be an absolute path or a relative name (joined with cfg.OutputDir).
func (r *VideoRecorder) SaveTranscript(sessionDir, turnID, transcript string) error {
	if !r.cfg.Enabled || turnID == "" || transcript == "" {
		return nil
	}

	outDir := r.resolveOutputDir(sessionDir)
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return fmt.Errorf("recording: create transcript dir: %w", err)
	}

	path := filepath.Join(outDir, turnID+".txt")
	if err := os.WriteFile(path, []byte(transcript), 0644); err != nil {
		return fmt.Errorf("recording: write transcript: %w", err)
	}

	log.Printf("recording transcript saved: %s", path)
	return nil
}

// writeWAVHeader writes a standard 44-byte PCM WAV header (mono, 16-bit LE).
func writeWAVHeader(w io.Writer, dataLen int, sampleRate int) {
	if sampleRate <= 0 {
		sampleRate = 24000
	}
	const (
		channels      = 1
		bitsPerSample = 16
	)
	byteRate := sampleRate * channels * bitsPerSample / 8
	blockAlign := channels * bitsPerSample / 8
	totalLen := 36 + dataLen

	write := func(v any) { binary.Write(w, binary.LittleEndian, v) }

	w.Write([]byte("RIFF"))
	write(uint32(totalLen))
	w.Write([]byte("WAVE"))
	w.Write([]byte("fmt "))
	write(uint32(16))
	write(uint16(1)) // PCM format
	write(uint16(channels))
	write(uint32(sampleRate))
	write(uint32(byteRate))
	write(uint16(blockAlign))
	write(uint16(bitsPerSample))
	w.Write([]byte("data"))
	write(uint32(dataLen))
}
