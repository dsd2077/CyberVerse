package orchestrator

import "testing"

func TestDesiredSamplesForVideo(t *testing.T) {
	got := desiredSamplesForVideo(28, 25, 24000)
	if got != 26880 {
		t.Fatalf("expected 26880, got %d", got)
	}
}

func TestVoiceAVSyncBufferTakeSegmentPCM(t *testing.T) {
	buf := newVoiceAVSyncBuffer(24000)
	input := make([]byte, 32000) // 16000 samples
	buf.appendPCM(input, 24000)

	out, outSamples, wantSamples, bufferedAfter := buf.takeSegmentPCM(25, 25, false) // 1s => 24000 samples
	if wantSamples != 24000 {
		t.Fatalf("expected wantSamples=24000, got %d", wantSamples)
	}
	if outSamples != 16000 {
		t.Fatalf("expected outSamples=16000, got %d", outSamples)
	}
	if len(out) != 48000 {
		t.Fatalf("expected strict padded out bytes=48000, got %d", len(out))
	}
	if bufferedAfter != 0 {
		t.Fatalf("expected bufferedAfter=0, got %d", bufferedAfter)
	}
}

func TestVoiceAVSyncBufferOverflowDropsOldest(t *testing.T) {
	buf := newVoiceAVSyncBuffer(10) // 10 samples => 20 bytes max
	dropped := buf.appendPCM(make([]byte, 30), 24000)
	if dropped != 10 {
		t.Fatalf("expected dropped=10, got %d", dropped)
	}
	bufferedSamples, _, _, _ := buf.snapshot()
	if bufferedSamples != 10 {
		t.Fatalf("expected bufferedSamples=10, got %d", bufferedSamples)
	}
}
