package inference

import "testing"

func TestVoiceLLMInputPBMapsImageFrame(t *testing.T) {
	req := voiceLLMInputPB(VoiceLLMInputEvent{
		Image: &ImageFrame{
			Data:        []byte{0xff, 0xd8, 0xff, 0x00},
			MimeType:    "image/jpeg",
			Width:       640,
			Height:      360,
			Source:      "screen",
			TimestampMS: 123,
			FrameSeq:    7,
		},
	})
	if req == nil {
		t.Fatal("expected request")
	}
	image := req.GetImage()
	if image == nil {
		t.Fatalf("expected image input, got %T", req.GetInput())
	}
	if string(image.GetData()) != string([]byte{0xff, 0xd8, 0xff, 0x00}) {
		t.Fatalf("unexpected image data: %v", image.GetData())
	}
	if image.GetMimeType() != "image/jpeg" || image.GetWidth() != 640 || image.GetHeight() != 360 {
		t.Fatalf("unexpected image metadata: %+v", image)
	}
	if image.GetSource() != "screen" || image.GetTimestampMs() != 123 || image.GetFrameSeq() != 7 {
		t.Fatalf("unexpected image source/timing metadata: %+v", image)
	}
}
