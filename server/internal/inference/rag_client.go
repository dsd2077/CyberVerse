package inference

import (
	"context"
	"time"

	pb "github.com/cyberverse/server/internal/pb"
)

type RAGIndexSourceRequest struct {
	CharacterID  string
	CharacterDir string
	SourceID     string
	SourceType   string
	Title        string
	Filename     string
	MimeType     string
	SourcePath   string
}

type RAGSearchRequest struct {
	CharacterID     string
	CharacterDir    string
	Query           string
	TopK            int
	MaxContextChars int
	MinScore        float32
}

type RAGSearchResult struct {
	SourceID   string
	SourceType string
	Title      string
	Filename   string
	Content    string
	Score      float32
}

type RAGService interface {
	IndexRAGSource(ctx context.Context, req RAGIndexSourceRequest) (int, error)
	DeleteRAGSource(ctx context.Context, characterID, characterDir, sourceID string) error
	SearchRAG(ctx context.Context, req RAGSearchRequest) ([]RAGSearchResult, error)
}

func (c *Client) IndexRAGSource(ctx context.Context, req RAGIndexSourceRequest) (int, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	resp, err := c.rag.IndexSource(ctx, &pb.RAGIndexSourceRequest{
		CharacterId:  req.CharacterID,
		CharacterDir: req.CharacterDir,
		SourceId:     req.SourceID,
		SourceType:   req.SourceType,
		Title:        req.Title,
		Filename:     req.Filename,
		MimeType:     req.MimeType,
		SourcePath:   req.SourcePath,
	})
	if err != nil {
		return 0, err
	}
	return int(resp.GetChunkCount()), nil
}

func (c *Client) DeleteRAGSource(ctx context.Context, characterID, characterDir, sourceID string) error {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	_, err := c.rag.DeleteSource(ctx, &pb.RAGDeleteSourceRequest{
		CharacterId:  characterID,
		CharacterDir: characterDir,
		SourceId:     sourceID,
	})
	return err
}

func (c *Client) SearchRAG(ctx context.Context, req RAGSearchRequest) ([]RAGSearchResult, error) {
	ctx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	resp, err := c.rag.Search(ctx, &pb.RAGSearchRequest{
		CharacterId:     req.CharacterID,
		CharacterDir:    req.CharacterDir,
		Query:           req.Query,
		TopK:            int32(req.TopK),
		MaxContextChars: int32(req.MaxContextChars),
		MinScore:        req.MinScore,
	})
	if err != nil {
		return nil, err
	}
	results := make([]RAGSearchResult, 0, len(resp.GetResults()))
	for _, item := range resp.GetResults() {
		results = append(results, RAGSearchResult{
			SourceID:   item.GetSourceId(),
			SourceType: item.GetSourceType(),
			Title:      item.GetTitle(),
			Filename:   item.GetFilename(),
			Content:    item.GetContent(),
			Score:      item.GetScore(),
		})
	}
	return results, nil
}
