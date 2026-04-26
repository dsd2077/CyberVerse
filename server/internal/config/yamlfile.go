package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

var yamlMu sync.Mutex

// ReadYAMLNode reads a YAML file into a yaml.Node tree without expanding env vars.
func ReadYAMLNode(path string) (*yaml.Node, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	// yaml.Unmarshal wraps in a DocumentNode; return the inner mapping.
	if doc.Kind == yaml.DocumentNode && len(doc.Content) > 0 {
		return &doc, nil
	}
	return &doc, nil
}

// mappingRoot returns the top-level mapping node from a document node.
func mappingRoot(doc *yaml.Node) *yaml.Node {
	if doc.Kind == yaml.DocumentNode && len(doc.Content) > 0 {
		return doc.Content[0]
	}
	return doc
}

// GetNodeAtPath traverses a yaml.Node tree by dot-separated path.
// An empty dotPath returns the root mapping node.
func GetNodeAtPath(doc *yaml.Node, dotPath string) (*yaml.Node, error) {
	node := mappingRoot(doc)
	if dotPath == "" {
		return node, nil
	}
	parts := strings.Split(dotPath, ".")

	for _, key := range parts {
		if node.Kind != yaml.MappingNode {
			return nil, fmt.Errorf("expected mapping at %q, got kind %d", key, node.Kind)
		}
		found := false
		for i := 0; i < len(node.Content)-1; i += 2 {
			if node.Content[i].Value == key {
				node = node.Content[i+1]
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("key %q not found in path %q", key, dotPath)
		}
	}
	return node, nil
}

// GetMappingKeys returns all keys of a mapping node at the given dot-path.
func GetMappingKeys(doc *yaml.Node, dotPath string) ([]string, error) {
	node, err := GetNodeAtPath(doc, dotPath)
	if err != nil {
		return nil, err
	}
	if node.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("node at %q is not a mapping", dotPath)
	}
	var keys []string
	for i := 0; i < len(node.Content)-1; i += 2 {
		keys = append(keys, node.Content[i].Value)
	}
	return keys, nil
}

// SetNodeAtPath sets a scalar value at the given dot-path.
// It auto-detects numeric types so the YAML tag is correct.
func SetNodeAtPath(doc *yaml.Node, dotPath string, value string) error {
	node, err := GetNodeAtPath(doc, dotPath)
	if err != nil {
		return err
	}
	if node.Kind != yaml.ScalarNode {
		return fmt.Errorf("node at %q is not a scalar (kind %d)", dotPath, node.Kind)
	}

	node.Value = value
	node.Tag = inferYAMLTag(value)
	node.Style = 0 // reset style so yaml.v3 picks the natural representation
	return nil
}

// WriteYAMLNode atomically writes a yaml.Node tree back to disk.
func WriteYAMLNode(path string, doc *yaml.Node) error {
	yamlMu.Lock()
	defer yamlMu.Unlock()

	out, err := yaml.Marshal(doc)
	if err != nil {
		return fmt.Errorf("marshal yaml: %w", err)
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, out, 0644); err != nil {
		return fmt.Errorf("write temp file: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}

// NodeScalarValue returns the string value of a scalar node,
// optionally expanding ${ENV_VAR} references for display.
func NodeScalarValue(node *yaml.Node, expandEnv bool) string {
	if node.Kind != yaml.ScalarNode {
		return ""
	}
	v := node.Value
	if expandEnv && strings.Contains(v, "${") {
		v = os.ExpandEnv(v)
	}
	return v
}

// NodeValue returns the value as an appropriate Go type (string, int, float64, bool).
func NodeValue(node *yaml.Node, expandEnv bool) any {
	if node.Kind != yaml.ScalarNode {
		return node.Value
	}
	raw := node.Value
	display := raw
	if expandEnv && strings.Contains(raw, "${") {
		display = os.ExpandEnv(raw)
	}

	// Try int
	if i, err := strconv.ParseInt(display, 10, 64); err == nil {
		return i
	}
	// Try float
	if f, err := strconv.ParseFloat(display, 64); err == nil {
		return f
	}
	// Try bool
	if display == "true" {
		return true
	}
	if display == "false" {
		return false
	}
	return display
}

func inferYAMLTag(value string) string {
	if _, err := strconv.ParseInt(value, 10, 64); err == nil {
		return "!!int"
	}
	if _, err := strconv.ParseFloat(value, 64); err == nil {
		return "!!float"
	}
	if value == "true" || value == "false" {
		return "!!bool"
	}
	return "!!str"
}
