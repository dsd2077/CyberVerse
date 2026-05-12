# LangGraph Agent Development Plan

> **For agentic workers:** REQUIRED: Use `superpowers:subagent-driven-development` if subagents are available, or `superpowers:executing-plans` to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 `features` 分支的 PersonaAgent 和 Agent Worker 基础上，继续用 LangGraph 补齐可落地的后台任务能力。

**Architecture:** 当前分支已经不是从零接入 LangGraph：`agent_runtime/graph.py` 已实现后台 research graph，`inference/plugins/voice_llm/persona_agent.py` 已用 PersonaAgent 包装实时 Omni provider，并用工具调用创建后台任务。后续开发应保持 Go TaskService 管状态、Python LangGraph Worker 执行任务、PersonaAgent 负责实时语音编排的边界。

**Tech Stack:** Python 3.10+, FastAPI, LangGraph, langgraph-checkpoint-sqlite, OpenAI-compatible LLM, Go TaskService, SQLite, Vue frontend.

---

## Current Understanding

- 当前本地分支为 `features`，跟踪 `origin/features`。
- `pyproject.toml` 的 `agent` extra 已包含 `langgraph>=0.2.0`、`langgraph-checkpoint-sqlite>=2.0.0`、`aiosqlite>=0.20`。
- `agent_runtime/graph.py` 已有 `classify_task -> plan_task -> run_research -> draft_artifact -> finalize` 的 LangGraph research graph。
- `agent_runtime/llm.py` 已有 OpenAI-compatible 文本 LLM 客户端，默认支持从 CyberVerse 配置读取 Qwen/OpenAI 兼容参数。
- `agent_runtime/tools.py` 当前只有 `NullSearchTool` 和 `MockSearchTool`，真实搜索适配仍是缺口。
- `inference/plugins/voice_llm/persona_agent.py` 已实现 Omni 包装、hidden tools、后台任务异步启动、任务终态回灌 Omni session。
- Go 侧 `server/internal/agenttask` 已负责任务持久化、事件、artifact、worker dispatch 和取消。
- 已有文档 `docs/zh-CN/features/2026-05-11-persona-agent-task-mvp.md` 记录了 MVP 主链路。

## Development Assumptions

- 不重写已经存在的 PersonaAgent、TaskService 或 LangGraph MVP 主链路。
- 短期任务类型继续聚焦 `research`，避免提前抽象多 Agent 类型。
- 搜索能力优先通过一个真实 `SearchTool` 适配器接入，而不是把搜索逻辑写进 graph 节点。
- checkpoint 继续使用 SQLite，先保证本地开发和单机部署稳定。
- 前端改动如涉及用户可见文本，需要同时维护中文和英文 i18n 文案。

## File Map

- Modify: `agent_runtime/tools.py` - 增加真实搜索工具适配器，保留 `NullSearchTool` 作为默认降级。
- Modify: `agent_runtime/server.py` - 根据配置选择 `SearchTool`，避免默认永远使用 `NullSearchTool`。
- Modify: `agent_runtime/graph.py` - 只在需要时补充节点事件、错误路径或更清晰的状态字段。
- Modify: `agent_runtime/llm.py` - 如有必要，收紧 JSON 输出解析和错误提示。
- Modify: `inference/plugins/voice_llm/persona_agent.py` - 只修正 PersonaAgent 工具调用和任务回灌的明确缺陷。
- Modify: `server/internal/agenttask/*.go` - 只在任务状态、取消、artifact API 有真实需求时小范围修改。
- Modify: `frontend/src/i18n/messages.ts` and related task UI files - 仅当任务进度或 artifact 展示需要补齐时修改。
- Test: `tests/unit/test_agent_runtime_*.py` - 覆盖 LangGraph、LLM、SearchTool、Callback 行为。
- Test: `tests/unit/test_persona_agent_plugin.py` - 覆盖 PersonaAgent 工具调用、半句等待、任务回灌。
- Test: `server/internal/agenttask/*_test.go` and `server/internal/api/*_test.go` - 覆盖 Go 任务状态和 API。

## Chunk 1: Baseline Verification

### Task 1: Verify existing LangGraph MVP behavior

**Files:**
- Read: `agent_runtime/graph.py`
- Read: `agent_runtime/server.py`
- Read: `inference/plugins/voice_llm/persona_agent.py`
- Test: existing unit tests

- [ ] **Step 1: Run focused Python tests**

Run:

```bash
python -m pytest tests/unit/test_agent_runtime_llm.py tests/unit/test_persona_agent_plugin.py -q
```

Expected: PASS. If it fails due missing optional dependencies, install the project with `agent` extra or document the exact missing package.

- [ ] **Step 2: Run focused Go task tests**

Run:

```bash
go test ./server/internal/agenttask ./server/internal/api
```

Expected: PASS.

- [ ] **Step 3: Record current gaps**

Expected gaps:

- `NullSearchTool` means research artifacts cannot include live external search yet.
- Agent Worker search provider is not configuration-driven yet.
- End-to-end task flow depends on runtime config and real LLM credentials.

## Chunk 2: Search Tool Adapter

### Task 2: Add a real configurable SearchTool

**Files:**
- Modify: `agent_runtime/tools.py`
- Modify: `agent_runtime/server.py`
- Test: `tests/unit/test_agent_runtime_tools.py`

- [ ] **Step 1: Write failing tests for provider selection**

Create tests that assert:

- default config returns `NullSearchTool`;
- explicit mock config returns deterministic mock results;
- future real provider config can be selected without changing graph code.

Run:

```bash
python -m pytest tests/unit/test_agent_runtime_tools.py -q
```

Expected: FAIL until factory code exists.

- [ ] **Step 2: Implement minimal factory**

Add a small factory such as `build_search_tool_from_runtime_config(config)` in `agent_runtime/tools.py`.

Keep it simple:

- support `provider: "null"`;
- support `provider: "mock"` for tests and local demos;
- leave real provider implementation behind one narrow class when credentials/API are known.

- [ ] **Step 3: Wire Agent Worker dependency**

Change `agent_runtime/server.py:get_search_tool()` or app state initialization so the worker uses the configured tool.

Do not modify `agent_runtime/graph.py` for provider-specific behavior.

- [ ] **Step 4: Verify**

Run:

```bash
python -m pytest tests/unit/test_agent_runtime_tools.py tests/unit/test_agent_runtime_llm.py -q
```

Expected: PASS.

## Chunk 3: LangGraph Robustness

### Task 3: Make graph failure states explicit

**Files:**
- Modify: `agent_runtime/graph.py`
- Test: `tests/unit/test_agent_runtime_graph.py`

- [ ] **Step 1: Add tests for node failures**

Cover:

- LLM classification returns invalid JSON;
- search tool raises an exception;
- artifact callback fails.

Expected behavior should be explicit: either emit `task.failed` through `agent_runtime/server.py` or emit a graph-level blocked event before failing.

- [ ] **Step 2: Implement the smallest error-path improvement**

Prefer one of these scoped options:

- let `_run_task()` own final failure events and keep graph nodes straightforward;
- or emit a specific `research.blocked` event only for search-provider unavailable cases.

Avoid broad retry frameworks until a concrete transient failure is observed.

- [ ] **Step 3: Verify**

Run:

```bash
python -m pytest tests/unit/test_agent_runtime_graph.py tests/unit/test_inference_server.py -q
```

Expected: PASS.

## Chunk 4: PersonaAgent Task UX

### Task 4: Confirm voice task flow stays non-blocking

**Files:**
- Modify: `inference/plugins/voice_llm/persona_agent.py`
- Test: `tests/unit/test_persona_agent_plugin.py`

- [ ] **Step 1: Add tests before behavior changes**

Add or extend tests for:

- `create_task` returns an accepted tool result before task completion;
- task completion prompt is injected only after assistant acknowledgement final event;
- cancelled or failed task produces a short natural-language final prompt.

- [ ] **Step 2: Patch only failing behavior**

If tests expose a bug, patch only the relevant branch in `converse_stream()` or task monitoring helpers.

Do not change tool schemas unless the real Omni provider requires it.

- [ ] **Step 3: Verify**

Run:

```bash
python -m pytest tests/unit/test_persona_agent_plugin.py -q
```

Expected: PASS.

## Chunk 5: Frontend Task Visibility

### Task 5: Ensure task events and artifacts are visible in the UI

**Files:**
- Inspect: `frontend/src/services/api.ts`
- Inspect: `frontend/src/composables/useChat.ts`
- Inspect: `frontend/src/components/ChatPanel.vue`
- Modify: `frontend/src/i18n/messages.ts` only if new user-facing text is needed.

- [ ] **Step 1: Verify existing UI support**

Check whether WebSocket `task_event` payloads are rendered and whether artifact links can be opened.

- [ ] **Step 2: Add missing minimal UI behavior**

If missing, add only:

- a task progress message in chat;
- an artifact link when `artifact_id` appears;
- Chinese and English i18n entries for new visible text.

- [ ] **Step 3: Verify**

Run:

```bash
cd frontend && npm run build
```

Expected: PASS.

## Chunk 6: End-to-End Local Run

### Task 6: Validate local task flow

**Files:**
- Read/Modify only if needed: `infra/cyberverse_config.example.yaml`
- Read: `README.md`
- Read: `README.zh-CN.md`

- [ ] **Step 1: Start services**

Run the project-standard local commands from RTK or README. Expected services:

- inference service with PersonaAgent;
- Go server with task service enabled;
- frontend dev server.

- [ ] **Step 2: Trigger a research request**

Use a PersonaAgent session and ask a research-style request, for example:

```text
帮我查一下今天知乎有哪些热门信息，并整理成简短报告。
```

Expected:

- digital human first acknowledges quickly;
- Go task event stream reports queued/running/completed;
- LangGraph creates a markdown artifact;
- final voice reply summarizes completion.

- [ ] **Step 3: Verify cancellation**

Start a task, then ask to stop it.

Expected:

- latest active task becomes `cancelled`;
- frontend receives task event;
- PersonaAgent produces a short cancellation reply.

## Final Verification

Run:

```bash
python -m pytest tests/unit/test_agent_runtime_llm.py tests/unit/test_persona_agent_plugin.py -q
go test ./server/internal/agenttask ./server/internal/api
cd frontend && npm run build
```

Expected: all commands pass, or any unavailable dependency is documented with the exact missing package and command that failed.

## Notes

- Keep changes narrow. The current branch already contains the core LangGraph architecture.
- Do not introduce a second task orchestrator beside Go TaskService.
- Do not make PersonaAgent parse visible JSON from model text; keep structured tool calls as the control path.
- Keep code comments in English.
- For frontend text, maintain both Chinese and English translations.
