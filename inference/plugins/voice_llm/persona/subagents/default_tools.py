from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from langchain.tools import tool

from inference.plugins.voice_llm.persona.llm import build_agent_llm
from inference.plugins.voice_llm.persona.i18n import Localizer
from inference.plugins.voice_llm.persona.schemas import ArtifactRequest, Task, TaskEvent
from inference.plugins.voice_llm.persona.subagents.agent import TaskCallbacks, run_subagent
from inference.plugins.voice_llm.persona.tools import SearchResult, SearchTool, ZhihuClient, ZhihuConfig, ZhihuToolExecutor


DEFAULT_TOOL_LABELS = {
    "zhihu_search": "知乎搜索",
    "global_search": "全网搜索",
    "zhida": "知乎直答",
    "hot_list": "知乎热榜",
    "create_html_report": "生成 HTML 页面",
}

DEFAULT_TERMINAL_TOOL_NAMES = {"create_html_report"}


ZHIHU_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "具体的知乎站内搜索关键词。"},
        "count": {"type": "integer", "minimum": 1, "maximum": 10, "default": 10},
    },
    "required": ["query"],
}

GLOBAL_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "具体的全网搜索关键词。"},
        "count": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
    },
    "required": ["query"],
}

ZHIDA_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "需要知乎直答回答的问题。"},
        "model": {
            "type": "string",
            "enum": ["zhida-fast-1p5", "zhida-thinking-1p5", "zhida-agent"],
            "default": "zhida-fast-1p5",
        },
    },
    "required": ["query"],
}

HOT_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 30},
    },
}

HTML_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "paragraphs": {"type": "array", "items": {"type": "string"}},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["heading"],
            },
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "source_type": {"type": "string"},
                    "author": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["title"],
            },
        },
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "sections", "sources"],
}


def _string_list(value: Any, limit: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value[:limit] if str(item or "").strip()]


def _dict_list(value: Any, limit: int = 50) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value[:limit] if isinstance(item, dict)]


def _safe_url(value: Any) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


def _model_provider(model: Any) -> str:
    return str(
        getattr(model, "model_provider", None)
        or getattr(model, "provider", None)
        or model.__class__.__name__
    )


def _model_name(model: Any) -> str:
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or getattr(model, "model_id", None)
        or model.__class__.__name__
    )


def _render_html_report(task: Task, payload: dict[str, Any], *, generated_at: datetime) -> str:
    title = str(payload.get("title") or task.title or "报告").strip()
    summary = str(payload.get("summary") or "").strip()
    sections = _dict_list(payload.get("sections"))
    sources = _dict_list(payload.get("sources"))
    caveats = _string_list(payload.get("caveats"))

    def esc(value: Any) -> str:
        return html.escape(str(value or "").strip(), quote=True)

    section_html: list[str] = []
    for section in sections:
        heading = esc(section.get("heading") or "分析")
        paragraphs = _string_list(section.get("paragraphs"))
        bullets = _string_list(section.get("bullets"))
        body = [f"<h2>{heading}</h2>"]
        body.extend(f"<p>{esc(paragraph)}</p>" for paragraph in paragraphs)
        if bullets:
            body.append("<ul>")
            body.extend(f"<li>{esc(bullet)}</li>" for bullet in bullets)
            body.append("</ul>")
        section_html.append(f"<section>{''.join(body)}</section>")

    source_html: list[str] = []
    for index, source in enumerate(sources, start=1):
        source_title = esc(source.get("title") or f"来源 {index}")
        source_url = _safe_url(source.get("url"))
        source_type = esc(source.get("source_type") or "")
        author = esc(source.get("author") or "")
        note = esc(source.get("note") or "")
        title_part = (
            f'<a href="{html.escape(source_url, quote=True)}" target="_blank" rel="noreferrer">{source_title}</a>'
            if source_url
            else source_title
        )
        meta = " · ".join(part for part in [source_type, author] if part)
        meta_part = f'<div class="source-meta">{meta}</div>' if meta else ""
        note_part = f"<p>{note}</p>" if note else ""
        source_html.append(
            "<li>"
            f"<div class=\"source-title\">{title_part}</div>"
            f"{meta_part}"
            f"{note_part}"
            "</li>"
        )

    caveat_html = ""
    if caveats:
        caveat_html = "<section><h2>注意事项</h2><ul>" + "".join(f"<li>{esc(item)}</li>" for item in caveats) + "</ul></section>"

    generated = generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f6f7f9; color: #20242a; }}
    main {{ max-width: 960px; margin: 0 auto; padding: 40px 20px 56px; }}
    header {{ border-bottom: 1px solid #dde1e7; padding-bottom: 24px; margin-bottom: 28px; }}
    h1 {{ font-size: 34px; line-height: 1.18; margin: 0 0 14px; letter-spacing: 0; }}
    h2 {{ font-size: 21px; margin: 0 0 12px; letter-spacing: 0; }}
    p, li {{ font-size: 15px; line-height: 1.72; }}
    section {{ background: #fff; border: 1px solid #e4e7eb; border-radius: 8px; padding: 22px 24px; margin: 16px 0; }}
    ul {{ padding-left: 22px; }}
    a {{ color: #155bd4; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .summary {{ font-size: 17px; line-height: 1.7; color: #3a424d; }}
    .meta {{ color: #667085; font-size: 13px; }}
    .sources {{ list-style: decimal; }}
    .source-title {{ font-weight: 650; }}
    .source-meta {{ margin-top: 4px; color: #667085; font-size: 13px; }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="meta">CyberVerse PersonaAgent · {esc(generated)}</div>
      <h1>{esc(title)}</h1>
      <p class="summary">{esc(summary)}</p>
    </header>
    {''.join(section_html) if section_html else '<section><h2>摘要</h2><p>没有可展示的分节内容。</p></section>'}
    <section>
      <h2>来源</h2>
      <ol class="sources">{''.join(source_html) if source_html else '<li>未提供可打开来源。</li>'}</ol>
    </section>
    {caveat_html}
  </main>
</body>
</html>
"""


def build_default_subagent_tools(
    *,
    task: Task,
    tool_executor: ZhihuToolExecutor,
    callbacks: Any,
    model: Any,
    tool_runtime_context: dict[str, Any],
) -> list[Any]:
    @tool(
        "zhihu_search",
        description="在知乎站内搜索与查询相关的问题、回答和文章。",
        args_schema=ZHIHU_SEARCH_SCHEMA,
    )
    async def zhihu_search(query: str, count: int = 10) -> dict[str, Any]:
        return await tool_executor.execute("zhihu_search", {"query": query, "count": count})

    @tool(
        "global_search",
        description="当需要知乎站外或更广泛的外部参考时，通过知乎开放平台进行全网搜索。",
        args_schema=GLOBAL_SEARCH_SCHEMA,
    )
    async def global_search(query: str, count: int = 10) -> dict[str, Any]:
        return await tool_executor.execute("global_search", {"query": query, "count": count})

    @tool(
        "zhida",
        description="向知乎直答提问，获取针对问题的直接回答或综合分析。",
        args_schema=ZHIDA_SCHEMA,
    )
    async def zhida(query: str, model: str = "") -> dict[str, Any]:
        return await tool_executor.execute("zhida", {"query": query, "model": model})

    @tool(
        "hot_list",
        description="获取当前知乎热榜列表。",
        args_schema=HOT_LIST_SCHEMA,
    )
    async def hot_list(limit: int = 30) -> dict[str, Any]:
        return await tool_executor.execute("hot_list", {"limit": limit})

    @tool(
        "create_html_report",
        description="生成最终 HTML 报告并结束任务。工具无结果或失败时也要调用，用 caveats 清楚说明来源不足和限制。",
        args_schema=HTML_REPORT_SCHEMA,
    )
    async def create_html_report(**payload: Any) -> dict[str, Any]:
        generated_at = datetime.now(timezone.utc)
        title = str(payload.get("title") or task.title or "报告").strip()
        summary = str(payload.get("summary") or "HTML 页面已生成。").strip()
        sections = _dict_list(payload.get("sections"))
        sources = _dict_list(payload.get("sources"))
        content = _render_html_report(task, payload, generated_at=generated_at)
        artifact = await callbacks.artifact(
            task.id,
            ArtifactRequest(
                type="html",
                title=title,
                mime_type="text/html; charset=utf-8",
                content=content,
                metadata={
                    "locale": task.locale,
                    "llm_provider": _model_provider(model),
                    "llm_model": _model_name(model),
                    "source_count": len(sources),
                    "section_count": len(sections),
                    "generated_at": generated_at.isoformat(),
                    "tool_trace": list(tool_runtime_context.get("tool_trace") or []),
                },
            ),
        )
        artifact_id = artifact.get("id") if isinstance(artifact, dict) else None
        await callbacks.event(
            task.id,
            TaskEvent(
                event_type="task.completed",
                status="completed",
                message=summary,
                progress=100,
                payload={"artifact_id": artifact_id},
            ),
        )
        return {"ok": True, "artifact_id": artifact_id, "summary": summary}

    return [zhihu_search, global_search, zhida, hot_list, create_html_report]


async def run_task_with_langgraph(
    task: Task,
    search_tool: SearchTool,
    callbacks: TaskCallbacks,
    llm: Any | None = None,
    *,
    tool_executor: ZhihuToolExecutor | None = None,
    max_agent_iterations: int = 8,
) -> None:
    model = llm or build_agent_llm()
    if tool_executor is None:
        tool_executor = ZhihuToolExecutor(ZhihuClient(ZhihuConfig()))
    tool_runtime_context: dict[str, Any] = {"tool_trace": []}
    tools = build_default_subagent_tools(
        task=task,
        tool_executor=tool_executor,
        callbacks=callbacks,
        model=model,
        tool_runtime_context=tool_runtime_context,
    )
    await run_subagent(
        task=task,
        model=model,
        tools=tools,
        callbacks=callbacks,
        max_agent_iterations=max_agent_iterations,
        terminal_tool_names=DEFAULT_TERMINAL_TOOL_NAMES,
        tool_labels=DEFAULT_TOOL_LABELS,
        tool_runtime_context=tool_runtime_context,
    )


def _draft_markdown(task: Task, results: list[dict[str, str]], localizer: Localizer) -> str:
    lines: list[str] = [
        f"# {task.title}",
        "",
        f"{localizer.text('artifact.user_request')}{localizer.text('artifact.label_separator')}{task.user_request}",
        "",
        f"## {localizer.text('artifact.current_status')}",
    ]
    if not results:
        lines.extend(
            [
                localizer.text("artifact.null_search_line_1"),
                localizer.text("artifact.null_search_line_2"),
            ]
        )
    else:
        lines.append(localizer.text("artifact.results_intro"))
        for index, result in enumerate(results, start=1):
            lines.extend(
                [
                    "",
                    f"### {index}. {result['title']}",
                    result["snippet"],
                    result["url"],
                ]
            )
    return "\n".join(lines).strip() + "\n"


def _result_dict(result: SearchResult) -> dict[str, str]:
    return {"title": result.title, "url": result.url, "snippet": result.snippet}
