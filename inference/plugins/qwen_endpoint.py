import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_WS_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"


def dashscope_base_url() -> str:
    return os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_DASHSCOPE_BASE_URL


def dashscope_realtime_ws_url(model: str, service_env_key: str) -> str:
    raw_url = (
        os.environ.get(service_env_key)
        or os.environ.get("DASHSCOPE_WS_URL")
        or _ws_url_from_base_url(dashscope_base_url())
    )
    return _with_model_query(raw_url, model)


def _ws_url_from_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        return DEFAULT_DASHSCOPE_WS_URL

    scheme = "wss" if parsed.scheme in {"http", "https"} else parsed.scheme
    return urlunsplit((scheme, parsed.netloc, "/api-ws/v1/realtime", "", ""))


def _with_model_query(url: str, model: str) -> str:
    parsed = urlsplit(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "model"
    ]
    if model:
        query.append(("model", model))
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )
