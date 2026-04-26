import os
import re
from pathlib import Path

import yaml

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_config(config_path: str | Path) -> dict:
    """Load YAML config with environment variable substitution.

    Only substitutes explicit ${VAR_NAME} patterns, not arbitrary env vars.
    Unmatched patterns are left as-is.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = f.read()

    def _replace_env(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    raw = _ENV_VAR_PATTERN.sub(_replace_env, raw)

    return yaml.safe_load(raw)
