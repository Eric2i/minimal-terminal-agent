"""Model configuration from environment variables (optionally loaded from a .env file).

Nothing about the endpoint is hard-coded: MTA_BASE_URL and MTA_MODEL must be
set; MTA_API_KEY may be omitted for local servers. See .env.example.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mta.model import ModelConfig


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines; real environment variables win."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def model_config_from_env() -> ModelConfig:
    missing = [k for k in ("MTA_BASE_URL", "MTA_MODEL") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing {', '.join(missing)}; copy .env.example to .env and fill it in.")
    env = os.environ
    return ModelConfig(
        base_url=env["MTA_BASE_URL"],
        model=env["MTA_MODEL"],
        api_key=env.get("MTA_API_KEY") or "EMPTY",  # local servers often need no key
        temperature=float(env["MTA_TEMPERATURE"]) if env.get("MTA_TEMPERATURE") else None,
        max_tokens=int(env["MTA_MAX_TOKENS"]) if env.get("MTA_MAX_TOKENS") else None,
        logprobs=env.get("MTA_LOGPROBS", "").lower() in ("1", "true"),
        extra_body=json.loads(env["MTA_EXTRA_BODY"]) if env.get("MTA_EXTRA_BODY") else {},
    )
