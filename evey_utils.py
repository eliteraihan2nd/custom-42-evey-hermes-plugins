"""Shared utilities for Evey plugins — LLM calls, HTTP helpers, retry logic, model routing.

All plugins should use these instead of rolling their own urllib code.
- call_llm(): Quick LLM call, returns string or None
- call_model(): Full LLM call with usage info, retries, reasoning recovery
- http_get(): GET with error handling
- http_post_json(): POST JSON with error handling
- resolve_model()/resolve_models(): role → model id, from the host model map
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error

logger = logging.getLogger("evey.utils")

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# EEVEY_LITELLM_URL wins over OPENAI_BASE_URL: Hermes reads OPENAI_BASE_URL as *its own*
# custom endpoint, so pointing the plugins at the same origin is the safe default, but an
# explicit EEVEY_LITELLM_URL lets the plugins target a different gateway.
LITELLM_URL = os.environ.get("EEVEY_LITELLM_URL") or os.environ.get("OPENAI_BASE_URL", "")
LITELLM_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Model map
# ---------------------------------------------------------------------------
# WHY THIS EXISTS: upstream hardcodes free-cloud aliases ("mimo-v2-pro",
# "qwen-coder-free", "llama70b-free") that no longer exist on the gateway, and
# the obvious replacement — the host's own "combo-hermes-*" aliases — is a
# ROTATING POOL: on 2026-10-19 all four of them were served by
# "deepseek/deepseek-v4.1-flash", on 2026-10-20 all four were served by
# "minimax/minimax-m3.1-flash:free". Pointing every role at one alias therefore
# silently collapses per-task routing into "one model, asked N times".
#
# The map below pins each role to a model id whose upstream is DISTINCT and
# verified to answer, so "code" and "reasoning" are genuinely different models.
# Override per host without editing code:
#   1. Hermes config:  plugins.evey-model-map: {code: [my-model]}
#   2. Env var:        EEVEY_MODEL_MAP=/path/to/model-map.json
# A host may pin a role to a *different* upstream by using the same
# "provider/model" ids OmniRoute exposes (e.g. "th/mimo-v2.6-flash:free").
#
# "no-think/" is an OmniRoute routing prefix that disables a reasoning model's
# thinking budget. It is not listed by /v1/models but is accepted at call time;
# it is what keeps short structured answers (validate/reflect verdicts) from
# being truncated by reasoning tokens.
DEFAULT_MODEL_MAP = {
    # role → ordered candidate list (first is preferred, rest are fallbacks)
    "code": ["combo/fast-mini"],
    "reasoning": ["kc/nvidia/nemotron-3-ultra-550b-a55b:free"],
    "research": ["th/mimo-v2.6-flash:free"],
    "summary": ["th/mimo-v2.6-flash:free"],
    "analysis": ["openrouter/poolside/laguna-s-2.1:free"],
    "creative": ["openrouter/dots-studio/dots-3-note-preview:free"],
    "vision": ["kc/nvidia/nemotron-3.5-super-vl:free"],
    "uncensored": ["kc/reflection/beam-free"],
    "delegate": ["kc/minimax/minimax-m3.1-flash:free"],
    "critique": ["no-think/th/mimo-v2.6-flash:free"],
    "validate": ["no-think/th/mimo-v2.6-flash:free"],
    "judge": ["kc/nvidia/nemotron-3-ultra-550b-a55b:free"],
    # Three members, three different upstreams — the point of a council.
    "council": [
        "th/mimo-v2.6-flash:free",
        "kc/reflection/beam-free",
        "openrouter/poolside/laguna-s-2.1:free",
    ],
}


def _model_map_file():
    """Optional JSON model map path from the environment."""
    path = os.environ.get("EEVEY_MODEL_MAP", "").strip()
    return path or None


def _host_model_map():
    """Model map overrides from the host, if any.

    Order: Hermes config (``plugins.evey-model-map``) then a JSON file pointed
    at by ``EEVEY_MODEL_MAP``. A host that does not run Hermes simply gets an
    empty override and the built-in map stands.
    """
    overrides = {}
    try:
        from hermes_cli.config import load_config_readonly, cfg_get

        raw = cfg_get(load_config_readonly(), "plugins", "evey-model-map", default={}) or {}
        if isinstance(raw, dict):
            overrides.update(raw)
    except Exception:
        pass

    path = _model_map_file()
    if path:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                overrides.update(data)
        except Exception as exc:
            logger.warning("EEVEY_MODEL_MAP=%s unreadable (%s); using built-in map", path, exc)
    return overrides


def model_map():
    """Effective model map: built-in defaults with host overrides applied per role."""
    effective = {role: list(models) for role, models in DEFAULT_MODEL_MAP.items()}
    for role, models in _host_model_map().items():
        if isinstance(models, str):
            models = [models]
        if isinstance(models, (list, tuple)) and models:
            effective[role] = [str(m) for m in models]
    return effective


def resolve_models(role, default=None):
    """Ordered candidate models for a role. Unknown role → *default* (or empty)."""
    models = model_map().get(role)
    if models:
        return list(models)
    if default:
        return [default] if isinstance(default, str) else list(default)
    return []


def resolve_model(role, default=None):
    """Preferred model for a role, or *default* when the role is unknown."""
    models = resolve_models(role, default)
    return models[0] if models else (default if isinstance(default, str) else None)


def call_llm(model, prompt, max_tokens=200, temperature=0.3, retries=2):
    """Quick LLM call. Returns content string or None on failure."""
    result = call_model(model, prompt, max_tokens=max_tokens, temperature=temperature, retries=retries)
    return result.get("content") if result else None


def call_model(model, prompt, max_tokens=2000, temperature=0.7, retries=2, timeout=60):
    """Full LLM call via LiteLLM with retry, reasoning recovery, and usage tracking.

    Returns dict: {"content": str, "tokens": int, "model": str, "attempts": int}
    Returns None on total failure.
    """
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    for attempt in range(1, retries + 2):
        try:
            req = urllib.request.Request(
                f"{LITELLM_URL}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {LITELLM_KEY}",
                    "User-Agent": _UA,
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())

            msg = result["choices"][0]["message"]
            content = msg.get("content", "") or ""
            usage = result.get("usage", {})

            # Reasoning recovery — extract from think-only responses
            if not content.strip():
                reasoning = (
                    msg.get("reasoning_content")
                    or msg.get("reasoning")
                    or (msg.get("provider_specific_fields") or {}).get("reasoning_content")
                )
                if reasoning and str(reasoning).strip():
                    content = str(reasoning).strip()

            if not content.strip():
                raise ValueError("Empty response from model")

            return {
                "content": content,
                "tokens": usage.get("total_tokens", 0),
                "model": result.get("model", model),
                "attempts": attempt,
            }
        except Exception as exc:
            if attempt > retries:
                logger.warning("Model call failed after %d attempts: %s", attempt, exc)
                return None
            time.sleep(2 ** attempt)
    return None


def http_get(url, timeout=10):
    """HTTP GET with error handling. Returns response text or None."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode()
    except Exception:
        return None


def http_get_json(url, timeout=10):
    """HTTP GET returning parsed JSON dict or None."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def http_post_json(url, payload, timeout=10):
    """HTTP POST JSON with error handling. Returns parsed JSON dict or None."""
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": _UA},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None
