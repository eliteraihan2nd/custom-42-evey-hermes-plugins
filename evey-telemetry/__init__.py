"""Evey Telemetry Plugin — Structured logging and observability.

Emits structured JSON events for every tool call, delegation, error,
and key agent action. Events go to a rotating log file at
<this process's hermes home>/telemetry/events.jsonl for dashboard
consumption (resolved at call time, see ``_telemetry_dir``).

Provides a telemetry_query tool for Evey to inspect her own metrics.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("evey.telemetry")


def _telemetry_dir() -> Path:
    """Telemetry dir of the Hermes home THIS process runs as (D-B).

    Resolved per call — an import-time constant is what let the path drift
    silently away from the home the process actually runs as.
    """
    try:
        from hermes_constants import get_hermes_home  # in-process plugin: the module is importable
        return Path(get_hermes_home()) / "telemetry"
    except Exception:  # standalone/unhardened import
        return Path(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")) / "telemetry"


def _events_file() -> Path:
    return _telemetry_dir() / "events.jsonl"


MAX_EVENTS_FILE_SIZE = 10 * 1024 * 1024  # 10MB before rotation

# In-memory metrics for the current session
SESSION_METRICS = {
    "session_id": str(uuid.uuid4())[:8],
    "start_time": datetime.now(timezone.utc).isoformat(),
    "tool_calls": 0,
    "delegations": 0,
    "errors": 0,
    "total_tokens": 0,
    "events_emitted": 0,
}


def _ensure_dir():
    _telemetry_dir().mkdir(parents=True, exist_ok=True)


def _rotate_if_needed():
    """Rotate events file if it exceeds max size."""
    events_file = _events_file()
    if events_file.exists() and events_file.stat().st_size > MAX_EVENTS_FILE_SIZE:
        rotated = events_file.parent / f"events.{int(time.time())}.jsonl"
        events_file.rename(rotated)
        # Keep only last 5 rotated files
        rotated_files = sorted(events_file.parent.glob("events.*.jsonl"))
        for old in rotated_files[:-5]:
            old.unlink(missing_ok=True)


def emit_event(event_type, data=None):
    """Write a structured event to the telemetry log."""
    try:
        _ensure_dir()
        _rotate_if_needed()
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "sid": SESSION_METRICS["session_id"],
            "type": event_type,
            **(data or {}),
        }
        with open(_events_file(), "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
        SESSION_METRICS["events_emitted"] += 1
    except Exception as e:
        logger.debug(f"Telemetry emit failed: {e}")


def track_tool_call(tool_name, duration_ms=0, success=True, tokens=0, error=None):
    """Track a tool call event."""
    SESSION_METRICS["tool_calls"] += 1
    if tokens:
        SESSION_METRICS["total_tokens"] += tokens
    if not success:
        SESSION_METRICS["errors"] += 1
    emit_event("tool_call", {
        "tool": tool_name,
        "duration_ms": duration_ms,
        "success": success,
        "tokens": tokens,
        "error": str(error)[:200] if error else None,
    })


def _on_post_tool_call(tool_name="", duration_ms=0, status=None, error_message=None, **kwargs):
    """Hermes ``post_tool_call`` adapter.

    The hook payload carries the tool name, its duration and an ``ok``/``error``
    status; tokens are not reported there, so they stay 0. ``**kwargs`` is required
    by the plugin contract (forward compatibility). Any failure here must degrade
    to a no-op — telemetry never breaks a tool call.
    """
    try:
        track_tool_call(
            tool_name or "unknown",
            duration_ms=duration_ms or 0,
            success=status != "error",
            error=error_message,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"post_tool_call hook failed: {e}")


def track_delegation(model, task_type, duration_ms=0, success=True, tokens=0, error=None):
    """Track a delegation event."""
    SESSION_METRICS["delegations"] += 1
    if tokens:
        SESSION_METRICS["total_tokens"] += tokens
    if not success:
        SESSION_METRICS["errors"] += 1
    emit_event("delegation", {
        "model": model,
        "task_type": task_type,
        "duration_ms": duration_ms,
        "success": success,
        "tokens": tokens,
        "error": str(error)[:200] if error else None,
    })


def track_error(source, error_msg, severity="error"):
    """Track an error event."""
    SESSION_METRICS["errors"] += 1
    emit_event("error", {
        "source": source,
        "message": str(error_msg)[:500],
        "severity": severity,
    })


def track_cron(job_name, status, duration_ms=0, output_preview=None):
    """Track a cron job execution."""
    emit_event("cron", {
        "job": job_name,
        "status": status,
        "duration_ms": duration_ms,
        "output_preview": str(output_preview)[:200] if output_preview else None,
    })


# ---- Tool: telemetry_query ----

QUERY_SCHEMA = {
    "name": "telemetry_query",
    "description": (
        "Query Evey's telemetry data. View recent events, session metrics, "
        "error rates, delegation performance, and tool usage patterns. "
        "Use this to understand your own performance and identify issues."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["session_metrics", "recent_errors", "recent_events", "delegation_stats", "tool_stats"],
                "description": "What to query",
            },
            "limit": {
                "type": "number",
                "description": "Max results to return (default: 20)",
            },
        },
        "required": ["query_type"],
    },
}


def _read_recent_events(limit=50, event_type=None):
    """Read recent events from this process's log file."""
    events_file = _events_file()
    if not events_file.exists():
        return []
    events = []
    try:
        with open(events_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    if event_type is None or ev.get("type") == event_type:
                        events.append(ev)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return events[-limit:]


def _query_impl(args, **kwargs):
    query_type = args.get("query_type", "session_metrics")
    limit = int(args.get("limit", 20))

    if query_type == "session_metrics":
        metrics = dict(SESSION_METRICS)
        metrics["uptime_seconds"] = int(
            (datetime.now(timezone.utc) - datetime.fromisoformat(metrics["start_time"])).total_seconds()
        )
        return json.dumps({"status": "ok", "metrics": metrics})

    elif query_type == "recent_errors":
        errors = _read_recent_events(limit, "error")
        return json.dumps({"status": "ok", "count": len(errors), "errors": errors})

    elif query_type == "recent_events":
        events = _read_recent_events(limit)
        return json.dumps({"status": "ok", "count": len(events), "events": events})

    elif query_type == "delegation_stats":
        delegations = _read_recent_events(200, "delegation")
        if not delegations:
            return json.dumps({"status": "ok", "message": "No delegation events yet"})
        models = {}
        for d in delegations:
            m = d.get("model", "unknown")
            if m not in models:
                models[m] = {"calls": 0, "successes": 0, "total_tokens": 0, "total_ms": 0}
            models[m]["calls"] += 1
            if d.get("success"):
                models[m]["successes"] += 1
            models[m]["total_tokens"] += d.get("tokens", 0)
            models[m]["total_ms"] += d.get("duration_ms", 0)
        for m in models:
            c = models[m]["calls"]
            models[m]["success_rate"] = f"{models[m]['successes']/c*100:.0f}%" if c else "0%"
            models[m]["avg_ms"] = int(models[m]["total_ms"] / c) if c else 0
        return json.dumps({"status": "ok", "models": models, "total_delegations": len(delegations)})

    elif query_type == "tool_stats":
        tool_calls = _read_recent_events(500, "tool_call")
        if not tool_calls:
            return json.dumps({"status": "ok", "message": "No tool call events yet"})
        tools = {}
        for tc in tool_calls:
            t = tc.get("tool", "unknown")
            if t not in tools:
                tools[t] = {"calls": 0, "errors": 0, "total_ms": 0}
            tools[t]["calls"] += 1
            if not tc.get("success"):
                tools[t]["errors"] += 1
            tools[t]["total_ms"] += tc.get("duration_ms", 0)
        for t in tools:
            c = tools[t]["calls"]
            tools[t]["error_rate"] = f"{tools[t]['errors']/c*100:.0f}%" if c else "0%"
            tools[t]["avg_ms"] = int(tools[t]["total_ms"] / c) if c else 0
        return json.dumps({"status": "ok", "tools": tools, "total_calls": len(tool_calls)})

    return json.dumps({"status": "error", "message": f"Unknown query type: {query_type}"})


def query_handler(args, **kwargs):
    """``_query_impl`` plus the provenance of the answer.

    ``path``/``home`` name the file that was read and ``sid`` this session's id,
    so a near-empty answer is explainable instead of looking like "no events".
    """
    result = json.loads(_query_impl(args, **kwargs))
    result["path"] = str(_events_file())
    result["home"] = str(_telemetry_dir().parent)
    result["sid"] = SESSION_METRICS["session_id"]
    return json.dumps(result)


def register(ctx):
    _ensure_dir()
    emit_event("plugin_loaded", {"plugin": "evey-telemetry", "version": "1.0.0"})
    ctx.register_tool(
        name="telemetry_query",
        toolset="evey_telemetry",
        schema=QUERY_SCHEMA,
        handler=query_handler,
    )
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    logger.info("evey-telemetry plugin loaded — structured logging active")
