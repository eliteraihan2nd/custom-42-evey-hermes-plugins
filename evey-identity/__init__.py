"""Evey Identity Plugin — self-updating personality.

Inspired by OpenClaw's diary pattern. Evey can update her own identity
based on what she learns. Adds a "Learned Behaviors" section to SOUL.md
that grows organically from experience.

Runs during self-improve cron (3am) or on-demand.
Uses local model (qwen35-4b) for reflection — $0 cost.
"""

import json
import os
import time
from pathlib import Path

import importlib.util as _iu, os as _os


def _load_evey_utils():
    """Load the shared evey_utils module.

    Sibling first: a plugin dir that carries its own copy is self-contained.
    Parent second: the whole-repo layout, where evey_utils.py sits one level up.
    """
    for _cand in (
        _os.path.join(_os.path.dirname(__file__), "evey_utils.py"),
        _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "evey_utils.py"),
    ):
        if _os.path.exists(_cand):
            _spec = _iu.spec_from_file_location("evey_utils", _cand)
            _mod = _iu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            return _mod
    raise ImportError("evey_utils.py not found next to or above this plugin")


_eu = _load_evey_utils()
call_llm = _eu.call_llm

SOUL_PATH = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "SOUL.md"
REFLECT_MODEL = _eu.resolve_model("critique", default="qwen35-4b")
MAX_LEARNED = 10  # Max learned behaviors to keep

SCHEMA = {
    "name": "update_identity",
    "description": (
        "Reflect on recent experiences and update your identity. "
        "Adds learned behaviors to SOUL.md based on what worked and what didn't. "
        "Use after completing a significant task, learning something new, "
        "or when your self-improve cron runs at 3am."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reflection": {
                "type": "string",
                "description": "What did you learn? What worked? What should you do differently?",
            },
            "behavior": {
                "type": "string",
                "description": "A new behavior rule to add (e.g., 'Always validate delegation results before reporting')",
            },
        },
        "required": ["reflection"],
    },
}

REFLECT_PROMPT = """Based on this reflection from an AI agent, extract ONE concise behavioral rule (max 15 words).

Reflection: {reflection}

Rules:
- Start with a verb (Always, Never, Prefer, Check, Use, etc.)
- Be specific and actionable
- Focus on what to DO, not what happened

Behavioral rule:"""


def _extract_rule(reflection):
    """Use cheap local model to distill reflection into a rule."""
    return call_llm(REFLECT_MODEL, REFLECT_PROMPT.format(reflection=reflection), max_tokens=50, temperature=0.3)


def handler(args, **kwargs):
    try:
        reflection = args.get("reflection", "")
        explicit_behavior = args.get("behavior", "")

        # Get or extract the behavior rule
        if explicit_behavior:
            rule = explicit_behavior
        else:
            rule = _extract_rule(reflection)
            if not rule:
                return json.dumps({"status": "skipped", "reason": "Could not extract rule"})

        # Read current SOUL.md
        if not SOUL_PATH.exists():
            return json.dumps({"error": "SOUL.md not found"})

        content = SOUL_PATH.read_text()

        # Add or update Learned Behaviors section
        marker = "## Learned Behaviors"
        date = time.strftime("%Y-%m-%d")

        if marker in content:
            # Extract existing behaviors
            parts = content.split(marker)
            before = parts[0]
            behaviors_text = parts[1] if len(parts) > 1 else ""
            behaviors = [l.strip() for l in behaviors_text.strip().split("\n") if l.strip().startswith("- ")]
            # Add new behavior
            behaviors.append(f"- {rule} ({date})")
            # Keep only the latest MAX_LEARNED
            behaviors = behaviors[-MAX_LEARNED:]
            content = before + marker + "\n" + "\n".join(behaviors) + "\n"
        else:
            # Create new section
            content += f"\n{marker}\n- {rule} ({date})\n"

        SOUL_PATH.write_text(content)

        return json.dumps({
            "status": "updated",
            "rule_added": rule,
            "reflection": reflection[:200],
            "total_behaviors": len([l for l in content.split("\n") if l.strip().startswith("- ") and "Learned" in content.split(l)[0]]),
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


def register(ctx):
    ctx.register_tool(name="update_identity", toolset="evey_identity",
        schema=SCHEMA, handler=handler)
