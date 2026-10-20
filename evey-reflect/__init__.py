"""Evey Reflect Plugin — Reflexion pattern for self-correction.

Before sending important outputs, run them through a critique loop:
Generate -> Critique -> Correct (max 3 iterations).
Uses cheap local model for critique to keep costs at $0.
"""

import json
import os

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

# Upstream pinned "qwen35-4b", a local model that no longer exists on the host
# gateway. The "critique" role resolves through the model map instead — the
# whole candidate LIST, so a dead provider falls through to the next route.
CRITIQUE_MODELS = _eu.resolve_models("critique", default=["qwen35-4b"])
MAX_ITERATIONS = 3

SCHEMA = {
    "name": "reflect_on_output",
    "description": (
        "Self-correct your output before sending it. Pass your draft response "
        "and the original task. The tool will critique it using a cheap model "
        "and suggest improvements. Use this for important outputs: research "
        "summaries, daily reports, delegation results, goal reviews. "
        "Returns: improved output + critique notes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The original task or question",
            },
            "draft": {
                "type": "string",
                "description": "Your draft response to critique",
            },
            "criteria": {
                "type": "string",
                "description": "What to check for (e.g., 'accuracy, completeness, actionability')",
            },
        },
        "required": ["task", "draft"],
    },
}

CRITIQUE_PROMPT = """You are a quality reviewer. Critique this draft response.

ORIGINAL TASK: {task}

DRAFT RESPONSE: {draft}

CRITERIA: {criteria}

Review for:
1. Factual accuracy — are claims verifiable?
2. Completeness — does it answer the full question?
3. Actionability — can the reader act on this?
4. Conciseness — any unnecessary fluff?

If the draft is GOOD (score >= 7/10), respond with: PASS: [brief note]
If it needs improvement, respond with: FIX: [specific issues to fix]

Keep your critique under 100 words."""


def _critique(prompt):
    # Upstream budget was 200 tokens; on a reasoning-model target that budget is consumed
    # by reasoning (finish_reason=length, critique cut mid-sentence). 400 returns the
    # complete verdict.
    result = _eu.call_first_available(CRITIQUE_MODELS, prompt, max_tokens=400, temperature=0.3)
    return (result or {}).get("content") or "PASS: Critique unavailable"


def handler(args, **kwargs):
    try:
        task = args.get("task", "")
        draft = args.get("draft", "")
        criteria = args.get("criteria", "accuracy, completeness, actionability")

        prompt = CRITIQUE_PROMPT.format(task=task, draft=draft[:1500], criteria=criteria)
        critique = _critique(prompt)

        passed = critique.upper().startswith("PASS")

        return json.dumps({
            "status": "pass" if passed else "needs_improvement",
            "critique": critique,
            "original_draft": draft,
            "suggestion": "Output looks good — send it." if passed else "Consider revising based on the critique above before sending.",
        })

    except Exception as e:
        return json.dumps({"error": str(e), "status": "pass", "critique": "Critique failed — defaulting to pass"})


def register(ctx):
    ctx.register_tool(
        name="reflect_on_output",
        toolset="evey_reflect",
        schema=SCHEMA,
        handler=handler,
    )
