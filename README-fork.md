# custom-42-evey-hermes-plugins

A maintained fork of [`42-evey/hermes-plugins`](https://github.com/42-evey/hermes-plugins) for
this deployment's gateway, plus the plugins that were not yet adopted.

Upstream is tracked as the `upstream` remote; this repo is the source of truth for what
runs on this host.

```
git remote -v
# origin    git@github.com:eliteraihan2nd/custom-42-evey-hermes-plugins.git
# upstream  https://github.com/42-evey/hermes-plugins.git
```

## Install

Whole-repo install. Hermes discovers the per-plugin directories inside the repo, so a
single install gives you all of them and keeps `evey_utils.py` (the shared module every
plugin loads) next to them:

```bash
hermes plugins install git@github.com:eliteraihan2nd/custom-42-evey-hermes-plugins.git
hermes plugins enable evey-council evey-delegate-model evey-reflect evey-validate
```

A per-plugin install (`...#evey-council`) also works; in that layout copy
`evey_utils.py` to the Hermes plugins directory, or the plugin falls back to the
parent-directory lookup.

## What this fork changes, and why

### 1. Model map — the reason per-task routing was dead

Upstream hardcodes free-cloud aliases (`mimo-v2-pro`, `qwen-coder-free`,
`llama70b-free`, `qwen35-4b`) that no longer exist on current gateways. The obvious
replacement — the host's own `combo-hermes-*` aliases — is a **rotating pool**: on
2026-10-19 all four were served by `deepseek/deepseek-v4.1-flash`, on 2026-10-20 all
four were served by `minimax/minimax-m3.1-flash:free`. Pointing every role at one
alias silently collapses "smart routing" into *one model, asked N times*.

`evey_utils.py` now carries a role→model map:

| role | model | upstream served |
|---|---|---|
| `code` | `combo/fast-mini` | `cohere/north-mini-code:free` |
| `reasoning` | `kc/nvidia/nemotron-3-ultra-550b-a55b:free` | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| `research` / `summary` | `th/mimo-v2.6-flash:free` | `mimo-v2.6-flash:free` |
| `analysis` | `openrouter/poolside/laguna-s-2.1:free` | `poolside/laguna-s-2.1:free` |
| `creative` | `openrouter/dots-studio/dots-3-note-preview:free` | `dots-studio/dots-3-note-preview:free` |
| `vision` | `kc/nvidia/nemotron-3.5-super-vl:free` | `nvidia/nemotron-3.5-super-vl:free` |
| `uncensored` | `kc/reflection/beam-free` | `reflection/beam-free` |
| `delegate` | `kc/minimax/minimax-m3.1-flash:free` | `minimax/minimax-m3.1-flash:free` |
| `critique` / `validate` | `no-think/th/mimo-v2.6-flash:free` | `mimo-v2.6-flash:free` |
| `judge` | `kc/nvidia/nemotron-3-ultra-550b-a55b:free` | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| `council` | the three members above | three distinct upstreams |

Every id was verified by reading the `model` field of the *response*, which is the
only way to know which upstream actually served the request. `no-think/` is a gateway
routing prefix that suppresses a reasoning model's thinking budget; it is accepted at
call time even though `/v1/models` does not list it, and it is what stops short
structured answers (validate/reflect verdicts) from being truncated.

**Override per host without editing code** — Hermes config:

```yaml
plugins:
  evey-model-map:
    code: ["my/model"]
    council: ["a/one", "b/two", "c/three"]
```

or a JSON file: `EEVEY_MODEL_MAP=/path/to/model-map.json`.

### 2. Reasoning-token budgets

`evey-validate` (100 tokens) and `evey-reflect` (200 tokens) had budgets too small for
a reasoning model: the budget was consumed by reasoning, the response came back
`finish_reason=length` with no verdict, and the plugin **reported success anyway**
(validate silently fell back to score 5 / CAUTION for correct input). Both now use
400 tokens.

### 3. Shared module loading

`evey-cache`, `evey-digest` and `evey-email-guard` used a bare
`from evey_utils import ...`, which assumes the plugins directory is on `sys.path`.
It is not, so `evey-cache` died with `ModuleNotFoundError: No module named
'evey_utils'`. All plugins now load the shared module by path, and the loader prefers a
sibling copy (self-contained plugin dir) before the parent directory (whole-repo
layout).

### 4. `~/.hermes` hardcoded paths

Paths are resolved from `HERMES_HOME`, so a profile's plugins write into that
profile's home instead of the default one. `evey-telemetry` resolves its directory per
call rather than at import time, which is what let its path drift away from the home
the process actually runs as.

### 5. Gateway hygiene

`evey_utils.py` sends a browser `User-Agent` and prefers `EEVEY_LITELLM_URL` over
`OPENAI_BASE_URL` (Hermes reads the latter as its own custom endpoint). The public
gateway zone bans urllib's default UA with Cloudflare 1010.

## Not adopted here

Kept out of this deployment for contract reasons, unchanged from upstream:
`evey-commands` (no manifest), `evey-identity` (self-edits `SOUL.md`; changes to SOUL
are propose-only), `evey-status` (expects a dashboard endpoint this host does not
expose), `evey-memory-adaptive` and `evey-memory-consolidate` (second writer to a
memory store that already has an owner).

## License

Upstream is MIT (Copyright (c) 2026 Evey) — see `LICENSE`.
