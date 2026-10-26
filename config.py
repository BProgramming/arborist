#!/usr/bin/env python3
"""Load and validate arborist.toml, and hand values to the shell launcher.

Usage (called by arborist.sh):
    config.py --shell        print `KEY=value` lines (shell-quoted) to eval
    config.py --server-args  print llama-server arguments, one per line

chat.py imports load() directly.
"""
from __future__ import annotations

import os
import shlex
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULTS: dict = {
    "paths": {"root": "~/.local/share/arborist"},
    "build": {
        "backend": "vulkan",
        "repo": "https://github.com/ggml-org/llama.cpp.git",
        "ref": "master",
        # AMD GPU architecture for the HIP backend (e.g. "gfx1201"); required
        # only when backend = "hip".
        "hip_target": "",
        "native": True,
        "jobs": 0,
    },
    # No default model: the toml has to say what to download and run.
    "model": {
        "name": "",
        "repo": "",
        "file": "",
        "vision": False,
        "mmproj_file": "",
        # Reasoning efforts the model's chat template accepts. Empty = the
        # model has no effort control, and none is sent.
        "efforts": [],
    },
    # Optional llama-server flags that only make sense for this model, as one
    # shell-style string. {efforts} and {effort} are filled in (see the toml).
    "options": {
        "server_args": "",
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8091,
        "gpu_layers": 99,
        "ctx_size": 65536,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "flash_attn": "on",
        "parallel": 1,
        "batch_size": 2048,
        "ubatch_size": 512,
        "threads": 0,
        "startup_timeout": 300,
        "extra_args": [],
    },
    "chat": {
        "system_prompt": "You are a helpful assistant.",
        "reasoning_effort": "",
        "show_thinking": False,
        "max_tokens": 16384,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.05,
        "presence_penalty": 0.0,
        "show_stats": True,
        # Environment variables shown to the model in its system prompt (only
        # the ones that are set). Names only; never put secrets here.
        "env_vars": ["VIRTUAL_ENV", "CONDA_DEFAULT_ENV"],
    },
    "tools": {
        "enabled": True,
        "approve_edits": "ask",
        "max_steps": 25,
        "max_read_lines": 400,
        "max_output_chars": 20000,
        "max_list_entries": 300,
        "max_grep_matches": 100,
        "max_tree_depth": 4,
        "run_timeout": 120,
        "test_timeout": 300,
        "max_consecutive_errors": 4,
        "max_repeat_calls": 3,
        "checkpoints": True,
        "checkpoint_days": 14,
        "web_enabled": True,
        "web_allowed_hosts": [],
        "web_page_chars": 12000,
        "web_max_pages": 25,
        "web_max_depth": 2,
        "web_timeout": 15,
    },
    "compaction": {
        "enabled": True,
        "threshold": 0.7,
        "target": 0.5,
        "keep_turns": 2,
        "keep_tool_results": 8,
        # Effort for writing summaries; "" = whatever the chat is using.
        "summary_effort": "",
    },
}

BACKENDS = {"vulkan", "hip", "cpu"}
KV_TYPES = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
FA_MODES = {"on", "off", "auto"}
EDIT_APPROVALS = {"ask", "auto"}


class ConfigError(Exception):
    pass


def _merge(base: dict, override: dict) -> dict:
    out = {}
    for section, defaults in base.items():
        user = override.get(section, {})
        if not isinstance(user, dict):
            raise ConfigError(f"[{section}] must be a table")
        unknown = set(user) - set(defaults)
        if unknown:
            raise ConfigError(f"unknown key(s) in [{section}]: {', '.join(sorted(unknown))}")
        out[section] = {**defaults, **user}
    unknown_sections = set(override) - set(base)
    if unknown_sections:
        raise ConfigError(f"unknown section(s): {', '.join(sorted(unknown_sections))}")
    return out


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _validate(c: dict) -> None:
    b, m, s, ch = c["build"], c["model"], c["server"], c["chat"]
    _check(b["backend"] in BACKENDS, f"build.backend must be one of {sorted(BACKENDS)}")
    if b["backend"] == "hip":
        _check(isinstance(b["hip_target"], str) and b["hip_target"].startswith("gfx"),
               'build.backend = "hip" needs build.hip_target, e.g. "gfx1201" (see rocminfo)')
    for key in ("name", "repo", "file", "mmproj_file"):
        _check(isinstance(m[key], str), f"model.{key} must be a string")
    _check(m["repo"].count("/") == 1, 'model.repo must be a Hugging Face repo id, e.g. "owner/name-gguf"')
    _check(m["file"].endswith(".gguf") and "/" not in m["file"],
           'model.file must be a .gguf file name in model.repo, e.g. "Model-Q4_K_M.gguf"')
    _check(isinstance(m["vision"], bool), "model.vision must be true or false")
    if m["vision"]:
        _check(m["mmproj_file"].endswith(".gguf") and "/" not in m["mmproj_file"],
               "model.vision = true needs model.mmproj_file (a .gguf file name in model.repo)")
    efforts = m["efforts"]
    _check(isinstance(efforts, list) and all(isinstance(e, str) and e and "," not in e for e in efforts),
           'model.efforts must be a list of effort names, e.g. ["low", "medium", "high"]')
    _check(len(set(efforts)) == len(efforts), "model.efforts has duplicates")
    if efforts:
        _check(ch["reasoning_effort"] in efforts,
               f"chat.reasoning_effort must be one of model.efforts {efforts}")
    _check(isinstance(c["options"]["server_args"], str), "options.server_args must be a string")
    model_options(c)  # raises ConfigError on a bad string or placeholder
    _check(s["cache_type_k"] in KV_TYPES and s["cache_type_v"] in KV_TYPES,
           f"server.cache_type_k/v must be one of {sorted(KV_TYPES)}")
    _check(s["flash_attn"] in FA_MODES, f"server.flash_attn must be one of {sorted(FA_MODES)}")
    if s["cache_type_v"] not in ("f16", "f32", "bf16"):
        _check(s["flash_attn"] != "off", "a quantized V cache needs server.flash_attn = 'on' or 'auto'")
    _check(1 <= int(s["port"]) <= 65535, "server.port out of range")
    _check(0 < int(s["ctx_size"]) <= 262144, "server.ctx_size must be 1..262144")
    _check(int(s["parallel"]) >= 1, "server.parallel must be >= 1")
    _check(int(s["ubatch_size"]) <= int(s["batch_size"]), "server.ubatch_size must be <= batch_size")
    _check(isinstance(s["extra_args"], list) and all(isinstance(a, str) for a in s["extra_args"]),
           "server.extra_args must be a list of strings")
    _check(int(ch["max_tokens"]) > 0, "chat.max_tokens must be > 0")
    t = c["tools"]
    _check(t["approve_edits"] in EDIT_APPROVALS,
           f"tools.approve_edits must be one of {sorted(EDIT_APPROVALS)} (running code always asks)")
    for key in ("max_steps", "max_read_lines", "max_output_chars", "max_list_entries",
                "max_grep_matches", "max_tree_depth", "run_timeout", "test_timeout",
                "web_page_chars", "web_max_pages", "web_timeout"):
        _check(isinstance(t[key], int) and not isinstance(t[key], bool) and t[key] > 0,
               f"tools.{key} must be a positive integer")
    # Below this a run's log has no room next to its header and instructions.
    _check(t["max_output_chars"] >= 2000, "tools.max_output_chars must be at least 2000")
    _check(t["web_page_chars"] <= t["max_output_chars"],
           "tools.web_page_chars must not exceed tools.max_output_chars")
    _check(isinstance(t["web_max_depth"], int) and 0 <= t["web_max_depth"] <= 5,
           "tools.web_max_depth must be 0..5")
    _check(t["web_max_pages"] <= 200, "tools.web_max_pages must be at most 200")
    _check(isinstance(t["web_enabled"], bool), "tools.web_enabled must be true or false")
    _check(isinstance(t["web_allowed_hosts"], list) and all(
        isinstance(h, str) and h and "/" not in h for h in t["web_allowed_hosts"]),
        "tools.web_allowed_hosts must be a list of host names, e.g. [\"docs.python.org\"]")
    for key in ("max_consecutive_errors", "max_repeat_calls"):
        _check(isinstance(t[key], int) and not isinstance(t[key], bool) and t[key] >= 0,
               f"tools.{key} must be a whole number (0 turns the check off)")
    _check(isinstance(t["checkpoints"], bool), "tools.checkpoints must be true or false")
    _check(isinstance(t["checkpoint_days"], int) and not isinstance(t["checkpoint_days"], bool)
           and t["checkpoint_days"] >= 0, "tools.checkpoint_days must be a whole number (0 keeps them forever)")
    _check(isinstance(ch["env_vars"], list) and all(
        isinstance(v, str) and v.replace("_", "").isalnum() for v in ch["env_vars"]),
        'chat.env_vars must be a list of variable names, e.g. ["VIRTUAL_ENV"]')
    k = c["compaction"]
    _check(isinstance(k["enabled"], bool), "compaction.enabled must be true or false")
    for key in ("threshold", "target"):
        _check(isinstance(k[key], (int, float)) and not isinstance(k[key], bool) and 0 < k[key] < 1,
               f"compaction.{key} must be a fraction between 0 and 1, e.g. 0.7")
    _check(k["target"] < k["threshold"], "compaction.target must be below compaction.threshold")
    for key in ("keep_turns", "keep_tool_results"):
        _check(isinstance(k[key], int) and not isinstance(k[key], bool) and k[key] >= 1,
               f"compaction.{key} must be at least 1")
    _check(isinstance(k["summary_effort"], str), "compaction.summary_effort must be a string")
    if k["summary_effort"]:
        _check(k["summary_effort"] in efforts,
               f"compaction.summary_effort must be \"\" or one of model.efforts {efforts}")


def load(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("ARBORIST_CONFIG") or HERE / "arborist.toml")
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with open(path, "rb") as f:
        try:
            raw = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{path}: {e}") from e
    cfg = _merge(DEFAULTS, raw)
    _validate(cfg)

    # Derived values, so the shell and Python agree on every path.
    root = Path(os.path.expanduser(cfg["paths"]["root"])).resolve()
    m = cfg["model"]
    model_file = m["file"]
    cfg["derived"] = {
        "config_path": str(path.resolve()),
        "root": str(root),
        "src_dir": str(root / "llama.cpp"),
        "build_dir": str(root / "llama.cpp" / f"build-{cfg['build']['backend']}"),
        "server_bin": str(root / "llama.cpp" / f"build-{cfg['build']['backend']}" / "bin" / "llama-server"),
        # What the chat calls the model; the file name if the toml gives none.
        "model_name": m["name"].strip() or Path(model_file).stem,
        "model_dir": str(root / "models"),
        "model_file": model_file,
        "model_path": str(root / "models" / model_file),
        "model_url": f"https://huggingface.co/{m['repo']}/resolve/main/{model_file}",
        "mmproj_path": str(root / "models" / m["mmproj_file"]),
        "mmproj_url": f"https://huggingface.co/{m['repo']}/resolve/main/{m['mmproj_file']}",
        "log_dir": str(root / "logs"),
        "history_dir": str(root / "history"),
        "memory_path": str(root / "memory" / "memory.toml"),
        "checkpoint_dir": str(root / "checkpoints"),
        "base_url": f"http://{cfg['server']['host']}:{cfg['server']['port']}",
    }
    return cfg


def model_options(cfg: dict) -> list[str]:
    """options.server_args, split like a shell would, with {efforts}/{effort} filled in."""
    raw = cfg["options"]["server_args"]
    try:
        words = shlex.split(raw)
    except ValueError as e:
        raise ConfigError(f"options.server_args: {e}") from None
    values = {"efforts": ",".join(cfg["model"]["efforts"]),
              "effort": cfg["chat"]["reasoning_effort"]}
    out = []
    for w in words:
        try:
            w = w.format(**values)
        except KeyError as e:
            raise ConfigError(f"options.server_args: unknown placeholder {{{e.args[0]}}}; "
                              "use {efforts} or {effort}") from None
        except (ValueError, IndexError) as e:
            raise ConfigError(f"options.server_args: {e} (write {{{{ or }}}} for a literal brace)") from None
        if ("{efforts}" in raw and not values["efforts"]) or ("{effort}" in raw and not values["effort"]):
            raise ConfigError("options.server_args uses {efforts}/{effort} but model.efforts "
                              "or chat.reasoning_effort is empty")
        out.append(w)
    return out


def server_args(cfg: dict) -> list[str]:
    s, d = cfg["server"], cfg["derived"]
    args = [
        "--model", d["model_path"],
        "--host", s["host"],
        "--port", str(s["port"]),
        "--n-gpu-layers", str(s["gpu_layers"]),
        "--ctx-size", str(s["ctx_size"]),
        "--flash-attn", s["flash_attn"],
        "--cache-type-k", s["cache_type_k"],
        "--cache-type-v", s["cache_type_v"],
        "--parallel", str(s["parallel"]),
        "--batch-size", str(s["batch_size"]),
        "--ubatch-size", str(s["ubatch_size"]),
        "--jinja",
        # Thinking arrives as a separate `reasoning_content` stream, so the
        # client can hide or dim it.
        "--reasoning-format", "deepseek",
        "--no-webui",
        # The fork reports model loading (incl. "offloaded N/N layers to GPU",
        # which arborist.sh checks) at level 4; the default of 3 hides it. Adds a
        # few lines per request to the log, nothing per token.
        "--log-verbosity", "4",
        # Default CORS is '*', which lets any web page open in your browser
        # call this server. Only allow localhost pages.
        "--cors-origins", "localhost",
    ]
    if int(s["threads"]) > 0:
        args += ["--threads", str(s["threads"])]
    if cfg["model"]["vision"]:
        args += ["--mmproj", d["mmproj_path"]]
    args += model_options(cfg)
    args += s["extra_args"]
    return args


def shell_lines(cfg: dict) -> list[str]:
    b, d, s, m = cfg["build"], cfg["derived"], cfg["server"], cfg["model"]
    values = {
        "CFG_BACKEND": b["backend"],
        "CFG_REPO": b["repo"],
        "CFG_REF": b["ref"],
        "CFG_HIP_TARGET": b["hip_target"],
        "CFG_NATIVE": "ON" if b["native"] else "OFF",
        "CFG_JOBS": str(b["jobs"] or os.cpu_count() or 4),
        "CFG_VISION": "1" if m["vision"] else "0",
        "CFG_MODEL_REPO": m["repo"],
        "CFG_STARTUP_TIMEOUT": str(s["startup_timeout"]),
        **{f"CFG_{k.upper()}": str(v) for k, v in d.items()},
    }
    return [f"{k}={shlex.quote(v)}" for k, v in values.items()]


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("--shell", "--server-args"):
        print(__doc__, file=sys.stderr)
        return 2
    try:
        cfg = load()
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1
    lines = shell_lines(cfg) if argv[1] == "--shell" else server_args(cfg)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
