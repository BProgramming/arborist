# arborist

A terminal coding assistant that runs a local model on your own GPU. One script
builds [llama.cpp](https://github.com/ggml-org/llama.cpp), downloads a GGUF
model from Hugging Face, starts the server, and drops you into a chat where the
model can read, search and edit files in the current directory, run code and
tests, and look things up on the web, asking you before anything that changes
or runs.

This is a personal project, built over a few days for my own PC (Linux Mint,
AMD RX 9070 16 GB, running PrismML's Bonsai 2 27B). It isn't a product, and
nothing here is novel. It's here in case you want to fiddle with your own
version. Fork it, gut it, ship it — see [License](#license).

## Requirements

- Linux (the launcher is bash and the install hints are for Debian/Ubuntu/Mint)
- Python 3.11+ (standard library only; `pytest` for the tests)
- `git`, `cmake`, `curl`, a C++ compiler; `ninja` optional but faster
- A GPU backend: **Vulkan** (default; `libvulkan-dev glslc spirv-headers
  mesa-vulkan-drivers`) or **ROCm/HIP**. CPU works, slowly.
- Disk for the build plus the model (~10 GB for the default)

```sh
sudo apt install build-essential cmake ninja-build git curl \
  libvulkan-dev glslc spirv-headers mesa-vulkan-drivers
```

## Quick start

```sh
git clone https://github.com/BProgramming/arborist
cd ~/some/project            # the model's tools work on this folder
~/path/to/arborist/arborist.sh
```

The first run clones and builds llama.cpp, downloads and checksums the model,
starts the server and opens the chat. Later runs reuse all of that. When you
leave the chat, a server the script started is stopped.

| Command | What it does |
|---|---|
| `./arborist.sh` | set up anything missing, start the server, open a chat |
| `./arborist.sh --workspace DIR` | same, with the tools working on `DIR` |
| `./arborist.sh --no-tools` | plain chat; the model gets no file, run or web tools |
| `./arborist.sh setup` | build and download only |
| `./arborist.sh server` | run the server in the foreground for other clients (OpenAI-compatible API) |
| `./arborist.sh update` | pull the configured llama.cpp ref and rebuild |
| `./arborist.sh rebuild` | rebuild from the current source |
| `./arborist.sh status` | what's built, downloaded and running |
| `./arborist.sh stop` | stop a server the script left running |

## Configuration

Everything is in [`arborist.toml`](arborist.toml), commented line by line;
lines tagged `PERF` affect speed or memory. Point `ARBORIST_CONFIG` at another
file to keep your own copy outside the repo.

**The shipped toml is tuned for my machine.** For anything else, look at:

- `build.backend`: `vulkan`, `hip` or `cpu`. With `hip`, set
  `build.hip_target` to your GPU's architecture (`rocminfo` shows it).
- `server.ctx_size` and `server.gpu_layers`: the main VRAM levers. After
  loading, the launcher prints how many layers made it onto the GPU, and warns
  if some didn't.

**Another model:** set `model.repo` and `model.file` to any GGUF on Hugging
Face. Then set `model.efforts` to the reasoning levels its chat template
accepts (or `[]`), clear `options.server_args`, and set `build.repo`/`ref` to
stock llama.cpp. The Bonsai-specific values (the PrismML fork, the effort list,
the sampling settings) exist only because Bonsai 2 needs them. The code
defaults are stock llama.cpp `master`.

Everything arborist keeps lives under `paths.root` (default
`~/.local/share/arborist`): the llama.cpp source and build, model weights,
server logs, saved chats, memory, and workspace checkpoints. Output from runs
the model makes is logged in `.arborist-logs/` inside the workspace.

## Chat commands

| Command | What it does |
|---|---|
| `/help` | list the commands |
| `/exit`, `/quit` | leave (Ctrl-D works too) |
| `/clear` | forget the conversation, keep the system prompt |
| `/effort LEVEL` | how long the model thinks (levels from `model.efforts`) |
| `/think` | toggle showing the model's thinking |
| `/tools` | toggle the file, run and web tools for this session |
| `/file PATH [PATH…]` | attach file contents to your next message |
| `/system [TEXT]` | show the system prompt, or replace it (clears the conversation) |
| `/retry` | regenerate the last reply (file changes are not undone) |
| `/save [PATH]` | write the conversation to Markdown |
| `/stats` | context usage and last-reply speed |
| `/recall [-n] TERM` | search saved memories and attach the hits to your next message (`-n` only shows them) |
| `/undo [ID]` | restore the workspace's files from before the model's last change (or checkpoint `ID`); previews and asks first |
| `/checkpoints` | list workspace checkpoints, newest first |
| `/compact` | shrink the conversation now: stub old tool output, then summarise older turns |

`"""` on its own line starts and ends a multi-line message. Ctrl-C stops the
current reply, tool or run. A pasted path like `/etc/hosts` is sent as text,
not treated as a command; `//help me` sends the literal `/help me`.

## What the model can do

| Tool | What it does | Asks you first? |
|---|---|---|
| `view_tree` | directory tree of the workspace | no |
| `outline_file` | a source file's imports, classes and functions with line numbers | no |
| `read_file` | read a text file, in pages | no |
| `list_files` | find files by glob | no |
| `grep` | regex search across files | no |
| `edit_file` | replace exact text in a file | yes, with a diff (unless `approve_edits = "auto"`) |
| `write_file` | create or overwrite a file | yes, with a diff (unless `approve_edits = "auto"`) |
| `check_dependencies` | check that a file's interpreter/compiler and imports are installed | no |
| `run_file` | run a `.py`, `.rs`, `.dart`, `.js` or `.ts` file | **always** |
| `run_tests` | run the project's tests (pytest/unittest, cargo, dart/flutter, npm, node) | **always** |
| `fetch_url` | read a web page as text | the first time per site |
| `crawl_docs` | follow links under one docs directory and search them | the first time per site |
| `remember` | save a one-line note that's loaded into future sessions | only after web content is in the conversation |

`view_tree`, `list_files` and `grep` skip `.gitignore`d files. There are
guardrails for small models: a step limit per message, and a forced pause
that makes the model explain itself and ask you when it fails several times
in a row or repeats the same call. When tests fail, the model lists the bugs
and the chat asks whether to let it fix them.

## Security

This runs a model that acts on your files with your permissions. The approval
prompts are the main safety mechanism, so read them. Specifically:

- **Everything in the workspace is readable without asking.** File tools are
  confined to the folder you start from (symlinks out of it are refused), but
  inside it the model can read anything, including `.env` files, keys, and
  credential files in any subfolder. The search tools skip `.gitignore`d
  files, but `read_file` still opens one asked for by name. **Don't start
  arborist from your home directory or any folder above secrets.** It warns
  if the workspace is your home directory, and refuses `/`.
- **Running code is not sandboxed.** `run_file` and `run_tests` execute with
  your user's full permissions: that code can read anything your account can
  (well outside the workspace), use the network, and delete things. Every run
  asks first and shows the command; look at what you're approving.
  `run_tests` runs whatever the project's test setup runs (`conftest.py`,
  `npm test` scripts and so on).
- **Web approval is per site, not per URL.** Once you approve a host, or list
  it in `tools.web_allowed_hosts`, later requests to it don't ask. So a model
  that has read a secret from the workspace could send it to an
  already-approved site inside a URL. There's no indication it would, but the
  harness can't prevent it. Set `tools.web_enabled = false` if that matters
  for a given folder.
- **The model is never given credentials.** Web requests are plain GETs with
  no cookies or auth headers, only to public addresses (localhost, LAN and
  private ranges are refused, redirects included). The system prompt includes the
  values of only the environment variables named in `chat.env_vars`, so
  never list one that holds a secret.
- **Web pages can contain prompt injection.** Fetched text is labelled as
  reference, not instructions. Once web content is in the conversation,
  edits and new memories ask again even in `auto` mode. Runs always ask.
- **The server has no auth.** It listens on `127.0.0.1`, so any program
  running as any user on your machine can use it. Don't change `server.host`
  to `0.0.0.0` on a network you don't trust.
- **Local data is plain text.** Saved chats include file contents the model
  read; memory notes are in `memory/memory.md`; checkpoints are copies of
  workspace files (anything covered by `.gitignore` is skipped, but an
  un-ignored `.env` gets copied). All of it is under `paths.root`.
- **Supply chain.** The build uses whatever is at `build.ref`; pin a commit
  hash once you have a build you trust. The model checksum comes from the same
  Hugging Face repo as the file, so it catches a corrupt download, not a
  compromised repo.

## Tests

```sh
pip install -r requirements-dev.txt
python -m pytest
```

Tests for missing toolchains (Rust, Dart, Node) skip themselves. CI runs them
on Python 3.11 and 3.13, plus `shellcheck` on the launcher.

## License

[0BSD](LICENSE): do anything you like with it, no attribution required.
