# arborist

A terminal coding assistant that runs a local model on your own GPU. One script
builds [llama.cpp](https://github.com/ggml-org/llama.cpp), downloads a GGUF
model from Hugging Face, starts the server, and drops you into a chat where the
model can read, search and edit files in the current directory, run code and
tests, and look things up on the web, asking you before anything that changes
a file, runs code, or goes online.

This is a personal project, built over a few days for my own PC (Linux Mint,
AMD RX 9070 16 GB, running PrismML's Bonsai 2 27B). It's here in case you want
to fiddle with your own version, see [License](#license).

## Requirements

- Linux (the launcher is bash and the install hints are for Debian/Ubuntu/Mint)
- Python 3.11+, plus [`prompt_toolkit`](https://github.com/prompt-toolkit/python-prompt-toolkit)
  for the chat prompt (listed in `pyproject.toml`). The launcher installs it with
  `pip install --target` into arborist's data folder (`~/.local/share/arborist/python-packages`)
  if it isn't already importable, so the system Python is left alone; that needs `pip`
  (`sudo apt install python3-pip`). Without it the chat still works, with a plain
  `input()` prompt. `pytest` for the tests.
- `bubblewrap` (`bwrap`) to run code the model writes
- `git`, `cmake`, `curl`, a C++ compiler; `ninja` optional but faster
- A GPU backend: **Vulkan** (default; `libvulkan-dev glslc spirv-headers
  mesa-vulkan-drivers`) or **ROCm/HIP**. CPU works, slowly.
- Disk for the build plus the model (~10 GB for the default)

```sh
sudo apt install build-essential cmake ninja-build git curl bubblewrap \
  libvulkan-dev glslc spirv-headers mesa-vulkan-drivers
```

## Quick start

```sh
git clone https://github.com/BProgramming/arborist ~/arborist
cd ~/some/project            # the model's tools work on this folder
~/arborist/arborist.sh
```

The first run clones and builds llama.cpp, downloads and checksums the model,
starts the server and opens the chat. Later runs reuse all of that. When you
leave the chat, a server the script started is stopped.

### Run it from anywhere with one word

Add an alias to your shell so you can just type `agent` in any project folder.
Open your bash config:

```sh
nano ~/.bashrc
```

Add this line at the bottom (change the path if you cloned it somewhere else):

```sh
alias agent="$HOME/arborist/arborist.sh"
```

Save with Ctrl-O, Enter, then exit with Ctrl-X. Load it into the current
terminal (new terminals pick it up automatically):

```sh
source ~/.bashrc
```

Now `cd` into a project and run `agent`. Everything below works through the
alias too: `agent --no-tools`, `agent status`, `agent stop`. The script finds
its own config wherever it's called from. (On zsh, use `~/.zshrc` instead.)

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
| `./arborist.sh task run --task FILE` | delegate one task (see [Delegated tasks](#delegated-tasks)) |

## Configuration

Everything is in [`.config/config`](.config/config) (TOML), commented line by line;
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
server logs, saved chats, memory (`memory/memory.toml`), workspace checkpoints, and parked
chats with their KV caches (`sessions/`, `kv/`). Output from runs
the model makes is logged in `.arborist-logs/` inside the workspace.

## Memory

The model saves notes with its `remember` tool to `memory/memory.toml` under
`paths.root`. Each note goes in one of two places: `[general]`, for facts about
you that hold everywhere, or a block named after the folder arborist was started
in, for facts about that project:

```toml
[general]
memories = ["2026-10-26: Prefers tabs over spaces"]
outdated = []

["~/code/project1"]
memories = ["2026-10-26: Tests now use pytest"]
outdated = ["2026-10-20: Tests use unittest"]
```

A session sees `[general]` plus the blocks for its own folder, the folders
above it and the folders inside it. Started in `~/code/project1` it sees notes
saved for `~/code` (say, "all my repos use ruff"); started in `~/code` it sees
the `project1` notes too. Notes for sibling folders stay out. A `--no-tools`
chat uses the folder it was started in.

`memories` go into the system prompt (up to a size limit: this folder's notes
first, then parent folders, nearest first, then general, then subfolders). `outdated` notes stay out of the prompt,
but `/recall` still finds them. When the model saves a note that looks like one
already in that block, you're shown both and asked whether to **replace** the
old one (it moves to `outdated`), **keep** both, or **skip** saving. Edit the
file freely, comments included: arborist edits it in place, adding or moving
only the lines for the notes it saves, and checks the result before writing. If
you've laid a block out in a way it can't edit safely, it reports that and
leaves the file alone. An old `memory.md` is imported into `[general]` the
first time, and renamed `memory.md.migrated`.

## Resuming

When you leave a chat (`/exit`, Ctrl-D), arborist parks it: the conversation
goes to `sessions/` under `paths.root`, and llama-server saves its KV cache
for it to `kv/`. That cache is the model's processed reading of the whole
conversation, the expensive part of every request. Start arborist in the same
folder later and the banner shows `Parked: …`; `/resume` puts the conversation
back and loads the cache into the server, so your next message is the only
new text the model has to read. `/stats` shows how much came from the cache
(`prompt 85 @ 900 tok/s, 41,203 cached`).

- **One parked chat per folder** (a `--no-tools` chat parks separately). The
  next chat you leave there replaces it. A chat you leave empty doesn't.
- **The resumed chat is the same conversation.** Same system prompt, memories
  and date as when it was parked (that's what lets the cache match), with a
  note to the model saying when it was resumed and that files may have
  changed. The model has to read files again before editing them. If web
  content was in the conversation, edits still ask in `auto` mode.
- **The cache is skipped, not wrong, when it can't match:** another model or
  KV cache type, a changed system prompt (settings or an arborist update), or
  a server that was already running before you turned this on (`agent stop`,
  then start again). The conversation still comes back; the model just reads
  it all once.
- **It costs disk**: about 64 KiB per token at f16 (a 30k-token chat is ~2 GB,
  and saving or loading it takes a few seconds). Parked chats are deleted after
  `chat.resume_days` (default 7). `chat.resume = false` turns it all off.
- The cache is saved for the chat's slot (slot 0, the only one with
  `server.parallel = 1`). If you run a delegated task or a second chat on the
  same server while the chat is open, the slot may hold the other session,
  and the saved cache won't match; nothing breaks, it just isn't reused.
  While another session is [private](#private-sessions), the cache isn't
  saved at all (the messages still park).
- **A server stays up while anyone is using it.** If the chat whose launcher
  started the server leaves while another chat or task is still attached,
  the server keeps running; `./arborist.sh stop` stops it afterwards.

## Private sessions

Type `/private` and nothing from the session is kept once you leave. You can
do it partway through a chat: the conversation carries on, and what the
session has already written is taken back first:

- **Memories the model saved this session**: you're asked about each one
  (Enter removes it, `n` keeps it). A removed note's replaced notes become
  current again. Ctrl-C at the question calls `/private` off, with nothing
  changed.
- **Input history**: this session's entries come out of the history file;
  lines from other arborist windows stay.
- **Checkpoints**: this session's move to a store in the scratch folder below
  (with the older ones, so `/undo` and `/diff` still reach as far back), and
  the normal store is rewound to how it was when the session started and
  garbage-collected, so their labels (which quote your messages) are gone
  from disk. A store the session created is deleted.
- **Logs and build output**: run logs and compiled programs are deleted, and
  this session's entries come out of the error logs.

From then on:

- **Not parked.** No `sessions/` file, no saved KV cache; `/resume` is off. On
  exit arborist asks llama-server to erase the chat's slot, so the server
  doesn't hold the conversation either. (That needs the server's cache saving,
  which `chat.resume = true` turns on; without it you're told the cache stays
  in the server's memory, never on disk, until the server stops.)
- **No memory saves.** The model gets no `remember` tool and is told why.
  Existing memories are still read, and `/recall` still works.
- **No input history.** Lines you type stay in memory for ↑ and are gone on exit.
- **Logs and checkpoints go to a scratch folder**, `private/<pid>/` under
  `paths.root`, deleted on exit. If arborist is killed before it can clean
  up, the next start deletes the folder.

**Other sessions on the same server.** Every chat and delegated task uses the
server's one cache slot, so a private session isn't alone with the model if
another session is attached. Arborist keeps track of who is (a locked file
per session under `active/` in `paths.root`; a session that crashes drops
out on its own) and tells you:

- a session that **starts while a private one is open** is warned at startup;
- a session that **goes private while others are open** is warned then, and
  again before your next message if another one has started since;
- **no session saves the cache while another one is private**, so a normal
  chat that parks can't write the private conversation to disk. It says so
  when it parks.

Only sessions sharing `paths.root` and the server's host and port see each
other; some other program pointed at the same server is invisible to this.

It lasts until you leave; `/clear` doesn't end it. What it can't take back:
files the model changed in your workspace stay changed, `/save` writes the
file you asked for, and a parked chat you `/resume`d is still parked (until
the next normal chat you leave in that folder replaces it). If anything can't
be taken back, it says what. The toolbar shows `PRIVATE` while it's on.

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
| `/resume` | carry on with the chat parked when you last left one in this folder, KV cache included (see [Resuming](#resuming)) |
| `/stats` | context usage and last-reply speed |
| `/recall [-n] TERM` | search the memories this folder can see, outdated ones too, and attach the hits to your next message (`-n` only shows them) |
| `/undo [ID]` | restore the workspace's files from before the model's last change (or checkpoint `ID`); previews and asks first. Files in a separate git repository nested inside the workspace aren't covered (arborist says so at startup); start it inside that repository instead |
| `/checkpoints` | list workspace checkpoints, newest first |
| `/compact` | shrink the conversation now: stub old tool output, then summarise older turns |
| `/diff [-f] [PATH…]` | files changed since this session started, in the workspace or under `PATH` (`-f`/`--full`: the whole diff) |
| `/config` | open `.config/config` in nano; `[tools]` changes apply right away, the rest after a restart |
| `/private` | keep nothing from this session once you leave, taking back what it has written so far (see [Private sessions](#private-sessions)) |

### The prompt

With `prompt_toolkit` (see [Requirements](#requirements)) the prompt helps as
you type:

- **Command menu.** Typing `/` opens a menu of commands with their help text,
  narrowing as you type. ↑/↓ or Tab pick one; Enter runs it. A name you've
  typed partway that only one command matches is finished for you when you
  press Enter (`/che` runs `/checkpoints`).
- **`/undo `** lists recent checkpoint IDs with when they were taken and what
  they were for. **`/save `** suggests `arborist_conversation_YYYYMMDDHHMMSS.md`
  (the current time) in the folder you started in. Pressing Enter without
  picking one keeps the usual meaning: newest checkpoint, or the history folder.
- **Multi-line messages.** Alt+Enter starts a new line, Enter sends. You can
  move up into earlier lines to edit them. Pasted text arrives in one piece,
  newlines included (bracketed paste), and is sent only when you press Enter.
- **History.** ↑/↓ step through earlier messages (kept in
  `~/.local/share/arborist/history/prompt_history`). Matching history shows as
  grey text after the cursor; → accepts it.
- **Toolbar** along the bottom: `Model: …, Effort: …, Tools: ON/OFF, Context: N%`.

When prompt_toolkit is missing, or input isn't a terminal, you get the plain
readline prompt instead: Enter sends, and a multi-line paste is joined into one
message.

`"""` on its own line starts and ends a multi-line message (in either prompt). Ctrl-C stops the
current reply, tool or run. A pasted path like `/etc/hosts` is sent as text,
not treated as a command; `//help me` sends the literal `/help me`. Commands
are handled by arborist and never reach the model. Arguments split like a
shell's, so quote ones with spaces: `/file "my notes.md"`.

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
| `fetch_url` | read a web page as text | **every new page**, showing the URL and the model's reason |
| `crawl_docs` | follow links under one docs directory and search them | **once per crawl**, showing the start URL and how far it may go |
| `remember` | save a one-line note for future sessions, for this folder or everywhere | when it looks like a saved note (replace/keep/skip), or after web content is in the conversation |

`view_tree`, `list_files` and `grep` skip `.gitignore`d files. There are
guardrails for small models: a step limit per message, and a forced pause
that makes the model explain itself and ask you when it fails several times
in a row or repeats the same call. When tests fail, the model lists the bugs
and the chat asks whether to let it fix them.

## Security

This runs a model that acts on your files with your permissions. The approval
prompts are the main safety mechanism, so read them. Specifically:

- **Everything in the workspace is readable without asking.** File tools are
  confined to the folder you start from (symlinks out of it are refused, by
  `grep` and the listings as well as `read_file`), but
  inside it the model can read anything, including `.env` files, keys, and
  credential files in any subfolder. The search tools skip `.gitignore`d
  files, but `read_file` still opens one asked for by name. **Don't start
  arborist from your home directory or any folder above secrets.** It refuses
  `/`, and warns when the workspace is your home directory, `~/Documents`,
  `~/Desktop` or `~/Downloads`, or a folder holding several separate projects.
  Repos nested inside the workspace are listed with their own `.gitignore`.
- **arborist protects itself.** Its own code and config can be read by the
  model (useful for reviewing arborist with arborist) but never changed:
  edits are refused before you're asked, and runs see them read-only. Its
  data folder (`paths.root`: models, memory, chats, checkpoints) can't be
  read at all, and runs see it empty.
- **Code the model writes runs in a bubblewrap sandbox.** `run_file`,
  `run_tests` and the dependency checks before them run under `bwrap`, which
  must be installed (package `bubblewrap`); without it nothing runs. Inside,
  a run sees `/usr`, `/etc` and the toolchain folders it needs (read-only,
  with credential files in them blanked), the workspace (writable), an empty
  throwaway home directory, and a fresh `/tmp`. It has no network (loopback
  only, its own), can't reach local sockets (Docker, D-Bus, your desktop,
  the llama-server), can't read anything else of yours (`~/.ssh`, other
  projects, arborist's memory), can't see other processes, and can't create
  namespaces of its own. In the workspace, every `.git`, virtualenv and
  `node_modules`, however deep (a cloned repo in a subfolder, a venv called
  `env`), is read-only, so a run can't plant a git hook or poison an
  environment you use yourself. A run can still create a new repo or venv;
  if it does, arborist warns you and names the hooks and `.pth` files to check
  before you use git or Python there; the same goes for a symlink it
  points out of the workspace. Dependency checks get a read-only
  workspace, and the one check that asks an interpreter where its packages
  live (which may be the project's own `.venv/bin/python`) sees only `/usr`,
  `/etc` and that interpreter's install folders.
  Every run asks first, says whether the model wrote the file this session
  and whether it changed after the diff you were shown (`[v]iew` shows it),
  warns when web content is in the conversation, and lists any folder
  outside the workspace it can write; afterwards arborist lists every file it added, changed or deleted
  (by ctime, so resetting a timestamp doesn't hide an edit). The openings
  are all off by default and named in `[tools]`: `run_writable`,
  `run_env_keep`, `run_display` (your Wayland socket only; X11 isn't
  offered) and `run_gpu`. `run_sandbox = false` turns it all off, and every
  run prompt then says so in red. The model can't edit anything in `.git`,
  and arborist's own git calls ignore a repo's `core.fsmonitor`, which could
  otherwise run a command.
- **Nothing on screen can be disguised.** Model replies, program output, file
  contents in diffs, file names and URLs pass through a filter that shows
  any terminal control sequence as visible text (`␛[2K`), so nothing can
  erase or redraw a prompt, hide text, or copy to your clipboard (OSC 52).
  Bidirectional overrides ("Trojan Source") show as `<U+202E>`. Edit prompts
  highlight dotted names (`file.py`, `some.host.com`); a diff too long to
  show at once must be viewed in full before `y` is accepted; edits applied
  without asking (`auto`, or `[a]ll`) are still printed in full; saved
  memories are shown in full.
- **Every web request the model chooses asks first.** The prompt shows the
  full URL, so you can see anything tucked into it (a model that has read a
  secret could otherwise send it out inside a query string). Approval covers
  only that one call. Reading further into a page already fetched doesn't
  ask, since nothing is sent. Within one approved call, requests to the same
  host go ahead: a same-host redirect, `robots.txt`, and the pages a crawl
  reaches by following the site's own links (those URLs come from the site,
  not the model). A redirect to a different host asks again.
  `tools.web_allowed_hosts` turns the prompt off for the hosts you list,
  which reopens that leak for them; it's empty by default.
- **The model is never given credentials.** Web requests are plain GETs with
  no cookies or auth headers, only to public addresses (localhost, LAN and
  private ranges are refused, redirects included). The system prompt includes the
  values of only the environment variables named in `chat.env_vars`, so
  never list one that holds a secret.
- **Web pages can contain prompt injection.** Fetched text is labelled as
  reference, not instructions. Once web content is in the conversation,
  edits and new memories ask again even in `auto` mode. Runs always ask.
- **The server has no auth.** It listens on `127.0.0.1`, so any program
  running as any user on your machine can use it. With `chat.resume` on, that
  includes loading a parked chat's cache into a slot and prompting the model
  with it, which would let another local user read that conversation through
  the server. The server only takes bare file names, so it can't be made to
  read or write outside `kv/`. Don't change `server.host`
  to `0.0.0.0` on a network you don't trust.
- **Local data is plain text.** Saved chats include file contents the model
  read; memory notes are in `memory/memory.toml`; parked chats are in `sessions/`
  (readable only by you), with their KV caches in `kv/`; checkpoints are copies of
  workspace files (anything covered by `.gitignore` is skipped, but an
  un-ignored `.env` gets copied). All of it is under `paths.root`.
- **Supply chain.** The build uses whatever is at `build.ref`; pin a commit
  hash once you have a build you trust. The model checksum comes from the same
  Hugging Face repo as the file, so it catches a corrupt download, not a
  compromised repo.

## Delegated tasks

`task run` hands the model one job to do on its own, so another assistant (or
you) can assign work and review the result afterwards:

```sh
cd ~/some/project
agent task run --task task.md                     # auto mode (default)
agent task run --task task.md --mode interactive  # you approve each edit and run
agent task status [ID]                            # what the newest (or named) task is doing now
agent task list                                   # tasks, newest first
agent task clean ID…                              # or: clean --all
```

The model never works on your project directly. It gets a copy under
`paths.root/tasks/ID/work`, made with git: your commits plus anything
uncommitted (staged, unstaged and untracked, minus what `.gitignore` excludes).
Runs use your project's `.venv`/`venv` read-only, since the copy has none. After
a run, the harness tells the model to carry on (or to fix failing tests) until
it finishes, gets stuck, or uses up `--max-turns` (default 6) or `--timeout`
(default 30 minutes). Then the task folder holds:

| File | What's in it |
|---|---|
| `report.json` | status, the model's final report, files changed, test results, warnings |
| `changes.patch` | everything the task changed; `git -C REPO apply --3way changes.patch` |
| `transcript.md` | the whole conversation, tool calls included |
| `task.md`, `logs/` | the task as given, and output from every run |

The status is `done`, `tests_failing` (the model said it was done but its last
test run failed), `blocked`, `out_of_steps`, `out_of_turns`, `stopped`,
`interrupted`, `timed_out` or `error`.

**auto** asks nothing: edits apply in the copy, and runs go ahead only in the
bwrap sandbox with every `[tools]` opening (`run_writable`, `run_env_keep`,
`run_display`, `run_gpu`) forced off. It refuses to start without a working
sandbox. Web and memory tools are off. **interactive** uses the same copy and
report, but asks you about every edit and run as the chat does (web tools
follow `web_enabled`), and asks before telling the model to carry on. It needs
a terminal. Either way, nothing reaches your project until someone reviews and
applies the patch.

`task run` starts the server if it isn't up and leaves it running, since tasks
tend to come in batches; `agent stop` frees the GPU.

[`claude/arborist-delegate/SKILL.md`](claude/arborist-delegate/SKILL.md) is a
Claude Code skill for using this from Claude Code: when to delegate, how to
write a task a small model can follow, and how to review the result. Link it
into `~/.claude/skills/` to use it.

## Tests

```sh
pip install -r requirements-dev.txt   # pytest and prompt_toolkit
python -m pytest
```

The prompt tests drive prompt_toolkit through a pipe instead of a terminal;
they skip if it isn't installed.

Tests for missing toolchains (Rust, Dart, Node) skip themselves. Sandbox tests
skip if bwrap can't run, unless `ARBORIST_REQUIRE_SANDBOX=1` is set; CI sets it,
so they never skip there. `tests/test_audit.py` holds the two security audits:
nothing leaves without your approval or a config option, and nothing changes
on disk without being shown to you. CI runs everything on Python 3.11 and 3.13,
plus `shellcheck` on the launcher.

For what CI can't check (your distro's bwrap, your toolchains, your Wayland
session, GPU, D-Bus and Docker sockets, a running llama-server, your real
home folder):

```sh
python3 scripts/local-checks.py           # checks, then the test suite
python3 scripts/local-checks.py --window  # also opens a pygame window via Wayland
```

## License

[0BSD](LICENSE): do anything you like with it, no attribution required.
