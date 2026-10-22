#!/usr/bin/env bash
# bonsai.sh — build, download and chat with Bonsai 2 27B locally.
#
#   ./bonsai.sh            set up anything missing, start the server, open a chat
#                          (tools work on the current directory; pass
#                          --workspace DIR or --no-tools to change that)
#   ./bonsai.sh setup      build + download only
#   ./bonsai.sh server     run the server in the foreground (for other clients)
#   ./bonsai.sh update     pull the latest fork and rebuild
#   ./bonsai.sh rebuild    rebuild from the current source
#   ./bonsai.sh status     show what's installed and whether a server is up
#   ./bonsai.sh stop       stop a server this script left running
#
# Settings live in bonsai.toml (or point BONSAI_CONFIG at another file).
# Inference runs in C++ (PrismML's llama.cpp fork); config + chat are Python.

set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

# ------------------------------------------------------------------ output ---
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  B=$'\033[1m'; D=$'\033[2m'; R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; X=$'\033[0m'
else
  B='' D='' R='' G='' Y='' X=''
fi
step() { printf '%s==>%s %s\n' "$G$B" "$X" "$*"; }
info() { printf '    %s%s%s\n' "$D" "$*" "$X"; }
warn() { printf '%swarning:%s %s\n' "$Y$B" "$X" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$R$B" "$X" "$*" >&2; exit 1; }

trap 'die "failed at line $LINENO: $BASH_COMMAND"' ERR

# ------------------------------------------------------------------ config ---
load_config() {
  command -v "$PY" >/dev/null || die "python3 not found"
  "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
    || die "Python 3.11+ is needed (for tomllib); found $("$PY" --version 2>&1)"
  local out
  out="$("$PY" "$HERE/config.py" --shell)" || die "could not load config"
  eval "$out"
  mapfile -t SERVER_ARGS < <("$PY" "$HERE/config.py" --server-args)
  mkdir -p "$CFG_ROOT" "$CFG_MODEL_DIR" "$CFG_LOG_DIR" "$CFG_HISTORY_DIR"
  PID_FILE="$CFG_ROOT/server.pid"
  SERVER_LOG="$CFG_LOG_DIR/server.log"
}

# ------------------------------------------------------------ dependencies ---
check_deps() {
  local missing=() hints=()
  for c in git cmake curl; do command -v "$c" >/dev/null || missing+=("$c"); done
  command -v c++ >/dev/null || command -v g++ >/dev/null || missing+=("c++ compiler")
  command -v ninja >/dev/null || info "Ninja not found; using make (slower builds, same binary)."
  hints+=("build-essential cmake ninja-build git curl")

  case "$CFG_BACKEND" in
    vulkan)
      command -v glslc >/dev/null || missing+=("glslc")
      [[ -f /usr/include/vulkan/vulkan.h ]] || missing+=("Vulkan headers")
      hints+=("libvulkan-dev glslc spirv-headers mesa-vulkan-drivers")
      ;;
    hip)
      command -v hipconfig >/dev/null || missing+=("ROCm (hipconfig)")
      hints+=("(ROCm 7.x from AMD's repo — see rocm.docs.amd.com)")
      ;;
  esac

  if ((${#missing[@]})); then
    printf '%smissing:%s %s\n' "$R$B" "$X" "${missing[*]}" >&2
    printf 'On Linux Mint / Ubuntu:\n  sudo apt install %s.\n' "${hints[*]}" >&2
    exit 1
  fi

  if [[ "$CFG_BACKEND" == vulkan ]] && command -v vulkaninfo >/dev/null; then
    if ! vulkaninfo --summary 2>/dev/null | grep -qiE 'deviceType *= *PHYSICAL_DEVICE_TYPE_DISCRETE_GPU'; then
      warn "Vulkan doesn't report a discrete GPU. RDNA4 needs Mesa 25.0+; check 'vulkaninfo --summary'."
    fi
  fi
}

# ---------------------------------------------------------------- build ---
build_stamp() {
  # Anything that changes the binary goes in here; a mismatch forces a rebuild.
  local commit
  commit="$(git -C "$CFG_SRC_DIR" rev-parse HEAD 2>/dev/null || echo none)"
  echo "$commit|$CFG_BACKEND|$CFG_HIP_TARGET|$CFG_NATIVE"
}

fetch_source() {
  local update="${1:-0}"
  if [[ ! -d "$CFG_SRC_DIR/.git" ]]; then
    step "Cloning PrismML llama.cpp fork ($CFG_REF)..."
    git clone --filter=blob:none "$CFG_REPO" "$CFG_SRC_DIR"
    update=1
  fi
  if ((update)); then
    step "Checking out $CFG_REF..."
    git -C "$CFG_SRC_DIR" fetch --quiet origin "$CFG_REF"
    git -C "$CFG_SRC_DIR" checkout --quiet --detach FETCH_HEAD
    info "at $(git -C "$CFG_SRC_DIR" log -1 --format='%h %cs %s')"
  fi
}

build() {
  local force="${1:-0}" stamp_file="$CFG_BUILD_DIR/.bonsai-stamp"
  if ((!force)) && [[ -x "$CFG_SERVER_BIN" && -f "$stamp_file" ]] \
     && [[ "$(cat "$stamp_file")" == "$(build_stamp)" ]]; then
    info "Llama-server is built and current (running via $CFG_BACKEND)."
    return
  fi

  step "Building llama-server ($CFG_BACKEND, $CFG_JOBS jobs)..."
  local args=(
    -S "$CFG_SRC_DIR" -B "$CFG_BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release            # PERF: optimized build; Debug is many times slower
    -DGGML_NATIVE="$CFG_NATIVE"           # PERF: -march=native for CPU-side code
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_OPENSSL=OFF                   # we download weights ourselves; no HTTPS in the server
  )
  command -v ninja >/dev/null && args+=(-G Ninja)
  local env_prefix=()

  case "$CFG_BACKEND" in
    vulkan)
      args+=(-DGGML_VULKAN=ON)
      ;;
    hip)
      # PERF: compile kernels only for your GPU's architecture. A mismatch
      # here doesn't fail the build — it fails at runtime with "no kernel image".
      args+=(-DGGML_HIP=ON -DGPU_TARGETS="$CFG_HIP_TARGET"
             # Required for quantized KV cache with flash attention on HIP;
             # without it mixed K/V types abort. Costs build time only.
             -DGGML_CUDA_FA_ALL_QUANTS=ON)
      env_prefix=(env HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)")
      ;;
    cpu) ;;
  esac

  rm -f "$stamp_file"
  "${env_prefix[@]}" cmake "${args[@]}" >"$CFG_LOG_DIR/cmake-configure.log" 2>&1 \
    || { tail -30 "$CFG_LOG_DIR/cmake-configure.log" >&2; die "CMake configure failed (log: $CFG_LOG_DIR/cmake-configure.log)."; }
  # Full output goes to the log; the terminal gets a one-line progress counter.
  if ! "${env_prefix[@]}" cmake --build "$CFG_BUILD_DIR" -j "$CFG_JOBS" --target llama-server 2>&1 \
      | tee "$CFG_LOG_DIR/cmake-build.log" \
      | awk '/^\[[0-9]+\/[0-9]+\]/ { printf "\r\033[K    %s", substr($0, 1, 90); fflush() } END { print "" }' \
      || [[ ! -x "$CFG_SERVER_BIN" ]]; then
    tail -30 "$CFG_LOG_DIR/cmake-build.log" >&2
    die "Build failed (log: $CFG_LOG_DIR/cmake-build.log)."
  fi
  build_stamp >"$stamp_file"
  info "Built $CFG_SERVER_BIN."
}

# ------------------------------------------------------------- download ---
expected_sha256() {
  # LFS hash from the Hugging Face tree API; empty if unavailable.
  local file="$1"
  curl -fsSL "https://huggingface.co/api/models/$CFG_MODEL_REPO/tree/main" 2>/dev/null | "$PY" -c '
import json, sys
name = sys.argv[1]
try:
    for f in json.load(sys.stdin):
        if f.get("path") == name and f.get("lfs"):
            print(f["lfs"]["oid"]); break
except Exception:
    pass
' "$file" || true
}

download() {
  local url="$1" dest="$2" name ok_marker
  name="$(basename "$dest")"
  ok_marker="$dest.verified"
  [[ -f "$dest" && -f "$ok_marker" ]] && { info "$name present"; return; }

  if [[ ! -f "$dest" ]]; then
    local avail_gb
    avail_gb="$(df -Pk "$(dirname "$dest")" | awk 'NR==2 { print int($4 / 1048576) }')"
    ((avail_gb >= 10)) || die "Only ${avail_gb} GB free in $(dirname "$dest"); need ~10 GB."
    step "Downloading $name..."
    # -C - resumes a partial .part file if a previous download was interrupted.
    curl -fL --retry 5 --retry-delay 3 -C - --progress-bar -o "$dest.part" "$url" \
      || die "Download failed — rerun to resume ($dest.part)."
    mv "$dest.part" "$dest"
  fi

  local want got
  want="$(expected_sha256 "$name")"
  if [[ -n "$want" ]]; then
    step "Verifying $name..."
    got="$(sha256sum "$dest" | cut -d' ' -f1)"
    if [[ "$got" != "$want" ]]; then
      mv "$dest" "$dest.bad"
      die "Checksum mismatch for $name (moved to $dest.bad); rerun to download again."
    fi
    info "SHA256 ok."
  else
    warn "Couldn't fetch a checksum for $name; skipping verification."
  fi
  touch "$ok_marker"
}

fetch_models() {
  download "$CFG_MODEL_URL" "$CFG_MODEL_PATH"
  if [[ "$CFG_VISION" == 1 ]]; then
    download "$CFG_MMPROJ_URL" "$CFG_MMPROJ_PATH"
  fi
}

# --------------------------------------------------------------- server ---
server_up() { curl -fs -o /dev/null --max-time 2 "$CFG_BASE_URL/health"; }

start_server() {
  if server_up; then
    info "Reusing the server already running at $CFG_BASE_URL."
    STARTED_SERVER=0
    return
  fi
  step "Starting llama-server on $CFG_BASE_URL..."
  info "Log: $SERVER_LOG."
  # setsid puts the server in its own session so Ctrl-C in the chat (which
  # stops a reply) doesn't also reach the server and kill it.
  local launcher=()
  command -v setsid >/dev/null && launcher=(setsid)
  "${launcher[@]}" "$CFG_SERVER_BIN" "${SERVER_ARGS[@]}" </dev/null >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" >"$PID_FILE"
  STARTED_SERVER=1

  local waited=0
  # /health returns 503 while the model loads, 200 when ready.
  until server_up; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -25 "$SERVER_LOG" >&2
      die "Llama-server exited during startup (full log: $SERVER_LOG)."
    fi
    ((waited >= CFG_STARTUP_TIMEOUT)) && { stop_server; die "Server not ready after ${CFG_STARTUP_TIMEOUT}s."; }
    printf '\r    %sLoading model… %ss%s' "$D" "$waited" "$X"
    sleep 1
    ((waited += 1))
  done
  printf '\r    Loading model… Done.'
  report_offload
}

report_offload() {
  # Surface the line that tells you whether the GPU is actually doing the work.
  # (The fork logs the GPU's name only at debug level, so this is the one to use.)
  [[ "$CFG_BACKEND" == cpu ]] && return 0
  local offload="" _
  # The server writes its log from a background thread; give it a moment.
  for _ in {1..20}; do
    offload="$(grep -m1 -E 'Offloaded [0-9]+/[0-9]+ layers' "$SERVER_LOG" || true)."
    [[ -n "$offload" ]] && break
    sleep 0.25
  done
  if [[ -z "$offload" ]]; then
    info "Couldn't find the layer-offload line in the server log. If replies run at only a few tokens/sec, the GPU isn't being used (see log)."
  elif [[ "$offload" =~ offloaded\ ([0-9]+)/([0-9]+) ]] && ((BASH_REMATCH[1] < BASH_REMATCH[2])); then
    warn "Only ${BASH_REMATCH[1]} of ${BASH_REMATCH[2]} layers are on the GPU — expect a large slowdown."
    warn "Lower server.ctx_size or use the PTQ1_0 packing."
  else
    info "${offload##*: }"
  fi
}

stop_server() {
  local pid="${SERVER_PID:-}"
  [[ -z "$pid" && -f "$PID_FILE" ]] && pid="$(cat "$PID_FILE")"
  [[ -z "$pid" ]] && return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in {1..20}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}

# ------------------------------------------------------------- commands ---
setup() {
  check_deps
  fetch_source 0
  build 0
  fetch_models
}

cmd_chat() {
  setup
  start_server
  if ((STARTED_SERVER)); then
    trap 'stop_server' EXIT
    trap 'exit 130' INT TERM
  fi
  # Ctrl-C belongs to the chat client (it stops a reply); keep it from
  # killing this script while chat.py runs.
  trap '' INT
  "$PY" "$HERE/chat.py" "$@" || true
  trap 'exit 130' INT
}

cmd_server() {
  setup
  server_up && die "A server is already running at $CFG_BASE_URL."
  step "Serving on $CFG_BASE_URL (Ctrl-C to stop)."
  exec "$CFG_SERVER_BIN" "${SERVER_ARGS[@]}"
}

cmd_status() {
  printf '%sconfig%s   %s\n' "$B" "$X" "$CFG_CONFIG_PATH"
  printf '%sbackend%s  %s\n' "$B" "$X" "$CFG_BACKEND"
  if [[ -x "$CFG_SERVER_BIN" ]]; then
    printf '%sbinary%s   %s (%s)\n' "$B" "$X" "$CFG_SERVER_BIN" \
      "$(git -C "$CFG_SRC_DIR" log -1 --format='%h %cs' 2>/dev/null || echo '?')"
  else
    printf '%sbinary%s   Not built\n' "$B" "$X"
  fi
  if [[ -f "$CFG_MODEL_PATH" ]]; then
    printf '%smodel%s    %s (%s)\n' "$B" "$X" "$CFG_MODEL_FILE" "$(du -h "$CFG_MODEL_PATH" | cut -f1)"
  else
    printf '%smodel%s    Not downloaded\n' "$B" "$X"
  fi
  if server_up; then
    printf '%sserver%s   %sUp%s at %s\n' "$B" "$X" "$G" "$X" "$CFG_BASE_URL"
  else
    printf '%sserver%s   Down\n' "$B" "$X"
  fi
}

main() {
  load_config
  STARTED_SERVER=0
  local cmd="${1:-chat}"
  # Options with no command go to the chat: ./bonsai.sh --workspace ~/proj
  if [[ "$cmd" == -* && "$cmd" != -h && "$cmd" != --help ]]; then
    cmd=chat
  elif (($#)); then
    shift
  fi
  case "$cmd" in
    chat)    cmd_chat "$@" ;;
    setup)   setup; step "Ready. Run ./bonsai.sh to chat." ;;
    server)  cmd_server ;;
    update)  check_deps; fetch_source 1; build 1 ;;
    rebuild) check_deps; fetch_source 0; build 1 ;;
    status)  cmd_status ;;
    stop)    stop_server; info "Stopped" ;;
    -h|--help|help) sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
    *) die "Unknown command: $1 (try --help)." ;;
  esac
}

main "$@"
