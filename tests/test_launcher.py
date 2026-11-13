"""arborist.sh: the parts that can be exercised without building or serving a model."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import needs

HERE = Path(__file__).resolve().parent.parent
SCRIPT = HERE / "arborist.sh"
MODEL = '[model]\nrepo = "someone/Some-Model-GGUF"\nfile = "Some-Model-Q4_K_M.gguf"\n'

pytestmark = needs("bash", "git")


@pytest.fixture
def env(tmp_path):
    cfg = tmp_path / "test.toml"
    cfg.write_text(f'[paths]\nroot = "{tmp_path / "root"}"\n' + MODEL)
    return {**os.environ, "ARBORIST_CONFIG": str(cfg), "PYTHON": sys.executable, "NO_COLOR": "1"}


def functions(body: str, env: dict, **variables: str) -> subprocess.CompletedProcess:
    """Run `body` with arborist.sh's functions defined (everything but its last line, the call to main)."""
    # Sourced from a pipe, the script can't tell where it lives: say so.
    script = f'source <(sed \'$ d\' "{SCRIPT}")\nHERE="{HERE}"\n{body}\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          env={**env, **variables}, timeout=60)


def test_unknown_command_gets_a_proper_message(env):
    r = subprocess.run(["bash", str(SCRIPT), "frobnicate"], capture_output=True, text=True, env=env)
    assert r.returncode == 1
    assert "Unknown command: frobnicate" in r.stderr and "unbound variable" not in r.stderr


def test_stop_leaves_a_process_that_isnt_the_server_alone(env, tmp_path):
    bystander = subprocess.Popen(["sleep", "30"])
    pid_file = tmp_path / "server.pid"
    pid_file.write_text(f"{bystander.pid}\n")
    try:
        r = functions("stop_server", env, PID_FILE=str(pid_file))
        assert r.returncode == 0, r.stderr
        assert "isn't llama-server" in r.stderr
        assert bystander.poll() is None  # still running
        assert not pid_file.exists()     # the stale file is cleared either way
    finally:
        bystander.kill()


def test_stop_still_stops_a_real_server(env, tmp_path):
    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\nexec sleep 30\n")
    fake.chmod(0o755)
    # `exec -a` keeps the name in the command line, as the real binary's path does.
    server = subprocess.Popen(["bash", "-c", f'exec -a "{fake}" sleep 30'])
    pid_file = tmp_path / "server.pid"
    pid_file.write_text(f"{server.pid}\n")
    try:
        r = functions("stop_server", env, PID_FILE=str(pid_file))
        assert r.returncode == 0, r.stderr
        assert server.wait(timeout=10) != 0 and not pid_file.exists()
    finally:
        server.kill()


def _repo(path: Path, marker: str) -> None:
    git = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    (path / "which").write_text(marker + "\n")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", marker], check=True)
    subprocess.run([*git, "branch", "other"], check=True)
    (path / "which").write_text(marker + " (other branch)\n")
    subprocess.run([*git, "checkout", "-q", "other"], check=True)
    subprocess.run([*git, "commit", "-qam", marker + " other"], check=True)
    subprocess.run([*git, "checkout", "-q", "main"], check=True)


def test_changing_build_repo_or_ref_switches_the_source(env, tmp_path):
    fork, stock, src = tmp_path / "fork", tmp_path / "stock", tmp_path / "src"
    _repo(fork, "fork")
    _repo(stock, "stock")

    def fetch(repo: Path, ref: str) -> str:
        r = functions("fetch_source 0", env, CFG_SRC_DIR=str(src), CFG_REPO=str(repo), CFG_REF=ref)
        assert r.returncode == 0, r.stderr
        return (src / "which").read_text().strip()

    assert fetch(fork, "main") == "fork"
    assert fetch(fork, "main") == "fork"                    # nothing changed: nothing fetched
    assert fetch(fork, "other") == "fork (other branch)"    # build.ref edited
    assert fetch(stock, "other") == "stock (other branch)"  # build.repo edited
    assert fetch(stock, "main") == "stock"


def test_a_clone_from_before_the_ref_file_isnt_rechecked_out(env, tmp_path):
    fork, src = tmp_path / "fork", tmp_path / "src"
    _repo(fork, "fork")
    subprocess.run(["git", "clone", "-q", str(fork), str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "checkout", "-q", "other"], check=True)
    r = functions("fetch_source 0", env, CFG_SRC_DIR=str(src), CFG_REPO=str(fork), CFG_REF="main")
    assert r.returncode == 0, r.stderr
    assert (src / "which").read_text().strip() == "fork (other branch)"  # left as it was
    assert (src / ".git" / "arborist-ref").read_text().strip() == "main"


def test_local_checks_start_without_a_terminal():
    r = subprocess.run([sys.executable, str(HERE / "scripts" / "local-checks.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "ValueError" not in r.stderr


# ---- downloads --------------------------------------------------------------------

DOWNLOAD = """
curl() {{  # stands in for the real download: writes 4 bytes to the -o file
  local out=""
  while (($#)); do [[ "$1" == -o ]] && out="$2"; shift; done
  printf 'data' >"$out"
}}
remote_info() {{ {remote}; }}
df() {{ printf 'Filesystem 1024-blocks Used Available Capacity Mounted\\n/dev/x 0 0 {avail_kb} 0%% /\\n'; }}
download "https://example.invalid/m.gguf" "$DEST"
"""
SHA_OF_DATA = "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7"


def download(env, tmp_path, remote: str, avail_kb: int = 10**9):
    dest = tmp_path / "models" / "m.gguf"
    dest.parent.mkdir(exist_ok=True)
    r = functions(DOWNLOAD.format(remote=remote, avail_kb=avail_kb), env, DEST=str(dest))
    return r, dest


def test_download_is_marked_verified_only_when_it_was(env, tmp_path):
    r, dest = download(env, tmp_path, "true")  # no checksum available
    assert r.returncode == 0, r.stderr
    assert dest.read_text() == "data" and "unverified" in r.stderr
    assert not Path(f"{dest}.verified").exists()
    # Next start: the checksum can be fetched now, so the same file gets checked.
    r, dest = download(env, tmp_path, f"echo {SHA_OF_DATA} 4")
    assert r.returncode == 0, r.stderr
    assert Path(f"{dest}.verified").exists() and "SHA256 ok" in r.stdout


def test_download_with_a_wrong_checksum_is_set_aside(env, tmp_path):
    r, dest = download(env, tmp_path, f"echo {'0' * 64} 4")
    assert r.returncode == 1 and "Checksum mismatch" in r.stderr
    assert not dest.exists() and Path(f"{dest}.bad").exists()


def test_free_space_is_checked_against_the_files_real_size(env, tmp_path):
    two_gb = 2 * 1024**3
    r, dest = download(env, tmp_path, f"echo {SHA_OF_DATA} {two_gb}", avail_kb=1024 * 1024)  # 1 GB free
    assert r.returncode == 1 and "needs 3 GB" in r.stderr and not dest.exists()
    # A small file no longer needs 10 GB free.
    r, dest = download(env, tmp_path, f"echo {SHA_OF_DATA} 4", avail_kb=1024 * 1024)
    assert r.returncode == 0, r.stderr


# ---- a server other sessions are using ------------------------------------------------

def test_exit_leaves_the_server_up_while_another_session_is_attached(env, tmp_path):
    # Another session, in a process of its own, registered for this config's server.
    code = ("import sys, time; sys.path.insert(0, sys.argv[1]); import config, presence\n"
            "p = presence.Presence(presence.folder(config.load())); p.start()\n"
            "print('up', flush=True); time.sleep(60)")
    other = subprocess.Popen([sys.executable, "-c", code, str(HERE)], stdout=subprocess.PIPE,
                             text=True, env=env)
    fake = tmp_path / "llama-server"
    server = subprocess.Popen(["bash", "-c", f'exec -a "{fake}" sleep 60'])
    leave = f'load_config; SERVER_PID={server.pid}; stop_server_unless_shared'
    try:
        assert other.stdout.readline().strip() == "up"
        r = functions(leave, env)
        assert r.returncode == 0, r.stderr
        assert "1 other session(s) still use it" in r.stdout and server.poll() is None
        other.kill()
        other.wait()
        r = functions(leave, env)  # the last one out stops it
        assert r.returncode == 0, r.stderr
        assert server.wait(timeout=10) != 0
    finally:
        other.kill()
        server.kill()


# ---- other distros, other GPUs -------------------------------------------------------

def _only(tmp_path, *commands: str) -> str:
    """A PATH holding just these (empty) commands, for `command -v`."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    for old in bin_dir.iterdir():
        old.unlink()
    for name in commands:
        (bin_dir / name).write_text("#!/bin/sh\n")
        (bin_dir / name).chmod(0o755)
    return str(bin_dir)


@pytest.mark.parametrize("manager, expected", [
    ("apt-get", "sudo apt install build-essential cmake ninja-build git curl libvulkan-dev glslc"),
    ("dnf", "sudo dnf install gcc-c++ make cmake ninja-build git curl vulkan-headers"),
    ("pacman", "sudo pacman -S base-devel cmake ninja git curl vulkan-headers"),
])
def test_install_hints_name_this_distros_packages(env, tmp_path, manager, expected):
    r = functions(f'PATH="{_only(tmp_path, manager)}"; install_hint base vulkan', env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith(expected), r.stdout


def test_install_hint_on_an_unknown_distro_lists_what_to_look_for(env, tmp_path):
    r = functions(f'PATH="{_only(tmp_path)}"; install_hint bwrap pip', env)
    assert r.stdout.strip() == "install with your package manager (Debian/Ubuntu names): bubblewrap python3-pip"


def test_missing_cuda_toolkit_is_reported_with_where_to_get_it(env, tmp_path):
    path = _only(tmp_path, "dnf", "git", "cmake", "curl", "c++", "ninja", "bwrap")
    r = functions(f'PATH="{path}"; CFG_BACKEND=cuda; check_deps', env)
    assert r.returncode == 1
    assert "CUDA toolkit (nvcc)" in r.stderr and "developer.nvidia.com" in r.stderr
    assert "sudo dnf install gcc-c++" in r.stderr


def test_first_run_creates_your_config_from_the_example(env, tmp_path):
    env = {k: v for k, v in env.items() if k != "ARBORIST_CONFIG"}
    env["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")
    env["HOME"] = str(tmp_path / "home")  # paths.root in the example is under ~
    r = subprocess.run(["bash", str(SCRIPT), "status"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    mine = tmp_path / "xdg" / "arborist" / "config"
    assert mine.read_text() == (HERE / ".config" / "config").read_text()
    assert f"Created your config: {mine}" in r.stderr
    assert f"config   {mine}" in r.stdout
    again = subprocess.run(["bash", str(SCRIPT), "status"], capture_output=True, text=True, env=env)
    assert "Created your config" not in again.stderr and f"config   {mine}" in again.stdout
