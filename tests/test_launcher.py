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
    script = f'source <(sed \'$ d\' "{SCRIPT}")\n{body}\n'
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
