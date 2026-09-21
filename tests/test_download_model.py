"""Exercise the Bash setup tool with a fake curl; no network or real tokens."""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = b"GGUF fixture model\n"
TOKEN = "hf_fake_token_for_tests_only"

FAKE_CURL = r'''#!/usr/bin/env bash
set -eu
printf '%s\n' "$@" > "$CURL_ARGS"
printf '%s' "${HF_TOKEN-unset}" > "$CURL_ENV"
output=''
while (( $# )); do
    if [[ $1 == --output ]]; then output=$2; shift 2; else shift; fi
done
case ${FAKE_MODE-ok} in
    ok)
        offset=$(stat -c '%s' "$output")
        tail -c "+$((offset + 1))" "$FAKE_PAYLOAD" >> "$output"
        if (( offset )); then printf 206; else printf 200; fi ;;
    corrupt) printf 'XXXX fixture model\n' > "$output"; printf 200 ;;
    short) printf G > "$output"; printf 200 ;;
    extra) { cat "$FAKE_PAYLOAD"; printf x; } > "$output"; printf 200 ;;
    network) printf G > "$output"; printf 000; exit 28 ;;
    denied) printf 403; exit 22 ;;
    missing) printf 404; exit 22 ;;
    partial) cp -- "$FAKE_PAYLOAD" "$output"; printf 206 ;;
    race) printf competitor > "$RACE_TARGET"; cp -- "$FAKE_PAYLOAD" "$output"; printf 200 ;;
    block) printf G > "$output"; : > "$CURL_READY"; while :; do sleep 1; done ;;
esac
'''


@pytest.fixture
def setup(tmp_path):
    scripts = tmp_path / "project/scripts"
    scripts.mkdir(parents=True)
    script = scripts / "download_model.sh"
    shutil.copyfile(ROOT / "scripts/download_model.sh", script)
    manifest = scripts / "model_manifest.conf"
    text = (ROOT / "scripts/model_manifest.conf").read_text()
    lines = []
    for line in text.splitlines():
        if line.startswith("sha256="):
            line = "sha256=" + hashlib.sha256(DATA).hexdigest()
        elif line.startswith("size="):
            line = f"size={len(DATA)}"
        lines.append(line)
    manifest.write_text("\n".join(lines) + "\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    curl = binary / "curl"
    curl.write_text(FAKE_CURL)
    curl.chmod(0o755)
    payload = tmp_path / "payload"
    payload.write_bytes(DATA)
    env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}", "HF_TOKEN": TOKEN,
           "FAKE_PAYLOAD": str(payload), "CURL_ARGS": str(tmp_path / "args"),
           "CURL_ENV": str(tmp_path / "env"), "CURL_READY": str(tmp_path / "ready")}
    env.pop("HF_TOKEN", None)
    return script, manifest, env, tmp_path


def run(setup, *args, mode="ok", trace=False):
    script, _, env, directory = setup
    return subprocess.run(
        ["bash", *(["-x"] if trace else []), str(script), *map(str, args)],
        cwd=directory, env={**env, "FAKE_MODE": mode}, text=True, capture_output=True, timeout=10,
    )


def assert_clean(directory):
    assert list(directory.rglob("*.part")) == []


def test_download_and_curl_security_options(setup):
    _, _, env, directory = setup
    result = run(setup)
    assert result.returncode == 0, result.stderr
    target = directory / "project/models/model.gguf"
    assert target.read_bytes() == DATA
    assert target.stat().st_mode & 0o777 == 0o644
    args = Path(env["CURL_ARGS"]).read_text().splitlines()
    assert args[0] == "-q"  # Disable user curlrc, including trace/auth forwarding settings.
    assert "--config" not in args
    assert not any("Authorization" in arg for arg in args)
    assert args[args.index("--retry") + 1] == "3"
    assert args[args.index("--continue-at") + 1] == "-"
    assert args[args.index("--proto") + 1] == "=https"
    assert args[args.index("--proto-redir") + 1] == "=https"
    assert "--location" in args and "--location-trusted" not in args
    assert "--insecure" not in args
    assert TOKEN not in " ".join(args) + result.stdout + result.stderr
    assert Path(env["CURL_ENV"]).read_text() == "unset"
    assert f"MODEL_SHA256={hashlib.sha256(DATA).hexdigest()}" in result.stdout
    assert_clean(directory)


def test_existing_file_and_force(setup):
    _, _, env, directory = setup
    target = directory / "existing.gguf"
    target.write_bytes(b"old model")
    assert run(setup, "--output", target).returncode == 1
    assert not Path(env["CURL_ARGS"]).exists()
    assert target.read_bytes() == b"old model"
    assert run(setup, "--output", target, "--force").returncode == 0
    assert target.read_bytes() == DATA
    assert_clean(directory)


@pytest.mark.parametrize("mode", ["corrupt", "short", "extra", "network", "denied", "missing"])
def test_failure_preserves_old_file(setup, mode):
    directory = setup[3]
    target = directory / "existing.gguf"
    target.write_bytes(b"old model")
    result = run(setup, "--output", target, "--force", mode=mode)
    assert result.returncode != 0
    assert target.read_bytes() == b"old model"
    assert TOKEN not in result.stdout + result.stderr
    if mode in {"corrupt", "extra"}:
        assert_clean(directory)
    else:
        assert len(list(directory.rglob("*.part"))) == 1
    assert "Accept Google's terms" not in result.stderr


def test_racing_writer_is_not_overwritten(setup):
    _, _, env, directory = setup
    target = directory / "race.gguf"
    env["RACE_TARGET"] = str(target)
    result = run(setup, "--output", target, mode="race")
    assert result.returncode == 1
    assert target.read_text() == "competitor"
    assert len(list(directory.rglob("*.part"))) == 1
    # A verified full partial can be installed on retry without another transfer.
    Path(env["CURL_ARGS"]).unlink()
    assert run(setup, "--output", target, "--force").returncode == 0
    assert target.read_bytes() == DATA
    assert not Path(env["CURL_ARGS"]).exists()
    assert_clean(directory)


def test_symlink_is_replaced_without_touching_target(setup):
    directory = setup[3]
    original = directory / "original.gguf"
    original.write_bytes(b"old model")
    target = directory / "link.gguf"
    target.symlink_to(original)
    assert run(setup, "--output", target).returncode == 1
    assert run(setup, "--output", target, "--force").returncode == 0
    assert not target.is_symlink()
    assert original.read_bytes() == b"old model"
    assert target.read_bytes() == DATA


def test_custom_path_with_spaces_and_backslash(setup):
    target = setup[3] / "space dir" / "model\\name.gguf"
    result = run(setup, "--output", target)
    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == DATA


@pytest.mark.parametrize("args", [("--timeout", "0"), ("--timeout", "nan"), ("--timeout",), ("--output",), ("--unknown",)])
def test_invalid_arguments_fail_before_network(setup, args):
    result = run(setup, *args)
    assert result.returncode == 1
    assert not Path(setup[2]["CURL_ARGS"]).exists()


@pytest.mark.parametrize("token", ["", 'hf_bad"token', "hf_bad\nconfig"])
def test_unneeded_token_is_ignored(setup, token):
    setup[2]["HF_TOKEN"] = token
    result = run(setup)
    assert result.returncode == 0
    assert Path(setup[2]["CURL_ENV"]).read_text() == "unset"
    if token:
        assert token not in result.stdout + result.stderr


def test_shell_trace_does_not_expose_token(setup):
    setup[2]["HF_TOKEN"] = TOKEN
    result = run(setup, trace=True)
    assert result.returncode == 0
    assert TOKEN not in result.stdout + result.stderr


def test_manifest_and_help_do_not_need_credentials(setup):
    setup[2].pop("HF_TOKEN", None)
    manifest = run(setup, "--print-manifest")
    assert manifest.returncode == 0
    assert json.loads(manifest.stdout)["revision"] == "f9c28bcd85737ffc5aef028638d3341d49869c27"
    assert run(setup, "--help").returncode == 0
    assert not Path(setup[2]["CURL_ARGS"]).exists()


@pytest.mark.parametrize("change", ["unpinned", "duplicate", "injection", "missing"])
def test_manifest_is_strict_data_not_executable_code(setup, change):
    _, manifest, env, directory = setup
    text = manifest.read_text()
    if change == "unpinned":
        text = text.replace("revision=f9c28bcd85737ffc5aef028638d3341d49869c27", "revision=main")
    elif change == "duplicate":
        text += "size=12\n"
    elif change == "injection":
        text += f"evil=$(touch {directory}/executed)\n"
    else:
        text = "\n".join(line for line in text.splitlines() if not line.startswith("size="))
    manifest.write_text(text)
    assert run(setup).returncode == 1
    assert not (directory / "executed").exists()
    assert not Path(env["CURL_ARGS"]).exists()


def test_termination_preserves_partial_and_blocks_concurrent_download(setup):
    script, _, env, directory = setup
    target = directory / "existing.gguf"
    target.write_bytes(b"old model")
    process = subprocess.Popen(
        ["bash", str(script), "--output", str(target), "--force"],
        env={**env, "FAKE_MODE": "block"}, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not Path(env["CURL_READY"]).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert Path(env["CURL_READY"]).exists()
        competing = run(setup, "--output", target, "--force")
        assert competing.returncode == 1
        assert "already running" in competing.stderr
        os.killpg(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode != 0
        assert TOKEN not in stdout + stderr
        assert target.read_bytes() == b"old model"
        partials = list(directory.rglob("*.part"))
        assert len(partials) == 1 and partials[0].read_bytes() == b"G"
        assert run(setup, "--output", target, "--force").returncode == 0
        assert target.read_bytes() == DATA
        assert_clean(directory)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()


def test_downloader_runs_without_python_or_jq_on_path(setup):
    # The test runner uses Python, but the script and fake curl have only these tools.
    _, _, env, directory = setup
    binary = directory / "bin"
    for name in ["bash", "cat", "cp", "sha256sum", "stat", "chmod", "mv", "ln", "rm", "mkdir", "dirname", "basename", "flock", "tail"]:
        (binary / name).symlink_to(shutil.which(name))
    env["PATH"] = str(binary)
    result = run(setup)
    assert result.returncode == 0, result.stderr


def test_network_failure_resumes_on_next_run(setup):
    directory = setup[3]
    target = directory / "resume.gguf"
    assert run(setup, "--output", target, mode="network").returncode == 1
    assert not target.exists()
    partial = next(directory.rglob("*.part"))
    assert partial.read_bytes() == b"G"
    assert run(setup, "--output", target).returncode == 0
    assert target.read_bytes() == DATA
    assert_clean(directory)


def test_partial_content_status_is_valid_when_whole_file_verified(setup):
    result = run(setup, mode="partial")
    assert result.returncode == 0, result.stderr
