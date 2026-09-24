The setup downloader selects **Google Gemma 4 E2B IT, Q4_K_L**, converted and
published by [bartowski](https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF).
The base model is from Google DeepMind; Qwen and models from
Chinese developers are excluded by this project's model-selection requirement.

No HF token or Hugging Face login is required. Run from the project root
on Ubuntu/Linux (Bash 4+, curl, CA certificates, GNU coreutils and util-linux's
flock; no Python or jq):

```bash
bash scripts/download_model.sh
```

The source commit, expected size (4,129,050,080 bytes), and SHA-256 are pinned in
[`scripts/model_manifest.conf`](../scripts/model_manifest.conf). The manifest is
strict key=value data, never sourced as shell code. The script
streams to a temporary file in the destination directory, verifies both size
and hash, then publishes `models/model.gguf`. Use `--force` to replace an existing
file, `--output` to choose an offline staging path, or `--print-manifest` to view
the source as JSON without downloading. The script uses `curl --retry 3 -C -`.
Network failures and interruptions retain `<destination>.<sha256>.part` for the
next run to resume. The hash in its name prevents mixing different models.
Corrupt complete or oversized partials are discarded. A `.download.lock` file
serializes writers; the empty lock file remains, but its OS lock is released on exit.
`--timeout` sets the connect/low-speed timeout in seconds (default 60); each
transfer attempt has a one-hour limit. Keep the host curl package security-updated.
Stop the inference service before replacing a model it has memory-mapped.

Copy the printed `MODEL_SHA256` value to `.env` for verification on every worker
start. No token is saved, no model is fetched during service startup, and no model
weights are committed to this repository. An offline deployment only needs the
verified file and manifest. The script ignores `.curlrc`, only permits HTTPS
redirects, and sends no Authorization header. Update `MODEL_SHA256` and set `CHAT_FORMAT=chat_template.default` in `.env` after switching.
The pinned model uses Apache-2.0. Keep thinking disabled (do not add `<|think|>`).
The existing llama-cpp-python 0.3.35 pin includes upstream Gemma 4 support.

Changing the pin requires reviewing the upstream commit and LFS SHA-256, updating
the manifest, and repeating the real-model integration and university evaluation
tests. Also record the license, chat template, runtime version, and PoC results
in the release manifest. A successful download verifies integrity, not model quality.

Gemma 4 replaces the Gemma 3 baseline; classification accuracy and CPU latency
must be measured with the dialogue evaluator before accepting it for deployment.
Only text classification is implemented; no multimodal projector is needed.

For an existing server, stop the classifier, preserve `models/model.gguf` as a
rollback copy, run `bash scripts/download_model.sh --force`, update the `.env`
checksum and chat format, then recreate the container. Download verification
alone does not verify inference or classification quality.

Server upgrade (run in the project directory after updating the code):

```bash
docker compose stop classifier
# Keep the old GGUF for rollback; this refuses to overwrite an existing backup.
cp -n models/model.gguf models/gemma-3-backup.gguf
bash scripts/download_model.sh --force
```

Set these values in the existing `.env` (keep your API key and other settings):

```dotenv
MODEL_SHA256=55f18873822c8b1f27d2e76b204fafc6cd1d1ff268526a70dc95c6ef6b83d52e
CHAT_FORMAT=chat_template.default
```

```bash
unset MODEL_SHA256 CHAT_FORMAT
docker compose up -d --build classifier
# After readiness succeeds, use the deployment's API_KEY in your shell:
python3 -m scripts.evaluate examples/evaluation.jsonl --output artifacts/gemma4-dialogue-evaluation.json
```

If needed, rollback by stopping the classifier, restoring the saved GGUF, restoring
its old `MODEL_SHA256` and `CHAT_FORMAT`, and recreating the container. Do not
reuse Gemma 3's checksum with Gemma 4.
