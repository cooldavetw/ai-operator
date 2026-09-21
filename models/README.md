The setup downloader selects **Google Gemma 3 1B IT, Q4_K_M**, converted and
published by [ggml-org](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF).
The base model is from Google DeepMind; Qwen and models from
Chinese developers are excluded by this project's model-selection requirement.

No HF token or Hugging Face login is required. Run from the project root
on Ubuntu/Linux (Bash 4+, curl, CA certificates, GNU coreutils and util-linux's
flock; no Python or jq):

```bash
bash scripts/download_model.sh
```

The source commit, expected size (806,058,240 bytes), and SHA-256 are pinned in
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
redirects, and sends no Authorization header. This Q4_K_M file has a different
checksum from the previous Google QAT Q4_0 model; update `.env` after switching.

Changing the pin requires reviewing the upstream commit and LFS SHA-256, updating
the manifest, and repeating the real-model integration and university evaluation
tests. Also record the license, chat template, runtime version, and PoC results
in the release manifest. A successful download verifies integrity, not model quality.

Gemma 4 E2B remains a comparison candidate if Gemma 3 1B is insufficient. Neither
model's Traditional Chinese accuracy or chat template compatibility has yet been
validated in this project. Only text classification is implemented.
