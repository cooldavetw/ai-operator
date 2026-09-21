#!/usr/bin/env bash
# Ubuntu/Linux setup tool. No Python or jq required.
set +x
set -euo pipefail
umask 077

# Public download: do not forward credentials inherited from the caller.
unset HF_TOKEN

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'HELP'
Download the public ggml-org Gemma 3 1B IT Q4_K_M GGUF. No HF token needed.
Requires Bash 4+, curl, CA certificates, GNU coreutils and flock (Ubuntu/Linux).

Usage: bash scripts/download_model.sh [options]
  --output PATH      Destination (default: <repository>/models/model.gguf)
  --force            Replace an existing file only after verification
  --timeout SECONDS  Positive integer connect/low-speed timeout (default: 60)
  --print-manifest   Print pinned metadata as JSON without downloading
  -h, --help         Show this help

Uses curl --retry 3 -C -. Network failures retain a .part file for the next run.
Each transfer attempt has a one-hour limit. Corrupt completed files are discarded.
Stop the inference service before replacing a model that it has memory-mapped.
HELP
}

(( BASH_VERSINFO[0] >= 4 )) || fail 'Bash 4 or newer is required.'
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
destination="$script_dir/../models/model.gguf"
force=false
print_manifest=false
timeout=60
while (( $# )); do
    case $1 in
        --output)
            (( $# >= 2 )) && [[ -n $2 ]] || fail '--output requires a file path.'
            destination=$2; shift 2 ;;
        --timeout)
            (( $# >= 2 )) || fail '--timeout requires a positive integer.'
            timeout=$2; shift 2 ;;
        --force) force=true; shift ;;
        --print-manifest) print_manifest=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail 'Unknown option. Use --help.' ;;
    esac
done
[[ $timeout =~ ^[1-9][0-9]{0,5}$ ]] || fail '--timeout requires a positive integer (1–999999).'

# Parse a deliberately simple, strict format instead of sourcing shell code or
# depending on Python/jq. Validation also makes --print-manifest JSON-safe.
declare -A manifest=()
[[ -r "$script_dir/model_manifest.conf" ]] || fail 'Cannot read model_manifest.conf.'
while IFS= read -r line || [[ -n $line ]]; do
    [[ -z $line || $line == \#* ]] && continue
    [[ $line == *=* ]] || fail 'Invalid manifest entry.'
    key=${line%%=*}
    value=${line#*=}
    case $key in
        repo_id|revision|filename|sha256|size) ;;
        *) fail 'Unknown manifest field.' ;;
    esac
    [[ ! -v manifest[$key] ]] || fail 'Duplicate manifest field.'
    manifest[$key]=$value
done < "$script_dir/model_manifest.conf"
[[ ${#manifest[@]} == 5 ]] || fail 'Manifest must contain all five fields.'
repo_id=${manifest[repo_id]}
revision=${manifest[revision]}
filename=${manifest[filename]}
sha256=${manifest[sha256]}
size=${manifest[size]}
[[ $repo_id =~ ^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$ ]] || fail 'Invalid repository ID.'
[[ $revision =~ ^[a-f0-9]{40}$ ]] || fail 'Revision must be an immutable commit SHA.'
[[ $filename =~ ^[a-zA-Z0-9_.-]+\.gguf$ ]] || fail 'Invalid model filename.'
[[ $sha256 =~ ^[a-f0-9]{64}$ ]] || fail 'Invalid SHA-256.'
[[ $size =~ ^[1-9][0-9]{0,11}$ ]] || fail 'Invalid model size.'
if $print_manifest; then
    printf '{\n  "repo_id": "%s",\n  "revision": "%s",\n  "filename": "%s",\n  "sha256": "%s",\n  "size": %s\n}\n' \
        "$repo_id" "$revision" "$filename" "$sha256" "$size"
    exit 0
fi

for utility in curl sha256sum stat chmod mv ln rm mkdir dirname basename flock; do
    command -v "$utility" >/dev/null || fail "Missing required utility: $utility"
done
[[ ! -d $destination && $destination != */ ]] || fail 'Destination must be a file, not a directory.'
if [[ -e $destination || -L $destination ]] && ! $force; then
    fail 'Destination already exists. Use --force to replace it after verification.'
fi
mkdir -p -- "$(dirname -- "$destination")"
destination_dir=$(cd -- "$(dirname -- "$destination")" && pwd -P)
destination="$destination_dir/$(basename -- "$destination")"
# Serialize writers sharing the resumable file. Keep the lock file so another
# process cannot acquire a different inode while an existing lock is held.
[[ ! -L "$destination.download.lock" ]] || fail 'Download lock must not be a symlink.'
exec 9> "$destination.download.lock"
flock -n 9 || fail 'Another download is already running for this destination.'
if [[ -e $destination || -L $destination ]] && ! $force; then
    fail 'Destination already exists. Use --force to replace it after verification.'
fi
temporary="$destination.$sha256.part"
[[ ! -L $temporary && ( ! -e $temporary || -f $temporary ) ]] || fail 'Partial download must be a regular file.'
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
if [[ -e $temporary ]]; then
    partial_size=$(stat -c '%s' -- "$temporary")
    if (( partial_size > size )); then
        rm -f -- "$temporary"
    fi
fi
if [[ ! -e $temporary ]]; then : > "$temporary"; fi
chmod 0600 -- "$temporary"

printf 'Downloading %s @ %s\nSize: %s bytes. Destination: %s\n' "$repo_id" "$revision" "$size" "$destination"
# Skip the network if a previous run completed the transfer but was interrupted
# before publication. The entire file is still verified below.
if [[ $(stat -c '%s' -- "$temporary") != "$size" ]]; then
    # -q MUST be first: ignore ~/.curlrc, including any authentication settings.
    curl_status=0
    http_code=$(
        curl -q --fail --location --retry 3 --continue-at - --progress-bar --show-error --max-redirs 5 \
            --proto '=https' --proto-redir '=https' \
            --connect-timeout "$timeout" --speed-limit 1 --speed-time "$timeout" \
            --max-time 3600 --max-filesize "$size" \
            --header 'Accept-Encoding: identity' \
            --user-agent 'ntou-domain-classifier-model-setup/0.1.0' \
            --output "$temporary" --write-out '%{http_code}' \
            --url "https://huggingface.co/$repo_id/resolve/$revision/$filename"
    ) || curl_status=$?
    if (( curl_status != 0 )); then
        fail "Download failed (curl exit $curl_status, HTTP $http_code). Rerun to resume: $temporary"
    fi
    [[ $http_code == 200 || $http_code == 206 ]] || fail 'Server did not return model data; rerun to resume.'
fi
actual_size=$(stat -c '%s' -- "$temporary")
if (( actual_size > size )); then
    rm -f -- "$temporary"
    fail 'Download exceeds the pinned size; partial file discarded.'
fi
[[ $actual_size == "$size" ]] || fail 'Download is incomplete; rerun to resume.'
# Hash stdin so filenames containing backslashes cannot change sha256sum's format.
actual_sha256=$(sha256sum < "$temporary")
if [[ ${actual_sha256%% *} != "$sha256" ]]; then
    rm -f -- "$temporary"
    fail 'SHA-256 verification failed; corrupt partial file discarded. Rerun to download again.'
fi
chmod 0644 -- "$temporary"
if $force; then
    # Same-filesystem rename atomically replaces the directory entry, not a symlink target.
    mv -fT -- "$temporary" "$destination"
else
    # Atomic no-clobber publication protects against concurrent setup runs.
    ln -T -- "$temporary" "$destination" 2>/dev/null || fail 'Cannot install model; destination may have appeared during download.'
    rm -f -- "$temporary"
fi
printf 'Verified model installed at %s\nSet this in .env:\nMODEL_SHA256=%s\n' "$destination" "$sha256"
