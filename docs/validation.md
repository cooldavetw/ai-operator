# 實作驗證紀錄

2026-09-21 在目前開發環境執行：

```text
Python 3.10.12
python3 -m pytest -q -p no:logfire -o faulthandler_timeout=20
40 passed, 1 skipped (4.58s)
```

另通過 Python compileall、Compose/CI YAML 解析、OpenAPI 匯出及檔案空白檢查。
一項環境套件的 DeprecationWarning 來自 opentelemetry，非分類服務。

測試包含：

- 完整句子規則、語意分類輸出驗證、範圍外問題、多意圖。
- 兩次澄清上限、第二次回答仍可成功分類、避免重複問句。
- 請求去重、同 ID 不同內容、競爭更新、完成／過期狀態。
- SQLite 交易失敗回滾、重啟保存、設定快照及過期清除。
- API key、body 上限、輸入驗證不回傳個資、日誌不記錄問題。
- 佇列滿載／逾時、取消等待不釋放執行中的推論位置。
- 真實子程序的逾時終止、崩潰、載入失敗與後續復原（使用測試 worker）。
- JSON schema 參數、截斷／非法輸出、token 上限與評估指標。

尚未驗證的交付環境項目：

- 本機沒有 Python 3.12；已提供以 3.12 執行測試的 CI workflow，尚未執行遠端 CI。
- 本機沒有可用 Docker daemon，因此未建置或啟動 Container；YAML 解析不等同 Compose 執行驗證。
- 本機沒有 .NET SDK，因此 C# 範例未本機編譯；已提供 CI build job。
- 尚未提供 GGUF，也未安裝原生 llama-cpp-python；真實模型整合測試明確略過。
- 因此目前不宣稱 Gemma 4 相容性、分類準確率、CPU 延遲或併發容量已達標。

取得模型及目標 VM 後，依 README 的真實模型測試與 docs/deployment.md 的
PoC 流程完成驗收，並將實際映像檔、模型檢查碼與測試報告一併封存。

## 模型下載工具

改用 ggml-org 公開 Q4_K_M GGUF、移除 token 需求並加入續傳後，針對 Bash 腳本執行：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_download_model.py -q
29 passed (2.17s)
```

涵蓋 revision 固定、檔案大小與雜湊驗證、安全覆蓋、競爭寫入、symlink 處理、
訊號中斷保留 partial、下次執行續傳、並行下載鎖、完整暫存檔直接驗證安裝、
忽略 token、不傳認證 header、HTTPS-only redirect、三次重試設定、
嚴格設定解析、CLI 錯誤及特殊路徑。
另在不含 Python／jq 的 PATH 下完成下載流程測試。
`bash -n scripts/download_model.sh`、`--help` 與 `--print-manifest` 也已實際執行。

固定 revision、SHA-256 及大小來自 ggml-org repository 的 commit 與 LFS pointer。
測試使用 fake curl 與小型檔案，不會連網。Pytest 僅供開發測試使用，下載腳本本身無 Python 依賴。
另外使用真實 curl、不帶 token，對固定 revision URL 送出 bytes 0–3 請求，
成功取得 HTTP 206 與 4 bytes；確認公開下載／redirect 可用。
尚未下載完整 806 MB 檔案或以真實模型執行推論。

## 2026-09-24 Gemma 4 E2B Q4_K_L local verification

Switched the downloader pin to bartowski/google_gemma-4-E2B-it-GGUF,
revision `81012ba3538e061d5ee003f11f25335b17f82e2d`, file
`google_gemma-4-E2B-it-Q4_K_L.gguf` (4,129,050,080 bytes).
The complete downloaded file matched SHA-256
`55f18873822c8b1f27d2e76b204fafc6cd1d1ff268526a70dc95c6ef6b83d52e`.
The previous local model was retained as `models/gemma-3-1b-q4_k_m-backup.gguf`.

Verified the embedded template with llama-cpp-python 0.3.35 and
`CHAT_FORMAT=chat_template.default`: system/user text is present and the rendered
prompt does not enable thinking. The native CPU wheel was installed under `/tmp`
for this check; the Docker source build was not exercised.

Regression suite: 86 passed, 1 optional real-model pytest skipped.
Separately ran all 12 sample dialogues through the actual `Classifier`,
`LlamaRuntime`, PydanticAI adapter, and isolated SQLite database with the new GGUF.
No HTTP server or Docker container was used for that local model evaluation.
The short prompt, schema, temperature=0, n_ctx=4096, max_tokens=192, n_threads=4,
and 30-second inference timeout were retained.

- Dialogue pass rate: 11/12 (91.7%).
- Planned turn pass rate: 15/16 (93.75%).
- Final domain matches: 12/12.
- Premature completions, skipped turns, technical errors: 0.
- Median / P95 inference request latency: 8.29 / 18.77 seconds on this host.
- Remaining failure: multiple-intents turn 1 returned a clarification with
  INSUFFICIENT_INFORMATION rather than MULTIPLE_DOMAINS; its follow-up classified correctly.

Local report: `artifacts/gemma4-local-evaluation.json` (ignored runtime artifact).
These development cases are not an independent accuracy benchmark. Repeat the
HTTP evaluator on the deployment VM to measure its actual behavior and latency.
