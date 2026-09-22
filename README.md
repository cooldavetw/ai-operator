# 海洋大學地端 AI 總機分類服務

Python 3.12 / FastAPI / llama-cpp-python / SQLite / Docker Compose。
接收文字問題，回傳 `ACADEMIC_SYSTEM`、`LIBRARY_SYSTEM` 或 `UNKNOWN`；
資訊不足時最多詢問兩次。只回傳分類與核准的澄清問句，不回答業務問題。

## 快速部署

1. 使用下方下載腳本取得 ggml-org 發布的 Google Gemma 3 1B IT Q4_K_M GGUF，或手動放入經驗證的 `models/model.gguf`。
2. 複製 `.env.example` 為 `.env`，設定隨機 `API_KEY`（不可為空，無最短長度限制），以及模型的 `MODEL_SHA256`。
3. 校方確認 `config/domains.yaml` 中的分類邊界、範例、完整句子比對規則與問句。
4. 在已安裝 Docker Engine 與 Compose plugin 的 Ubuntu VM 執行：

```bash
docker compose build
docker compose up -d
curl http://127.0.0.1:8000/health/ready
```

首次建置需要網路；**執行時完全使用本機檔案，不會下載模型或呼叫雲端 API**。
離線部署請先在建置機產出映像檔，參閱 [部署與維運](docs/deployment.md)。
預設僅綁定 `127.0.0.1`；由校內 HTTPS 反向代理對 C# 系統提供服務。
服務不含語音辨識；語音輸入須由呼叫端先轉成文字。

## 下載 Gemma 3 1B

下載來源為 [ggml-org 的公開 GGUF](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF)，
檔案為 `gemma-3-1b-it-Q4_K_M.gguf`，不需要登入 Hugging Face 或設定 HF_TOKEN。
在有網路的 Ubuntu/Linux 準備機執行。主機只需 Bash 4+、curl、CA 憑證及
GNU coreutils（含 sha256sum、stat）與 util-linux 的 flock；不需要 Python 或 jq。
若尚未安裝 curl／憑證，可先執行 `sudo apt-get install curl ca-certificates`。

```bash
bash scripts/download_model.sh
```

腳本預設寫入專案的 `models/model.gguf`，下載約 806 MB。完成後將印出的 `MODEL_SHA256=...`
填入 `.env`。腳本不會修改 `.env`，也不會使用或傳送 token。
此檔案是 Q4_K_M 量化版，與先前 Google QAT Q4_0 檔案不同，檢查碼也已更新。

來源 revision、大小及 SHA-256 固定於 [model_manifest.conf](scripts/model_manifest.conf)。
設定檔以嚴格的 key=value 格式解析，不會作為 shell 程式執行。
完整下載並通過大小與 SHA-256 驗證後才安裝；已存在的檔案預設不覆蓋。

```bash
bash scripts/download_model.sh --print-manifest
bash scripts/download_model.sh --output /path/to/offline-bundle/model.gguf
bash scripts/download_model.sh --force
```

`--force` 僅在新檔驗證成功後替換舊檔。使用 `curl --retry 3 -C -` 重試／續傳；
中斷或網路失敗會保留 `<輸出路徑>.<SHA256>.part`，重新執行同一命令即可續傳。
雜湊不符或超出預期大小的檔案會刪除；正常完成後不保留 `.part`。
`<輸出路徑>.download.lock` 防止同時寫入，下載完成後保留空 lock 檔，執行中的鎖會自動釋放。
`--timeout 60` 設定連線／低速逾時秒數，每次傳輸嘗試另有一小時上限。
curl 只使用 HTTPS，忽略 `.curlrc`，不傳送 Authorization header。
替換前請停止正在使用模型的服務，更新 `.env` 的檢查碼後重新啟動。
離線 VM 可直接接收已驗證的 GGUF，不需 HF_TOKEN 或連外。

```bash
curl http://127.0.0.1:8000/api/v1/classify \
  -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"request_id":"6b4c2b9e-155b-46e6-a419-1e9777bfd58f","message":"我登不進去"}'
```

`.env` 由 Compose 讀取，不會自動匯入目前 shell；上述 curl 的 `API_KEY` 需另外設定。

## API 與狀態

```json
{
  "request_id": "6b4c2b9e-155b-46e6-a419-1e9777bfd58f",
  "conversation_id": "3f27d3bd-7d24-4723-a435-267be8bbbe09",
  "version": 1,
  "status": "NEEDS_CLARIFICATION",
  "domain": null,
  "clarification_count": 1,
  "clarification": {
    "id": "identify_system",
    "question": "您是指校務系統、圖書館系統，還是其他系統？"
  },
  "reason_code": "INSUFFICIENT_INFORMATION",
  "is_final": false,
  "config_version": "1"
}
```

下一輪使用新的 `request_id`，帶回 `conversation_id`、`expected_version: 1` 及使用者補充。
相同請求重試須保持 ID 和內容完全相同。兩次澄清之後仍會判斷第二次回答，能分類就完成；
仍不明確才回傳 `UNKNOWN / CLARIFICATION_LIMIT`。範圍外問題可直接 UNKNOWN。
模型故障回傳 HTTP 錯誤，不會假裝分類成功。

完整欄位、錯誤與重試語意見 [API 文件](docs/api.md)、[OpenAPI](docs/openapi.json)。
[C# 範例](examples/csharp/Program.cs) 使用 .NET 8，可執行：

```bash
dotnet run --project examples/csharp
```

## 開發與測試

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements-test.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

一般測試使用注入的推論替身，涵蓋政策、API、保存、併發與真實子程序的逾時終止；
正式服務沒有 mock 模式，也不會在模型故障時改用替身。

本機執行真實推論：

```bash
CMAKE_ARGS='-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS' pip install -r requirements.txt
export API_KEY='use-a-random-secret'
export MODEL_PATH="$PWD/models/model.gguf"
uvicorn app.main:create_app --factory --workers 1 --no-access-log
```

原生編譯需安裝 C/C++、CMake、OpenBLAS 開發套件。API 與 SQLite 僅支援單一 worker／replica，
檔案鎖會阻止兩個服務實例同時使用同一資料庫。

```bash
TEST_MODEL_PATH="$PWD/models/model.gguf" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -m model -q
python -m scripts.export_openapi
python -m scripts.evaluate examples/evaluation.jsonl --output artifacts/evaluation.json
```

範例資料只用來檢查流程，不代表準確率驗收。評估工具對執行中的 API 量測分類品質、
澄清次數、技術錯誤與單一請求端到端 P50/P95；正式驗收仍需獨立資料與併發壓測。

## 實作結構

| 檔案 | 責任 |
|---|---|
| `app/main.py` | API key、輸入限制、健康檢查、有界佇列 |
| `app/service.py` | 分類政策、最多兩次澄清、去重、版本衝突 |
| `app/runtime.py` | PydanticAI Agent、Decision 驗證、GGUF 子程序、逾時終止 |
| `app/local_agent.py` | PydanticAI 本機模型 adapter、llama.cpp JSON schema 推論 |
| `app/store.py` | SQLite 原子寫入、持久化、單一實例檔案鎖 |
| `app/config.py` | 設定驗證、Domain 定義 |
| `config/domains.yaml` | 可擴充分類、固定問句、完整句子規則 |

原始問題與澄清答案保存在本機 SQLite，以支援續接；一般日誌不記錄問題內容。
設定在啟動時載入；既有對話保留建立時的設定快照。新增 Domain 後必須更新 C# 路由及回歸案例。
LLM 的分類正確性仍取決於模型與校方定義；JSON schema 保證的是輸出結構，不是語意正確率。

目前不附模型權重；下載腳本預選 Google Gemma 3 1B IT 作為小型 CPU PoC 候選。
請依 [模型說明](models/README.md) 完成相容性、
中文準確率與 CPU 效能驗證後再上線。推論參數參考 [llama-cpp-python 官方文件](https://llama-cpp-python.readthedocs.io/en/latest/)。

### PydanticAI 比較測試

推論透過 `pydantic-ai-slim` 的 `Agent` 與 `NativeOutput(ConfiguredDecision)` 執行，
使用本機 llama.cpp adapter，不需要雲端 API key、額外模型伺服器或執行時網路。
PydanticAI 驗證欄位關係及設定內的 ID；JSON 或驗證失敗時回傳修正資訊，最多重試一次。
兩次生成共用原有 `INFERENCE_TIMEOUT_SECONDS` 期限；重試耗盡仍回傳 `INVALID_MODEL_OUTPUT`。
格式合法但語意錯誤的分類不會自動觸發重試。完整句子規則仍可略過模型。

保持模型、CHAT_FORMAT 及其他參數與 baseline 相同，更新程式後：

```bash
docker compose up -d --build classifier
# 等待 /health/ready 成功後執行；API_KEY 使用部署設定值。
export API_KEY='test'
python3 -m scripts.evaluate examples/evaluation.jsonl --output artifacts/gemma-pydanticai.json
```

比較原有 baseline 的逐筆結果、技術錯誤與延遲；更換 Agent 框架不保證提高語意準確率。
