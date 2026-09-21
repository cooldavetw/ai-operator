# 部署與維運

## 環境與建置

目標為 Ubuntu Server VM、Docker Engine + Compose plugin、x86-64 CPU。
PoC 起點為 8 vCPU、16–32 GB RAM、SSD；容量與延遲需使用選定模型實測。
確認 VMware 暴露 AVX2 指令集，避免主機超額配置；預設映像檔按 AVX2 建置。
若 VM 相容模式較舊，調整 Dockerfile 的 CMake 參數並重新建置、驗證。
OpenBLAS threads 設為 1，推論 threads 由 `N_THREADS` 控制，避免過度競爭。

依賴直接版本固定於 `requirements.txt`。每次建置產生 `/srv/dependency-manifest.txt`，
記錄該映像檔的全部實際套件版本。基底映像檔及 apt 套件未以 digest 鎖定，
因此重新建置不保證位元一致；正式交付應封存**同一已驗證映像檔**與其 image ID／檢查碼。

```bash
docker compose build
docker compose run --rm --no-deps classifier cat /srv/dependency-manifest.txt > dependency-manifest.txt
docker image inspect ntou-domain-classifier:0.1.0 --format '{{.Id}}'
docker save -o ntou-domain-classifier-0.1.0.tar ntou-domain-classifier:0.1.0
sha256sum ntou-domain-classifier-0.1.0.tar models/model.gguf config/domains.yaml > SHA256SUMS
```

交付 tar、Compose、config、GGUF、依賴清單、SHA256SUMS、原始碼、文件與測試報告。
金鑰由校方另行建立，不放入交付包、版控或映像檔。模型授權與來源紀錄隨交付包保存。

模型可先依 README 執行 `bash scripts/download_model.sh` 下載；主機需要 Bash 4+、curl、
CA 憑證、GNU coreutils 與 util-linux 的 flock，不需要 Python 或 jq。
來源為 ggml-org 的公開 Gemma 3 1B IT Q4_K_M GGUF，不需要 HF_TOKEN；
支援三次重試與中斷續傳。將 `scripts/model_manifest.conf` 也封存於交付包。
此腳本是建置前的準備工具，不在 Container 啟動時執行。

## 離線安裝

```bash
sha256sum -c SHA256SUMS
docker load -i ntou-domain-classifier-0.1.0.tar
docker compose up -d --no-build --pull never
docker compose ps
curl http://127.0.0.1:8000/health/ready
```

校方仍需先準備 Docker 與 Compose；服務 Container 不需要網路下載。
以內網反向代理終止 HTTPS 並限制來源為 C# 主機。代理的 upstream timeout 應涵蓋
排隊、模型載入與推論時間（預設最差約 240 秒再加程序終止開銷）。不要直接對外公開服務。

Compose 使用 UID/GID 10001、唯讀 root filesystem、具名 `/data` volume。
若改為 bind mount，先將資料目錄權限授予 10001。`/models` 與 `/srv/config` 唯讀。
API_KEY 也可由 `API_KEY_FILE` 讀取檔案；採 Docker secrets 時自行在 Compose 掛載該檔案。

## 健康檢查與復原

- `/health/live` 200：API 活著。
- `/health/ready` 200：模型程序已載入且 SQLite 可讀；不代表已完成語意準確率驗收。
- 模型缺失、SHA-256 不符或模板不相容：ready 為 503，檢查 MODEL_PATH、MODEL_SHA256、CHAT_FORMAT。
- 推論逾時：worker 被終止，下一個需要語意推論的請求會嘗試重載模型。
- Docker healthcheck 不會自行重啟 unhealthy container；`restart` 僅處理程序退出。
- 模型重載成功後 readiness 恢復；無流量時可由維運 `docker compose restart classifier` 重新載入。
- SQLite 檔案鎖限制同一資料庫只能由一個服務實例持有，禁止 Uvicorn 多 worker 或 Compose scale。

輸入長度、JSON schema 與固定問句降低模型失控的輸出範圍，但不能保證抵抗所有語意誤導。
將否定句、跨 Domain、注入指令與離題文字加入驗收資料。

## 資料保存與備份

SQLite 保存原始問題、澄清回答、設定快照與重試結果。預設對話 30 分鐘過期，
過期資料再保存最多 24 小時供錯誤辨識，啟動、每次分類及每 60 秒執行清除。
`EXPIRED_RETENTION_SECONDS=0` 可在過期後清除；清除後舊對話回 404 而非 410。
此為邏輯刪除，不承諾 SSD、WAL、快照或備份的鑑識性抹除；備份需由校方管理保存期限。

一般日誌只記錄 request ID、結果、耗時或錯誤代碼，Compose 日誌輪替為 10 MB × 3。
反向代理也應避免記錄 body 或 X-API-Key。服務不提供原始問題查詢介面。

使用 SQLite backup API 取得一致備份，避免只複製仍在使用中的主檔：

```bash
docker compose exec classifier python -c "import sqlite3; a=sqlite3.connect('/data/classifier.sqlite3'); b=sqlite3.connect('/data/backup.sqlite3'); a.backup(b); b.close(); a.close()"
docker compose cp classifier:/data/backup.sqlite3 ./backup.sqlite3
```

備份暫存檔也包含對話資料，移出後依校方程序清除。還原前停止服務，
將備份放入一個**新的資料 volume** 作為 `classifier.sqlite3` 並設定 UID/GID 10001；
不要混用舊 WAL/SHM 檔案。更新 Compose 指向該 volume 後再啟動。

## 更新與回復

1. 封存舊映像檔、GGUF、設定及資料庫一致備份。
2. 在測試 VM 使用新版本跑 API 回歸、真實模型測試、校方保留集及併發壓測。
3. 停止舊服務，替換映像檔與設定，再 `docker compose up -d --no-build --pull never`。
4. 驗證 readiness 與實際分類；失敗時停止服務，還原成套的舊映像、模型、設定及相容資料庫。

本版設定只在啟動載入，既有對話維持舊快照。更新模型會影響所有後續推論，
建議模型切換安排在維護窗口，等待有效對話到期。未來 schema 變更須提供 migration，
不可直接假設跨版本 SQLite 相容。

## PoC 驗收

獨立標註資料應涵蓋每個 Domain、UNKNOWN、模糊登入、多意圖、兩輪澄清與用詞變化。
`scripts/evaluate.py` 可產生每類 precision/recall/F1、macro-F1、自動分類覆蓋率與正確率、
範圍外誤轉率、平均澄清次數及技術錯誤數；ERROR 與 UNRESOLVED 不會混入 UNKNOWN。
範例資料的同義改寫不可分散於開發與驗收集。

另以實際預期到達率和併發數進行壓測，記錄端到端 P95、429 比率、RSS、CPU、
重啟恢復及斷網運作。Docker `stats` 可觀察資源；本版不內建 Prometheus。
分類準確率、延遲與容量門檻由校方依目標 VM 的 PoC 結果確定。
