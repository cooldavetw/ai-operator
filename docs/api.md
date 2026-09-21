# API 契約

`POST /api/v1/classify` 與 `GET /openapi.json` 必須帶 `X-API-Key`。
`GET /health/live` 與 `GET /health/ready` 不需金鑰，只回傳簡短狀態。
OpenAPI 可離線閱讀或匯入內部 API 工具；服務不使用外部 CDN 的 Swagger UI。

## 請求

| 欄位 | 型別 | 規則 |
|---|---|---|
| request_id | UUID | 必填；同一請求重試保持不變，不同內容不可重用 |
| conversation_id | UUID/null | 首輪省略或 null，續接帶前次回應值 |
| expected_version | integer | 首輪 0（預設），續接必須等於前次回應 version |
| message | string | 1–2000 字元，不可全空白 |

請求 body 上限 16 KiB（包含 JSON 編碼），超過模型 token 容量回傳 422。
不接受額外欄位，也不接受呼叫端自行提供澄清次數或完整歷史。

## 回應

| status | domain | is_final | clarification |
|---|---|---|---|
| CLASSIFIED | 設定中的 Domain ID | true | null |
| NEEDS_CLARIFICATION | null | false | `{id, question}` |
| UNKNOWN | UNKNOWN | true | null |

其他欄位：`request_id`、`conversation_id`、遞增的 `version`、
`clarification_count`（已發出的問題數，0–2）、`reason_code`、`config_version`。

原因代碼：`RULE_MATCH`、`SEMANTIC_MATCH`、`INSUFFICIENT_INFORMATION`、
`MULTIPLE_DOMAINS`、`OUT_OF_SCOPE`、`CLARIFICATION_LIMIT`。
所有顯示給使用者的澄清問句均由伺服器設定提供；不傳出模型自由生成的業務回答。

## 重試、版本與期限

- 每個對話預設自建立起 30 分鐘有效，**不是滑動期限**。
- 同 ID、同內容回傳原結果，即使已有後續對話；呼叫端不可用較舊重試結果覆蓋較新版本。
- 去重有效期與對話期限相同；過期回傳 410。清除後回傳 404；已刪除的 request_id 不再可去重。
- 處理期間若對話過期，不提交結果，回傳 410；請建立新對話。
- 同 ID、不同內容回傳 409 `REQUEST_ID_REUSED`。
- 同一對話兩個競爭更新只會有一個成功，另一個回傳 409 `VERSION_CONFLICT`。
- 已完成對話不可繼續，回傳 409 `CONVERSATION_FINISHED`；新問題使用新對話。
- 模型錯誤不寫入對話或消耗澄清次數；相同 request_id 可安全重試。
- 用戶端取消等待不會取消已開始的工作；稍後以同 request_id 重試取得原結果。
- 預設最多一個執行中請求與八個等待請求。排隊和推論期限分開計算；故障後首次重試可能包含模型重載。

## 錯誤

```json
{"error":{"code":"INFERENCE_TIMEOUT","message":"模型推論逾時，請重試。"}}
```

| HTTP | 原因 | 呼叫端處理 |
|---|---|---|
| 401 | API key 無效 | 檢查設定 |
| 404/410 | 對話不存在／過期 | 建立新對話 |
| 409 | ID 重用、版本衝突或已完成 | 核對狀態；不得盲目重送舊版本 |
| 413/422 | body、欄位或 token 容量不合法 | 修正請求 |
| 429 | 佇列已滿／排隊逾時 | 遵循 Retry-After，有限次數退避重試 |
| 502 | 模型 JSON、enum、欄位關係或完整性不合法 | 同 ID 有限次數重試，持續失敗通知維運 |
| 503 | 模型未就緒、程序失敗或儲存故障 | 同 ID 退避重試、查健康狀態 |
| 504 | 推論逾時且程序已終止 | 同 ID 退避重試 |

不要將非 200 回應轉成業務上的 UNKNOWN。UNKNOWN 交人工分流，技術錯誤顯示服務暫時無法使用。
單一 API key 對應一個受信任的校內 C# 呼叫端；本版不提供多租戶或終端使用者授權隔離。
