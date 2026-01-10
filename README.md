# Gemini Demo

感知層（Perception）的 Demo 程式碼 - 多模態輸入處理。

## 安裝

```bash
pip install -r requirements.txt
```

## 設定 API Key

```bash
# 方法 1: 環境變數
export GOOGLE_API_KEY="your-api-key"

# 方法 2: .env 檔案
echo "GOOGLE_API_KEY=your-api-key" > .env
```

取得 API Key: https://makersuite.google.com/app/apikey

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `01_text_basics.py` | 文字輸入基礎 |
| `02_image_input.py` | 圖像輸入與描述 |
| `03_document_input.py` | 文件輸入與摘要 |
| `04_structured_output.py` | 多模態轉結構化資料 |
| `05_safety_analysis.py` | 工安圖像分析範例 |

## 執行

```bash
# 文字基礎
python 01_text_basics.py

# 圖像分析
python 02_image_input.py

# 工安分析範例
python 05_safety_analysis.py
```

## 注意事項

- 需要有效的 Google API Key
- 免費額度有限制（每分鐘請求數）
- 圖像分析建議使用 gemini-1.5-flash（較便宜）
