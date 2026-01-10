"""
Gemini 文件輸入與摘要

學習目標：
1. 上傳 PDF 文件
2. 文件摘要與問答
3. 長文處理策略
4. 多文件分析
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# 載入環境變數
load_dotenv()


def setup_gemini():
    """設定 Gemini API"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("錯誤：請設定 GOOGLE_API_KEY 環境變數")
        return None
    genai.configure(api_key=api_key)
    return True


def demo_pdf_upload():
    """PDF 文件上傳"""
    print("=== PDF 上傳 ===\n")

    print("""
# 上傳 PDF 文件
pdf_file = genai.upload_file("safety_manual.pdf")

# 建立模型
model = genai.GenerativeModel("gemini-1.5-flash")

# 問答
response = model.generate_content([
    "這份文件的主要內容是什麼？",
    pdf_file
])

print(response.text)

# 刪除上傳的檔案（可選）
genai.delete_file(pdf_file.name)
""")

    print("支援的文件格式：")
    print("- PDF (.pdf)")
    print("- 圖像 (.jpg, .png, .webp)")
    print("- 影片 (.mp4, .mov)")
    print("- 音訊 (.mp3, .wav)")


def demo_document_summary():
    """文件摘要"""
    print("\n=== 文件摘要 ===\n")

    # 模擬一份工安手冊內容
    sample_document = """
# 工地安全作業手冊

## 第一章 一般規定

1.1 適用範圍
本手冊適用於所有進入施工現場之人員，包括但不限於：
- 施工人員
- 監工人員
- 訪客
- 供應商

1.2 安全責任
- 雇主應提供必要之安全設備
- 員工應遵守安全規定
- 違規者將依規定處分

## 第二章 個人防護裝備

2.1 安全帽
- 所有人員進入施工區必須配戴安全帽
- 安全帽應符合 CNS 1336 標準
- 損壞之安全帽應立即更換

2.2 反光背心
- 夜間作業或有車輛通行之區域必須穿著反光背心
- 反光背心應保持清潔，反光條完整

2.3 安全鞋
- 施工現場應穿著鋼頭安全鞋
- 禁止穿著拖鞋或涼鞋進入施工區

## 第三章 危險作業

3.1 高空作業
- 2 公尺以上作業應使用安全帶
- 作業前應檢查鷹架穩固性
- 禁止在強風（6級以上）時進行高空作業

3.2 電氣作業
- 未經許可禁止操作電氣設備
- 作業前應確認斷電並上鎖
- 使用絕緣工具

## 第四章 緊急應變

4.1 火災
- 發現火災立即通報
- 使用滅火器初期滅火
- 無法控制時立即撤離

4.2 人員受傷
- 立即通報並呼叫救護
- 進行必要之急救處置
- 保護現場
"""

    model = genai.GenerativeModel("gemini-1.5-flash")

    # 摘要
    print("原文長度:", len(sample_document), "字元\n")

    response = model.generate_content(f"""
請為以下工安手冊製作摘要：

{sample_document}

請以條列方式列出：
1. 主要章節
2. 每章重點（各 1-2 句）
3. 關鍵規定（最重要的 3 項）
""")

    print("摘要結果：")
    print(response.text)


def demo_document_qa():
    """文件問答"""
    print("\n=== 文件問答 ===\n")

    document = """
職業安全衛生設施規則 第 281 條

雇主對於在高度二公尺以上之處所作業，勞工有墜落之虞者，應使勞工確實使用安全帽。

前項安全帽，應符合國家標準 CNS 1336 工業用安全帽或國家標準 CNS 14251
防護頭盔之規定，或具有同等以上之防護性能者。
"""

    model = genai.GenerativeModel("gemini-1.5-flash")

    # 建立對話
    chat = model.start_chat(history=[
        {
            "role": "user",
            "parts": [f"以下是一份法規文件，請記住它的內容：\n\n{document}"]
        },
        {
            "role": "model",
            "parts": ["好的，我已經閱讀並記住這份關於安全帽規定的法規內容。請問有什麼問題？"]
        }
    ])

    # 問答
    questions = [
        "什麼情況下必須戴安全帽？",
        "安全帽要符合什麼標準？",
        "如果只有 1.5 公尺高，需要戴嗎？",
    ]

    for q in questions:
        print(f"問: {q}")
        response = chat.send_message(q)
        print(f"答: {response.text}\n")


def demo_long_document():
    """長文處理策略"""
    print("=== 長文處理策略 ===\n")

    print("""
# Gemini 1.5 Flash 支援最多 1M tokens
# 但建議分段處理以獲得更好的結果

# 策略 1: 分段摘要
def summarize_long_doc(doc, chunk_size=10000):
    chunks = [doc[i:i+chunk_size] for i in range(0, len(doc), chunk_size)]
    summaries = []

    for i, chunk in enumerate(chunks):
        response = model.generate_content(f"摘要以下內容：\\n{chunk}")
        summaries.append(response.text)

    # 最終摘要
    combined = "\\n".join(summaries)
    final = model.generate_content(f"整合以下摘要成一份完整摘要：\\n{combined}")
    return final.text

# 策略 2: 使用 File API（推薦）
file = genai.upload_file("large_document.pdf")
response = model.generate_content([
    "這份文件的主要內容和結論是什麼？",
    file
])

# 策略 3: 重點提取
response = model.generate_content([
    '''分析這份文件，只提取：
    1. 標題和章節結構
    2. 關鍵定義
    3. 重要數字和日期
    4. 結論和建議''',
    file
])
""")


def demo_multi_document():
    """多文件分析"""
    print("\n=== 多文件分析 ===\n")

    print("""
# 比較多份文件
files = [
    genai.upload_file("policy_2023.pdf"),
    genai.upload_file("policy_2024.pdf"),
]

response = model.generate_content([
    "比較這兩份政策文件，列出主要的變更項目",
    files[0],
    files[1]
])

# 交叉引用
regulations = genai.upload_file("regulations.pdf")
violations = genai.upload_file("violation_report.pdf")

response = model.generate_content([
    '''根據法規文件，分析違規報告中的每項違規：
    1. 違反哪條法規
    2. 法規原文
    3. 建議處罰''',
    regulations,
    violations
])
""")


def demo_extraction():
    """結構化資訊提取"""
    print("\n=== 資訊提取 ===\n")

    document = """
違規報告

日期：2024 年 1 月 15 日
地點：A 區施工現場
報告人：王安全

事件描述：
於 14:30 巡查時發現，B 公司承包商李工人（員工編號 B-0042）
在 5 樓鷹架上作業時未配戴安全帽及安全帶。

經口頭提醒後，該員工表示安全帽遺留在休息區。
已要求其立即停止作業，前往取回安全帽後方可繼續。

違規項目：
1. 未配戴安全帽 - 違反職安法第 281 條
2. 未繫安全帶 - 違反職安法第 225 條

建議處置：
- 口頭警告
- 列入教育訓練名單
- 通知 B 公司加強管理
"""

    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(f"""
從以下違規報告中提取結構化資訊，以 JSON 格式輸出：

{document}

輸出格式：
```json
{{
    "report_date": "日期",
    "location": "地點",
    "reporter": "報告人",
    "violator": {{
        "name": "姓名",
        "company": "公司",
        "employee_id": "編號"
    }},
    "violations": [
        {{
            "type": "違規類型",
            "regulation": "違反法規",
            "description": "描述"
        }}
    ],
    "actions": ["處置措施"]
}}
```
""")

    print("原始報告：")
    print(document[:200] + "...\n")
    print("提取結果：")
    print(response.text)


# === 主程式 ===
if __name__ == "__main__":
    print("=" * 50)
    print("Gemini 文件輸入與摘要")
    print("=" * 50)
    print()

    if not setup_gemini():
        exit(1)

    # 1. PDF 上傳
    demo_pdf_upload()

    # 2. 文件摘要
    demo_document_summary()

    # 3. 文件問答
    demo_document_qa()

    # 4. 長文處理
    demo_long_document()

    # 5. 多文件
    demo_multi_document()

    # 6. 資訊提取
    demo_extraction()

    print("\n完成！")
