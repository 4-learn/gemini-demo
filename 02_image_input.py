"""
Gemini 圖像輸入與描述

學習目標：
1. 上傳圖像到 Gemini
2. 圖像描述與分析
3. 多圖像比較
4. 圖像問答
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io
import base64

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


def create_sample_image():
    """建立範例圖像（模擬工地場景）"""
    # 建立一個簡單的測試圖像
    img = Image.new('RGB', (400, 300), color=(200, 200, 200))

    # 儲存為位元組
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    return img_bytes


def demo_image_description():
    """圖像描述"""
    print("=== 圖像描述 ===\n")

    model = genai.GenerativeModel("gemini-1.5-flash")

    # 這裡使用一個公開的圖像 URL 作為範例
    # 實際使用時可以換成本地圖像
    print("注意：此範例需要提供實際圖像")
    print("可以使用以下方式載入圖像：\n")

    print("""
# 方法 1: 從檔案載入
img = Image.open("construction_site.jpg")
response = model.generate_content([
    "描述這張圖片中的安全狀況",
    img
])

# 方法 2: 從 URL 載入
import requests
img_data = requests.get("https://example.com/image.jpg").content
img = Image.open(io.BytesIO(img_data))

# 方法 3: 從 base64 載入
img_data = base64.b64decode(base64_string)
img = Image.open(io.BytesIO(img_data))
""")


def demo_image_analysis_prompt():
    """圖像分析提示詞設計"""
    print("\n=== 圖像分析提示詞 ===\n")

    # 展示不同的提示詞策略
    prompts = {
        "通用描述": "描述這張圖片的內容",

        "安全檢查": """分析這張工地照片，檢查以下安全項目：
1. 人員是否配戴安全帽
2. 人員是否穿著反光背心
3. 是否有其他安全隱患

請以 JSON 格式回報。""",

        "違規偵測": """你是工安監控系統的 AI 分析員。
分析這張圖片，找出任何違反安全規定的情況。

輸出格式：
- 違規類型：（如：未戴安全帽、未穿反光背心）
- 位置描述：（如：畫面左側）
- 嚴重程度：（高/中/低）
- 建議處理：（具體建議）""",

        "PPE 檢查清單": """檢查圖片中人員的個人防護裝備（PPE）：

□ 安全帽
□ 護目鏡
□ 反光背心
□ 安全鞋
□ 手套

對每項標記 ✓（有配戴）或 ✗（未配戴）""",
    }

    for name, prompt in prompts.items():
        print(f"【{name}】")
        print(prompt)
        print("-" * 40)
        print()


def demo_multi_image():
    """多圖像比較"""
    print("=== 多圖像比較 ===\n")

    print("""
# 比較兩張圖片
model = genai.GenerativeModel("gemini-1.5-flash")

img1 = Image.open("before.jpg")  # 違規狀態
img2 = Image.open("after.jpg")   # 改善後

response = model.generate_content([
    "比較這兩張工地照片，說明安全狀況的改善",
    img1,
    img2
])

print(response.text)
""")

    print("多圖像輸入的應用場景：")
    print("1. 違規前後對比")
    print("2. 不同攝影機角度整合")
    print("3. 時序變化分析")
    print("4. 標準圖 vs 實際圖比對")


def demo_image_qa():
    """圖像問答"""
    print("\n=== 圖像問答 ===\n")

    print("""
# 針對圖像進行多輪問答
model = genai.GenerativeModel("gemini-1.5-flash")

img = Image.open("construction_site.jpg")

# 建立對話
chat = model.start_chat(history=[])

# 第一輪：描述
response = chat.send_message([
    "這張圖片裡有多少人？",
    img
])
print(f"助手: {response.text}")

# 第二輪：追問（不需要再傳圖片）
response = chat.send_message("他們都有戴安全帽嗎？")
print(f"助手: {response.text}")

# 第三輪：詳細分析
response = chat.send_message("如果有人沒戴安全帽，他在圖片的什麼位置？")
print(f"助手: {response.text}")
""")


def demo_image_to_json():
    """圖像轉結構化資料"""
    print("\n=== 圖像轉 JSON ===\n")

    prompt_template = """分析這張工地照片，以 JSON 格式輸出分析結果。

輸出格式：
```json
{
    "timestamp": "分析時間",
    "scene_type": "場景類型（如：施工區、辦公區）",
    "people_count": 人數,
    "violations": [
        {
            "type": "違規類型",
            "description": "描述",
            "location": "位置",
            "severity": "high/medium/low"
        }
    ],
    "overall_safety_score": 0-100 的安全評分,
    "recommendations": ["建議1", "建議2"]
}
```

只輸出 JSON，不要其他文字。"""

    print("提示詞範本：")
    print(prompt_template)
    print()

    print("使用方式：")
    print("""
import json

model = genai.GenerativeModel("gemini-1.5-flash")
img = Image.open("site.jpg")

response = model.generate_content([prompt_template, img])

# 解析 JSON
result = json.loads(response.text.strip("```json").strip("```"))
print(result)
""")


def demo_batch_processing():
    """批次圖像處理"""
    print("\n=== 批次處理 ===\n")

    print("""
import os
from pathlib import Path

model = genai.GenerativeModel("gemini-1.5-flash")

# 處理資料夾中所有圖像
image_folder = Path("./images")
results = []

for img_path in image_folder.glob("*.jpg"):
    img = Image.open(img_path)

    response = model.generate_content([
        "這張圖片是否有安全違規？回答 yes 或 no，並簡述原因",
        img
    ])

    results.append({
        "file": img_path.name,
        "analysis": response.text
    })

    # 避免超過 API 限制
    import time
    time.sleep(1)

# 輸出結果
for r in results:
    print(f"{r['file']}: {r['analysis']}")
""")


# === 主程式 ===
if __name__ == "__main__":
    print("=" * 50)
    print("Gemini 圖像輸入與描述")
    print("=" * 50)
    print()

    if not setup_gemini():
        exit(1)

    # 1. 圖像描述
    demo_image_description()

    # 2. 分析提示詞
    demo_image_analysis_prompt()

    # 3. 多圖像
    demo_multi_image()

    # 4. 圖像問答
    demo_image_qa()

    # 5. 轉 JSON
    demo_image_to_json()

    # 6. 批次處理
    demo_batch_processing()

    print("\n完成！")
