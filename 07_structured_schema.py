"""
Demo：Structured Output（Gemini response_schema）

上一章用 prompt 說「只輸出 JSON」，但有時會失敗。
Gemini 原生的 response_schema 可以保證輸出合法 JSON。

執行方式：
  python 07_structured_schema.py
  python 07_structured_schema.py --mock

需要：
  pip install google-generativeai pillow python-dotenv
"""

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()


# === 方法 1：Prompt-based（舊方法，可能失敗） ===

def analyze_prompt_based(image_path, mock=False):
    """用 prompt 要求輸出 JSON（不保證格式正確）"""
    if mock:
        # 模擬偶爾格式錯誤
        import random
        random.seed(42)
        if random.random() < 0.3:
            return '```json\n{"violations": [{"type": "no_helmet"}]}\n```'
        return '{"person_count": 3, "violations": [{"type": "no_helmet", "severity": "high"}]}'

    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    image = Image.open(image_path)

    prompt = """分析這張工地照片，輸出 JSON 格式：
{
    "person_count": 人數,
    "violations": [{"type": "違規類型", "severity": "high/medium/low"}]
}
只輸出 JSON，不要其他文字。"""

    response = model.generate_content([prompt, image])
    return response.text


# === 方法 2：Schema-based（新方法，保證合法） ===

def analyze_schema_based(image_path, mock=False):
    """用 response_schema 保證輸出合法 JSON"""
    if mock:
        return {
            "person_count": 3,
            "violations": [
                {"type": "no_helmet", "severity": "high"},
            ],
            "recommendations": ["加強安全帽佩戴"],
        }

    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # 定義 Schema
    schema = {
        "type": "object",
        "properties": {
            "person_count": {"type": "integer", "description": "圖片中的人數"},
            "violations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "違規類型",
                            "enum": ["no_helmet", "no_vest", "no_goggles", "blocked_exit", "other"],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                    },
                    "required": ["type", "severity"],
                },
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["person_count", "violations"],
    }

    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    image = Image.open(image_path)
    response = model.generate_content(["分析這張工地照片的安全狀況", image])

    # response_schema 保證回傳合法 JSON
    return json.loads(response.text)


# === 比較 ===

if __name__ == "__main__":
    use_mock = "--mock" in sys.argv

    print("=" * 55)
    print("  Structured Output：prompt-based vs schema-based")
    print("=" * 55)

    # 方法 1：Prompt-based
    print("\n--- 方法 1：Prompt-based ---")
    print("（用 prompt 說「只輸出 JSON」）\n")

    success = 0
    fail = 0
    for i in range(5):
        raw = analyze_prompt_based("workplace.jpg", mock=use_mock)
        try:
            result = json.loads(raw)
            success += 1
            print(f"  第 {i+1} 次: ✅ 成功")
        except json.JSONDecodeError:
            fail += 1
            print(f"  第 {i+1} 次: ❌ JSON 解析失敗")
            print(f"    原始回傳: {raw[:80]}...")

    print(f"\n  成功: {success}/5, 失敗: {fail}/5")

    # 方法 2：Schema-based
    print("\n--- 方法 2：Schema-based ---")
    print("（用 response_schema 保證格式）\n")

    result = analyze_schema_based("workplace.jpg", mock=use_mock)
    print(f"  回傳類型: {type(result).__name__}")
    print(f"  內容:")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    print(f"\n  ✅ 一定是合法 JSON，不需要 try/except")

    # 結論
    print(f"\n{'=' * 55}")
    print("  結論")
    print(f"{'=' * 55}")
    print("""
  | 方法 | 優點 | 缺點 |
  |------|------|------|
  | prompt-based | 簡單 | 可能失敗 |
  | schema-based | 保證合法 | 需要定義 schema |

  建議：正式系統用 schema-based，快速測試用 prompt-based。
""")
