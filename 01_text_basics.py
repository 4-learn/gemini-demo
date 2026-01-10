"""
Gemini 文字輸入基礎

學習目標：
1. 設定 Gemini API
2. 基本文字問答
3. 對話歷史管理
4. 參數調整
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# 載入環境變數
load_dotenv()


def setup_gemini():
    """設定 Gemini API"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("錯誤：請設定 GOOGLE_API_KEY 環境變數")
        print("取得 API Key: https://makersuite.google.com/app/apikey")
        return None

    genai.configure(api_key=api_key)
    return True


def demo_basic_generation():
    """基本文字生成"""
    print("=== 基本文字生成 ===\n")

    # 建立模型
    model = genai.GenerativeModel("gemini-1.5-flash")

    # 簡單問答
    prompt = "什麼是個人防護裝備（PPE）？用一段話簡單說明。"

    response = model.generate_content(prompt)

    print(f"問題: {prompt}")
    print(f"\n回答: {response.text}")


def demo_system_instruction():
    """使用 System Instruction"""
    print("\n=== System Instruction ===\n")

    # 建立帶有系統指令的模型
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction="""你是一位工安專家助手。
你的職責是：
1. 回答職業安全相關問題
2. 提供簡潔、專業的建議
3. 必要時引用相關法規
4. 使用繁體中文回答"""
    )

    # 問答
    questions = [
        "施工現場必須穿戴哪些防護裝備？",
        "如果有人沒戴安全帽該怎麼處理？",
    ]

    for q in questions:
        response = model.generate_content(q)
        print(f"問: {q}")
        print(f"答: {response.text}\n")


def demo_chat_history():
    """對話歷史管理"""
    print("=== 對話歷史 ===\n")

    model = genai.GenerativeModel("gemini-1.5-flash")

    # 建立對話
    chat = model.start_chat(history=[])

    # 多輪對話
    messages = [
        "我是新進員工，請問工地有什麼安全規定？",
        "安全帽有什麼規格要求嗎？",
        "如果安全帽損壞了怎麼辦？",
    ]

    for msg in messages:
        print(f"用戶: {msg}")
        response = chat.send_message(msg)
        print(f"助手: {response.text}\n")

    # 顯示對話歷史
    print("--- 對話歷史 ---")
    for i, entry in enumerate(chat.history):
        role = "用戶" if entry.role == "user" else "助手"
        text = entry.parts[0].text[:50] + "..." if len(entry.parts[0].text) > 50 else entry.parts[0].text
        print(f"{i+1}. [{role}] {text}")


def demo_generation_config():
    """生成參數配置"""
    print("\n=== 生成參數 ===\n")

    # 建立不同配置的模型
    configs = [
        {
            "name": "保守（低 temperature）",
            "config": genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=100,
            )
        },
        {
            "name": "創意（高 temperature）",
            "config": genai.GenerationConfig(
                temperature=0.9,
                max_output_tokens=100,
            )
        },
    ]

    prompt = "用一句話描述安全帽的重要性"

    for cfg in configs:
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=cfg["config"]
        )

        response = model.generate_content(prompt)
        print(f"{cfg['name']}:")
        print(f"  {response.text}\n")


def demo_safety_settings():
    """安全設定"""
    print("=== 安全設定 ===\n")

    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    # 自訂安全設定
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        safety_settings=safety_settings
    )

    # 測試
    prompt = "說明工地意外事故的處理流程"
    response = model.generate_content(prompt)

    print(f"問題: {prompt}")
    print(f"回答: {response.text}")

    # 顯示安全評估
    if response.candidates:
        candidate = response.candidates[0]
        if candidate.safety_ratings:
            print("\n安全評估:")
            for rating in candidate.safety_ratings:
                print(f"  {rating.category.name}: {rating.probability.name}")


def demo_streaming():
    """串流輸出"""
    print("\n=== 串流輸出 ===\n")

    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = "列出 5 項工地常見的安全違規行為"

    print(f"問題: {prompt}")
    print("回答: ", end="", flush=True)

    # 串流生成
    response = model.generate_content(prompt, stream=True)

    for chunk in response:
        print(chunk.text, end="", flush=True)

    print()  # 換行


# === 主程式 ===
if __name__ == "__main__":
    print("=" * 50)
    print("Gemini 文字輸入基礎")
    print("=" * 50)
    print()

    if not setup_gemini():
        exit(1)

    # 1. 基本生成
    demo_basic_generation()

    # 2. System Instruction
    demo_system_instruction()

    # 3. 對話歷史
    demo_chat_history()

    # 4. 生成參數
    demo_generation_config()

    # 5. 安全設定
    demo_safety_settings()

    # 6. 串流輸出
    demo_streaming()

    print("\n完成！")
