"""
Demo：Function Calling（Gemini 原生）

讓 Gemini 自己決定要呼叫哪個函式，而不是你寫 if-else。

執行方式：
  python 08_function_calling.py
  python 08_function_calling.py --mock

需要：
  pip install google-generativeai python-dotenv
"""

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()


# === 定義工具函式 ===

def check_regulation(violation_type: str) -> str:
    """查詢違規類型對應的法規

    Args:
        violation_type: 違規類型（如 no_helmet, no_vest）

    Returns:
        相關法規條文
    """
    regulations = {
        "no_helmet": "職業安全衛生設施規則 第 281 條：雇主對於在高度 2 公尺以上之工作場所，應使勞工確實使用安全帽。",
        "no_vest": "職業安全衛生設施規則 第 21 條：雇主應提供適當之反光標示或背心。",
        "no_goggles": "職業安全衛生設施規則 第 287 條：從事焊接作業應配戴護目鏡。",
        "blocked_exit": "建築技術規則 第 97 條：安全出口不得堆放物品。",
    }
    return regulations.get(violation_type, f"找不到 {violation_type} 的相關法規")


def send_alert(message: str, severity: str) -> str:
    """發送告警通知

    Args:
        message: 告警訊息
        severity: 嚴重程度（high/medium/low）

    Returns:
        發送結果
    """
    print(f"  📨 發送告警: [{severity.upper()}] {message}")
    return f"告警已發送：{message}"


def log_incident(description: str, location: str) -> str:
    """記錄違規事件

    Args:
        description: 事件描述
        location: 事件地點

    Returns:
        記錄結果
    """
    print(f"  📝 記錄事件: {location} - {description}")
    return f"已記錄：{location} - {description}"


# === Gemini Function Calling ===

def run_with_function_calling(user_input, mock=False):
    """讓 Gemini 決定要呼叫哪些函式"""

    if mock:
        return _mock_function_calling(user_input)

    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        tools=[check_regulation, send_alert, log_incident],
    )

    chat = model.start_chat()
    response = chat.send_message(user_input)

    # 處理 function call
    results = []
    while response.candidates[0].content.parts:
        part = response.candidates[0].content.parts[0]

        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            func_name = fc.name
            func_args = dict(fc.args)

            print(f"  🔧 Gemini 呼叫: {func_name}({func_args})")

            # 執行函式
            func_map = {
                "check_regulation": check_regulation,
                "send_alert": send_alert,
                "log_incident": log_incident,
            }

            if func_name in func_map:
                result = func_map[func_name](**func_args)
                results.append({"function": func_name, "result": result})

                # 把結果送回 Gemini
                response = chat.send_message(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=func_name,
                                response={"result": result},
                            )
                        )]
                    )
                )
            else:
                break
        else:
            # Gemini 回傳文字（不再呼叫函式）
            final_text = part.text
            print(f"\n  💬 Gemini 回覆: {final_text}")
            break

    return results


def _mock_function_calling(user_input):
    """模擬 Function Calling"""
    print(f"  🔧 Gemini 呼叫: check_regulation(violation_type='no_helmet')")
    result1 = check_regulation("no_helmet")
    print(f"  🔧 Gemini 呼叫: send_alert(message='A區未戴安全帽', severity='high')")
    result2 = send_alert("A區未戴安全帽", "high")
    print(f"  🔧 Gemini 呼叫: log_incident(description='未戴安全帽', location='A區')")
    result3 = log_incident("未戴安全帽", "A區")
    print(f"\n  💬 Gemini 回覆: 已完成處理。依據職安法第281條，該區域人員需配戴安全帽。已發送高嚴重度告警並記錄事件。")
    return [
        {"function": "check_regulation", "result": result1},
        {"function": "send_alert", "result": result2},
        {"function": "log_incident", "result": result3},
    ]


# === 主程式 ===

if __name__ == "__main__":
    use_mock = "--mock" in sys.argv

    print("=" * 55)
    print("  Function Calling：讓 Gemini 自己決定呼叫什麼")
    print("=" * 55)

    print(f"\n可用工具：")
    print(f"  1. check_regulation(violation_type) → 查法規")
    print(f"  2. send_alert(message, severity) → 發告警")
    print(f"  3. log_incident(description, location) → 記錄事件")

    # 測試 1：簡單案例
    print(f"\n{'─' * 55}")
    print(f"  測試 1：「A區有人沒戴安全帽，幫我處理」")
    print(f"{'─' * 55}\n")

    results = run_with_function_calling(
        "A區有人沒戴安全帽，請查法規、發告警、記錄事件",
        mock=use_mock,
    )

    print(f"\n  共呼叫 {len(results)} 個函式")

    # 重點
    print(f"\n{'=' * 55}")
    print("  重點")
    print(f"{'=' * 55}")
    print("""
  1. 你定義函式（Python function + docstring）
  2. Gemini 看到使用者的需求，自己決定呼叫哪些
  3. 你執行函式，把結果送回 Gemini
  4. Gemini 用結果組成最終回覆

  這就是 LangChain Tool 的底層原理。
""")
