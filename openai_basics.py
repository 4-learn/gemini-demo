"""
Demo：OpenAI 原生 API

對應講義：OpenAI 原生 API

執行方式：
  python openai_basics.py

需要：
  pip install openai python-dotenv
  .env 裡設定 OPENAI_API_KEY
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI


def demo_chat_completion():
    """Chat Completion：基本問答"""
    print("=== Chat Completion ===\n")

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是工安專家，用繁體中文回答。"},
            {"role": "user", "content": "什麼是 PPE？"},
        ],
    )

    print(f"  回答：{response.choices[0].message.content[:200]}")
    print(f"  模型：{response.model}")
    print(f"  Token：{response.usage.total_tokens}")


def demo_function_calling():
    """Function Calling：兩段式"""
    print(f"\n{'=' * 50}")
    print("=== Function Calling ===\n")

    client = OpenAI()

    # 定義工具（JSON Schema）
    tools = [
        {
            "type": "function",
            "function": {
                "name": "check_regulation",
                "description": "查詢違規類型對應的法規",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "violation_type": {
                            "type": "string",
                            "description": "違規類型，如 no_helmet, no_vest",
                        },
                    },
                    "required": ["violation_type"],
                },
            },
        },
    ]

    # 第一段：OpenAI 選擇
    print("  第一段：OpenAI 選擇要呼叫什麼")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "查安全帽的法規"}],
        tools=tools,
    )

    tool_call = response.choices[0].message.tool_calls[0]
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)
    print(f"  → {func_name}({func_args})")

    # 第二段：你執行
    print(f"\n  第二段：你執行函式")
    regulations = {
        "no_helmet": "職安法第 281 條：應使勞工確實使用安全帽。",
        "no_vest": "職安法第 21 條：應提供反光背心。",
    }
    result = regulations.get(func_args["violation_type"], "查無")
    print(f"  → {result}")

    # 結果送回 OpenAI
    print(f"\n  結果送回 OpenAI")
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "查安全帽的法規"},
            response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            },
        ],
        tools=tools,
    )
    print(f"  → {response2.choices[0].message.content[:200]}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("錯誤：請在 .env 設定 OPENAI_API_KEY")
        exit(1)

    demo_chat_completion()
    demo_function_calling()
