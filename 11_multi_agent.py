"""
Demo：Multi-Agent（多個 Agent 協作）

3 個 Agent 各司其職：偵測 → 法規 → 報告

執行方式：
  python 11_multi_agent.py
  python 11_multi_agent.py --mock
"""

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()


class Agent:
    """簡單的 Agent 封裝"""

    def __init__(self, name, instruction, model="gemini-2.5-flash", mock=False):
        self.name = name
        self.instruction = instruction
        self.model_name = model
        self.mock = mock

        if not mock:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model = genai.GenerativeModel(model, system_instruction=instruction)

    def run(self, user_input):
        if self.mock:
            return self._mock_run(user_input)

        response = self.model.generate_content(user_input)
        return response.text

    def _mock_run(self, user_input):
        """模擬回傳"""
        if "偵測" in self.name:
            return json.dumps({
                "violations": [{"type": "no_helmet", "severity": "high"}],
                "person_count": 3,
            }, ensure_ascii=False)
        elif "法規" in self.name:
            return "職業安全衛生設施規則第 281 條：雇主對於在高度 2 公尺以上之工作場所，應使勞工確實使用安全帽。"
        elif "報告" in self.name:
            return "【緊急告警】偵測到人員未配戴安全帽。依據職安法第 281 條，應立即要求該人員配戴安全帽，並加強巡檢。"
        return "OK"


if __name__ == "__main__":
    use_mock = "--mock" in sys.argv

    print("=" * 55)
    print("  Multi-Agent：3 個 Agent 協作")
    print("=" * 55)

    # 1. 建立 3 個 Agent
    detection_agent = Agent(
        name="偵測 Agent",
        instruction="你是工安偵測專家。分析工地狀況，用 JSON 格式列出違規。只輸出 JSON。",
        mock=use_mock,
    )

    regulation_agent = Agent(
        name="法規 Agent",
        instruction="你是工安法規專家。根據違規類型，查詢對應的台灣法規條文。簡潔回答。",
        mock=use_mock,
    )

    report_agent = Agent(
        name="報告 Agent",
        instruction="你是告警報告撰寫專家。根據違規資訊和法規，生成專業的告警報告。用繁體中文。",
        mock=use_mock,
    )

    # 2. 偵測
    print(f"\n{'─' * 55}")
    print(f"  Step 1: 偵測 Agent")
    print(f"{'─' * 55}")

    detection_result = detection_agent.run("A區工地有 3 個工人，其中 1 人沒戴安全帽")
    print(f"  輸出: {detection_result}")

    # 3. 查法規
    print(f"\n{'─' * 55}")
    print(f"  Step 2: 法規 Agent")
    print(f"{'─' * 55}")

    regulation_result = regulation_agent.run(f"查詢以下違規的法規：{detection_result}")
    print(f"  輸出: {regulation_result}")

    # 4. 寫報告
    print(f"\n{'─' * 55}")
    print(f"  Step 3: 報告 Agent")
    print(f"{'─' * 55}")

    report = report_agent.run(
        f"根據以下資訊生成告警報告：\n違規：{detection_result}\n法規：{regulation_result}"
    )
    print(f"  輸出: {report}")

    print(f"\n{'=' * 55}")
    print(f"  3 個 Agent 協作完成")
    print(f"{'=' * 55}")
