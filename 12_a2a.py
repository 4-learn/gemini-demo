"""
Demo：A2A（Agent-to-Agent 通訊協定）

用 FastAPI 模擬兩個 Agent 的 A2A 溝通。
（單進程模擬，不需要真的啟動多個 server）

執行方式：
  python 12_a2a.py

不需要 API Key（純 A2A 協定示範）
"""

import json
import uuid
from datetime import datetime


# === Agent Card ===

class AgentCard:
    """A2A Agent 名片"""

    def __init__(self, name, description, url, capabilities, input_schema=None, output_schema=None):
        self.name = name
        self.description = description
        self.url = url
        self.capabilities = capabilities
        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "capabilities": self.capabilities,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


# === Task ===

class Task:
    """A2A Task"""

    def __init__(self, action, input_data):
        self.task_id = f"t-{uuid.uuid4().hex[:8]}"
        self.action = action
        self.input = input_data
        self.status = "pending"
        self.output = None

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "action": self.action,
            "input": self.input,
            "status": self.status,
            "output": self.output,
        }

    def complete(self, output):
        self.status = "completed"
        self.output = output


# === Agent 實作 ===

class DetectionAgent:
    """偵測 Agent"""

    card = AgentCard(
        name="偵測 Agent",
        description="分析工地狀況，找出違規",
        url="http://localhost:8000",
        capabilities=["detect_violations"],
        input_schema={"description": "string"},
        output_schema={"violations": "array", "person_count": "integer"},
    )

    def handle_task(self, task):
        # 模擬偵測
        task.complete({
            "violations": [{"type": "no_helmet", "severity": "high"}],
            "person_count": 3,
        })
        return task


class RegulationAgent:
    """法規 Agent"""

    card = AgentCard(
        name="法規查詢 Agent",
        description="根據違規類型查詢台灣工安相關法規",
        url="http://localhost:8001",
        capabilities=["regulation_search"],
        input_schema={"violation_type": "string"},
        output_schema={"regulation": "string", "source": "string"},
    )

    REGULATIONS = {
        "no_helmet": ("職安法第 281 條：應使勞工確實使用安全帽", "安全帽規定"),
        "no_vest": ("職安法第 21 條：應提供反光背心", "反光背心規定"),
    }

    def handle_task(self, task):
        vtype = task.input.get("violation_type", "unknown")
        reg, source = self.REGULATIONS.get(vtype, ("找不到相關法規", "N/A"))
        task.complete({"regulation": reg, "source": source})
        return task


class ReportAgent:
    """報告 Agent"""

    card = AgentCard(
        name="報告 Agent",
        description="根據違規和法規生成告警報告",
        url="http://localhost:8002",
        capabilities=["generate_report"],
        input_schema={"violation": "object", "regulation": "string"},
        output_schema={"report": "string", "severity": "string"},
    )

    def handle_task(self, task):
        violation = task.input.get("violation", {})
        regulation = task.input.get("regulation", "")
        vtype = violation.get("type", "unknown")
        severity = violation.get("severity", "medium")

        report = f"【{'緊急告警' if severity == 'high' else '一般告警'}】\n"
        report += f"偵測到違規：{vtype}\n"
        report += f"嚴重程度：{severity}\n"
        report += f"相關法規：{regulation}\n"
        report += f"建議：立即處理並加強巡檢。"

        task.complete({"report": report, "severity": severity})
        return task


# === A2A 協作流程 ===

if __name__ == "__main__":
    print("=" * 55)
    print("  A2A：Agent-to-Agent 通訊協定")
    print("=" * 55)

    # 建立 Agent
    detection = DetectionAgent()
    regulation = RegulationAgent()
    report = ReportAgent()

    # 1. 發現 Agent（讀 Agent Card）
    print(f"\n{'─' * 55}")
    print(f"  Step 1: 發現 Agent")
    print(f"{'─' * 55}")

    agents = [detection, regulation, report]
    for agent in agents:
        card = agent.card.to_dict()
        print(f"  {card['name']} ({card['url']})")
        print(f"    能力: {card['capabilities']}")

    # 2. 偵測
    print(f"\n{'─' * 55}")
    print(f"  Step 2: 偵測 Agent 執行")
    print(f"{'─' * 55}")

    task1 = Task("detect_violations", {"description": "A區有人沒戴安全帽"})
    detection.handle_task(task1)
    print(f"  Task: {task1.task_id}")
    print(f"  狀態: {task1.status}")
    print(f"  結果: {json.dumps(task1.output, ensure_ascii=False)}")

    # 3. 查法規
    print(f"\n{'─' * 55}")
    print(f"  Step 3: 法規 Agent 執行")
    print(f"{'─' * 55}")

    violation = task1.output["violations"][0]
    task2 = Task("regulation_search", {"violation_type": violation["type"]})
    regulation.handle_task(task2)
    print(f"  Task: {task2.task_id}")
    print(f"  狀態: {task2.status}")
    print(f"  法規: {task2.output['regulation']}")

    # 4. 生成報告
    print(f"\n{'─' * 55}")
    print(f"  Step 4: 報告 Agent 執行")
    print(f"{'─' * 55}")

    task3 = Task("generate_report", {
        "violation": violation,
        "regulation": task2.output["regulation"],
    })
    report.handle_task(task3)
    print(f"  Task: {task3.task_id}")
    print(f"  狀態: {task3.status}")
    print(f"\n{task3.output['report']}")

    # 總結
    print(f"\n{'=' * 55}")
    print(f"  完成：3 個 Agent 透過 A2A 協作")
    print(f"{'=' * 55}")
    print(f"  偵測({task1.task_id}) → 法規({task2.task_id}) → 報告({task3.task_id})")
