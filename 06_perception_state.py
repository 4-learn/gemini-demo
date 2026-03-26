"""
Demo：感知結果如何進入系統狀態

完整流程：
  圖片 → Gemini 分析 → JSON → 轉成 Event → 更新系統狀態 → 觸發告警

執行方式：
  有 API Key：  python 06_perception_state.py
  沒有 API Key：python 06_perception_state.py --mock

需要：
  pip install google-generativeai pillow python-dotenv
  .env 裡設定 GOOGLE_API_KEY
  workplace.jpg（測試圖片）
"""

import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# === Gemini 分析 ===

def analyze_image(image_path, mock=False):
    """用 Gemini 分析圖片，回傳結構化 JSON"""
    if mock:
        return _mock_analyze(image_path)

    import google.generativeai as genai
    from PIL import Image

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("錯誤：找不到 GOOGLE_API_KEY，改用 mock 模式")
        return _mock_analyze(image_path)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    image = Image.open(image_path)

    prompt = """分析這張工地照片，輸出 JSON 格式：
{
    "person_count": 人數,
    "violations": [
        {
            "person_id": 編號,
            "type": "違規類型（如 no_helmet, no_vest, blocked_exit）",
            "severity": "high/medium/low"
        }
    ],
    "recommendations": ["建議1", "建議2"]
}
只輸出 JSON，不要其他文字。"""

    response = model.generate_content([prompt, image])
    text = response.text.strip()

    # 移除 markdown 標記
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

    return json.loads(text)


def _mock_analyze(image_path):
    """模擬 Gemini 分析結果（不需要 API Key）"""
    filename = os.path.basename(image_path)
    # 用檔名產生不同結果
    h = sum(ord(c) for c in filename) % 3
    mock_results = [
        {
            "person_count": 3,
            "violations": [
                {"person_id": 2, "type": "no_helmet", "severity": "high"}
            ],
            "recommendations": ["加強安全帽佩戴"],
        },
        {
            "person_count": 2,
            "violations": [
                {"person_id": 1, "type": "no_vest", "severity": "medium"},
                {"person_id": 2, "type": "no_vest", "severity": "low"},
            ],
            "recommendations": ["補充反光背心"],
        },
        {
            "person_count": 4,
            "violations": [
                {"person_id": 3, "type": "no_vest", "severity": "medium"}
            ],
            "recommendations": ["加強入場檢查"],
        },
    ]
    return mock_results[h]


# === Gemini JSON → Event ===

def gemini_to_events(gemini_result, image_id="unknown"):
    """把 Gemini 分析結果轉成 Event list"""
    events = []
    timestamp = datetime.now().isoformat()

    for violation in gemini_result.get("violations", []):
        event = {
            "source": "gemini",
            "image_id": image_id,
            "timestamp": timestamp,
            "event_type": violation["type"],
            "severity": violation["severity"],
            "person_id": violation.get("person_id"),
        }
        events.append(event)

    return events


# === 系統狀態管理 ===

class SafetyState:
    def __init__(self):
        self.violations = []
        self.alert_count = 0
        self.last_analysis = None

    def update(self, events):
        """收到新事件，更新狀態"""
        self.violations.extend(events)
        self.last_analysis = datetime.now()

        for event in events:
            if self.should_alert(event):
                self.trigger_alert(event)

    def should_alert(self, event):
        """判斷是否觸發告警"""
        # 嚴重違規 → 立即告警
        if event["severity"] == "high":
            return True

        # 同類型違規累積超過 3 次 → 告警
        same_type = [v for v in self.violations if v["event_type"] == event["event_type"]]
        if len(same_type) >= 3:
            return True

        return False

    def trigger_alert(self, event):
        """觸發告警"""
        self.alert_count += 1
        print(f"  🚨 告警 #{self.alert_count}: {event['event_type']} ({event['severity']})")

    def get_summary(self):
        """取得狀態摘要"""
        counts = {}
        for v in self.violations:
            t = v["event_type"]
            counts[t] = counts.get(t, 0) + 1

        return {
            "total_violations": len(self.violations),
            "alert_count": self.alert_count,
            "by_type": counts,
        }


# === 主程式 ===

if __name__ == "__main__":
    use_mock = "--mock" in sys.argv

    print("=" * 50)
    print("  感知結果如何進入系統狀態")
    print("=" * 50)

    if use_mock:
        print("  模式: mock（不呼叫 Gemini）")
    else:
        print("  模式: Gemini API")

    state = SafetyState()

    # 分析圖片
    images = ["workplace.jpg", "workplace.jpg", "workplace.jpg"]

    if use_mock:
        # mock 模式用不同檔名產生不同結果
        images = ["cam01_001.jpg", "cam01_002.jpg", "cam01_003.jpg"]

    for image_path in images:
        print(f"\n[Gemini] 分析 {image_path}...")

        # 1. Gemini 分析圖片 → JSON
        gemini_result = analyze_image(image_path, mock=use_mock)
        violation_count = len(gemini_result.get("violations", []))
        print(f"  結果: {gemini_result.get('person_count', '?')} 人, {violation_count} 個違規")

        # 2. JSON → Event
        events = gemini_to_events(gemini_result, image_id=image_path)

        # 3. Event → 更新系統狀態
        state.update(events)

    # 4. 摘要
    summary = state.get_summary()
    print(f"\n{'=' * 50}")
    print(f"  系統狀態")
    print(f"{'=' * 50}")
    print(f"  總違規: {summary['total_violations']}")
    print(f"  告警數: {summary['alert_count']}")
    print(f"  違規分布:")
    for event_type, count in summary["by_type"].items():
        print(f"    {event_type}: {count}")
