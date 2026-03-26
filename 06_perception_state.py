"""
Demo：感知結果如何進入系統狀態

流程：
  Gemini 輸出 JSON → 轉成 Event → 更新系統狀態 → 觸發告警

執行方式：
  python 06_perception_state.py

不需要 Gemini API Key（用模擬資料）
"""

from datetime import datetime


# === Step 1：Gemini JSON → Event ===

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


# === Step 2：系統狀態管理 ===

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


# === Demo ===

if __name__ == "__main__":
    print("=" * 50)
    print("  感知結果如何進入系統狀態")
    print("=" * 50)

    state = SafetyState()

    # 模擬 3 次 Gemini 分析結果
    analyses = [
        {
            "image_id": "cam01_001.jpg",
            "result": {
                "person_count": 3,
                "violations": [
                    {"type": "no_helmet", "severity": "high", "person_id": 2}
                ],
            },
        },
        {
            "image_id": "cam01_002.jpg",
            "result": {
                "person_count": 2,
                "violations": [
                    {"type": "no_vest", "severity": "medium", "person_id": 1},
                    {"type": "no_vest", "severity": "low", "person_id": 2},
                ],
            },
        },
        {
            "image_id": "cam01_003.jpg",
            "result": {
                "person_count": 4,
                "violations": [
                    {"type": "no_vest", "severity": "medium", "person_id": 3}
                ],
            },
        },
    ]

    for analysis in analyses:
        image_id = analysis["image_id"]
        gemini_result = analysis["result"]
        violation_count = len(gemini_result["violations"])

        print(f"\n[分析] {image_id} → {violation_count} 個違規")

        # 1. Gemini JSON → Event
        events = gemini_to_events(gemini_result, image_id=image_id)

        # 2. 更新狀態（內部會判斷是否告警）
        state.update(events)

    # 3. 摘要
    summary = state.get_summary()
    print(f"\n{'=' * 50}")
    print(f"  系統狀態")
    print(f"{'=' * 50}")
    print(f"  總違規: {summary['total_violations']}")
    print(f"  告警數: {summary['alert_count']}")
    print(f"  違規分布:")
    for event_type, count in summary["by_type"].items():
        print(f"    {event_type}: {count}")
