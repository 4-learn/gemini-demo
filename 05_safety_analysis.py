"""
Workshop 解答：感知結果如何進入系統狀態

題目：
1. 設計感知結果的 Event 格式
2. 實作狀態更新邏輯
3. 實作觸發條件（如：嚴重違規 → 立即告警）

執行方式：
python 09_perception_state.py
"""

from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


def main():
    print("=== 感知結果進入系統狀態 ===\n")

    # === 1. 設計 Event 格式 ===

    print("1. 感知事件（Event）格式設計")
    print("-" * 50)

    class EventType(str, Enum):
        VIOLATION_DETECTED = "violation_detected"
        PERSON_ENTERED = "person_entered"
        PERSON_LEFT = "person_left"
        EQUIPMENT_ANOMALY = "equipment_anomaly"
        ENVIRONMENT_ALERT = "environment_alert"

    class RiskLevel(str, Enum):
        CRITICAL = "critical"  # 立即危險
        HIGH = "high"          # 高風險
        MEDIUM = "medium"      # 中風險
        LOW = "low"            # 低風險
        INFO = "info"          # 資訊性

    @dataclass
    class PerceptionEvent:
        """感知事件資料結構"""
        event_id: str
        event_type: EventType
        timestamp: datetime
        source: str  # 來源（如攝影機編號）
        zone: str    # 區域
        risk_level: RiskLevel
        description: str
        details: dict = field(default_factory=dict)
        requires_action: bool = False
        action_deadline: Optional[datetime] = None

        def to_json(self) -> str:
            return json.dumps({
                "event_id": self.event_id,
                "event_type": self.event_type.value,
                "timestamp": self.timestamp.isoformat(),
                "source": self.source,
                "zone": self.zone,
                "risk_level": self.risk_level.value,
                "description": self.description,
                "details": self.details,
                "requires_action": self.requires_action
            }, ensure_ascii=False, indent=2)

    # 範例事件
    sample_event = PerceptionEvent(
        event_id="EVT-20240115-001",
        event_type=EventType.VIOLATION_DETECTED,
        timestamp=datetime.now(),
        source="CAM-A01",
        zone="施工區A",
        risk_level=RiskLevel.HIGH,
        description="偵測到人員未配戴安全帽",
        details={
            "person_count": 1,
            "violation_type": "no_helmet",
            "confidence": 0.95,
            "image_url": "s3://bucket/images/evt-001.jpg"
        },
        requires_action=True
    )

    print("   Event 格式範例：")
    print(sample_event.to_json())

    # === 2. 系統狀態管理 ===

    print("\n2. 系統狀態管理")
    print("-" * 50)

    @dataclass
    class ZoneState:
        """區域狀態"""
        zone_id: str
        zone_name: str
        person_count: int = 0
        active_violations: list = field(default_factory=list)
        risk_level: RiskLevel = RiskLevel.INFO
        last_updated: datetime = field(default_factory=datetime.now)

    class SystemState:
        """系統整體狀態"""

        def __init__(self):
            self.zones: dict[str, ZoneState] = {}
            self.event_history: list[PerceptionEvent] = []
            self.active_alerts: list[dict] = []
            self.listeners: list[Callable] = []

        def register_listener(self, callback: Callable):
            """註冊事件監聽器"""
            self.listeners.append(callback)

        def notify_listeners(self, event: PerceptionEvent):
            """通知所有監聽器"""
            for listener in self.listeners:
                listener(event)

        def update_from_event(self, event: PerceptionEvent) -> dict:
            """
            根據感知事件更新系統狀態

            Args:
                event: 感知事件

            Returns:
                dict: 狀態變更摘要
            """
            changes = {
                "timestamp": datetime.now().isoformat(),
                "event_id": event.event_id,
                "actions": []
            }

            # 確保區域存在
            if event.zone not in self.zones:
                self.zones[event.zone] = ZoneState(
                    zone_id=event.zone,
                    zone_name=event.zone
                )

            zone = self.zones[event.zone]

            # 根據事件類型更新狀態
            if event.event_type == EventType.VIOLATION_DETECTED:
                zone.active_violations.append({
                    "event_id": event.event_id,
                    "description": event.description,
                    "risk_level": event.risk_level.value,
                    "timestamp": event.timestamp.isoformat()
                })
                changes["actions"].append(f"新增違規記錄：{event.description}")

                # 更新區域風險等級
                if event.risk_level.value in ["critical", "high"]:
                    zone.risk_level = event.risk_level
                    changes["actions"].append(f"區域風險等級升級為：{event.risk_level.value}")

            elif event.event_type == EventType.PERSON_ENTERED:
                zone.person_count += 1
                changes["actions"].append(f"人員進入，當前人數：{zone.person_count}")

            elif event.event_type == EventType.PERSON_LEFT:
                zone.person_count = max(0, zone.person_count - 1)
                changes["actions"].append(f"人員離開，當前人數：{zone.person_count}")

            zone.last_updated = datetime.now()
            self.event_history.append(event)

            # 通知監聽器
            self.notify_listeners(event)

            return changes

        def get_zone_status(self, zone_id: str) -> dict:
            """取得區域狀態"""
            if zone_id not in self.zones:
                return {"error": "Zone not found"}

            zone = self.zones[zone_id]
            return {
                "zone_id": zone.zone_id,
                "zone_name": zone.zone_name,
                "person_count": zone.person_count,
                "active_violations_count": len(zone.active_violations),
                "risk_level": zone.risk_level.value,
                "last_updated": zone.last_updated.isoformat()
            }

    # 建立系統狀態
    system = SystemState()
    print("   系統狀態管理器已初始化")

    # === 3. 觸發條件與告警 ===

    print("\n3. 觸發條件與告警邏輯")
    print("-" * 50)

    class AlertManager:
        """告警管理器"""

        def __init__(self, system_state: SystemState):
            self.system = system_state
            self.alert_rules = []
            # 註冊為事件監聽器
            system_state.register_listener(self.on_event)

        def add_rule(self, rule: dict):
            """新增告警規則"""
            self.alert_rules.append(rule)

        def on_event(self, event: PerceptionEvent):
            """事件回調處理"""
            self.evaluate_rules(event)

        def evaluate_rules(self, event: PerceptionEvent):
            """評估告警規則"""
            for rule in self.alert_rules:
                if self.check_condition(event, rule["condition"]):
                    self.trigger_alert(event, rule)

        def check_condition(self, event: PerceptionEvent, condition: dict) -> bool:
            """檢查是否符合觸發條件"""
            # 檢查風險等級
            if "risk_level" in condition:
                if event.risk_level.value not in condition["risk_level"]:
                    return False

            # 檢查事件類型
            if "event_type" in condition:
                if event.event_type.value not in condition["event_type"]:
                    return False

            # 檢查區域
            if "zones" in condition:
                if event.zone not in condition["zones"]:
                    return False

            return True

        def trigger_alert(self, event: PerceptionEvent, rule: dict):
            """觸發告警"""
            alert = {
                "alert_id": f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "rule_name": rule["name"],
                "event_id": event.event_id,
                "timestamp": datetime.now().isoformat(),
                "actions": rule["actions"]
            }

            self.system.active_alerts.append(alert)

            print(f"\n   ⚠️  告警觸發！")
            print(f"   規則：{rule['name']}")
            print(f"   事件：{event.description}")
            print(f"   動作：{', '.join(rule['actions'])}")

    # 建立告警管理器並設定規則
    alert_manager = AlertManager(system)

    # 規則 1：高風險違規立即告警
    alert_manager.add_rule({
        "name": "高風險違規即時告警",
        "condition": {
            "risk_level": ["critical", "high"],
            "event_type": ["violation_detected"]
        },
        "actions": ["notify_supervisor", "send_sms", "log_incident"]
    })

    # 規則 2：危險區域人員告警
    alert_manager.add_rule({
        "name": "危險區域人員監控",
        "condition": {
            "event_type": ["person_entered"],
            "zones": ["化學區", "高壓區"]
        },
        "actions": ["notify_security", "start_monitoring"]
    })

    print("   已設定 2 條告警規則")

    # === 4. 完整流程測試 ===

    print("\n4. 完整流程測試")
    print("-" * 50)

    # 模擬事件流
    test_events = [
        PerceptionEvent(
            event_id="EVT-001",
            event_type=EventType.PERSON_ENTERED,
            timestamp=datetime.now(),
            source="CAM-A01",
            zone="施工區A",
            risk_level=RiskLevel.INFO,
            description="偵測到 3 人進入施工區A",
            details={"person_count": 3}
        ),
        PerceptionEvent(
            event_id="EVT-002",
            event_type=EventType.VIOLATION_DETECTED,
            timestamp=datetime.now(),
            source="CAM-A01",
            zone="施工區A",
            risk_level=RiskLevel.HIGH,
            description="偵測到人員未配戴安全帽",
            details={"violation_type": "no_helmet"},
            requires_action=True
        ),
        PerceptionEvent(
            event_id="EVT-003",
            event_type=EventType.VIOLATION_DETECTED,
            timestamp=datetime.now(),
            source="CAM-B02",
            zone="倉庫區",
            risk_level=RiskLevel.MEDIUM,
            description="偵測到通道堆放雜物",
            details={"violation_type": "blocked_path"}
        )
    ]

    print("\n   事件處理流程：")
    print("   " + "=" * 45)

    for event in test_events:
        print(f"\n   [{event.timestamp.strftime('%H:%M:%S')}] 收到事件：{event.description}")
        changes = system.update_from_event(event)
        for action in changes["actions"]:
            print(f"   [{event.timestamp.strftime('%H:%M:%S')}] {action}")

    # 顯示最終狀態
    print("\n   " + "=" * 45)
    print("   最終系統狀態：")
    for zone_id in system.zones:
        status = system.get_zone_status(zone_id)
        print(f"\n   【{status['zone_name']}】")
        print(f"   - 人數：{status['person_count']}")
        print(f"   - 違規數：{status['active_violations_count']}")
        print(f"   - 風險等級：{status['risk_level']}")

    print(f"\n   活動告警數：{len(system.active_alerts)}")

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
