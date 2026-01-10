"""
工安圖像分析範例

學習目標：
1. 整合多模態輸入
2. 建立工安分析 Pipeline
3. 結構化輸出與系統整合
"""

import os
import json
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import google.generativeai as genai

# 載入環境變數
load_dotenv()


# === 資料結構 ===

@dataclass
class AnalysisResult:
    """分析結果"""
    timestamp: str
    source: str
    scene_description: str
    people_count: int
    violations: list
    safety_score: int
    recommendations: list
    raw_response: Optional[str] = None


# === 工安分析器 ===

class SafetyAnalyzer:
    """
    工安圖像分析器

    整合 Gemini 多模態能力，分析工地圖像中的安全狀況
    """

    SYSTEM_INSTRUCTION = """你是一位專業的工安檢查員 AI 助手。
你的職責是分析工地照片，找出安全違規和潛在風險。

分析時請注意：
1. 個人防護裝備（PPE）：安全帽、反光背心、安全鞋、護目鏡、手套
2. 高空作業安全：安全帶、鷹架穩固性、護欄
3. 環境危險：雜物堆積、通道阻塞、照明不足
4. 設備安全：機具狀況、警示標誌

請以專業、客觀的角度分析，並提供具體的改善建議。"""

    ANALYSIS_PROMPT = """分析這張工地照片的安全狀況。

請以 JSON 格式輸出，格式如下：
```json
{
    "scene_description": "場景描述（一句話）",
    "people_count": 人數,
    "violations": [
        {
            "type": "違規類型（如 no_helmet, no_vest, blocked_exit）",
            "description": "具體描述",
            "location_in_image": "在圖片中的位置",
            "severity": "high/medium/low",
            "regulation": "相關法規（如有）"
        }
    ],
    "safety_score": 0-100 的安全評分,
    "recommendations": ["建議1", "建議2"]
}
```

如果圖片不是工地或無法分析，請回傳：
```json
{
    "error": "無法分析",
    "reason": "原因"
}
```

只輸出 JSON。"""

    def __init__(self):
        """初始化分析器"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("請設定 GOOGLE_API_KEY 環境變數")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=self.SYSTEM_INSTRUCTION
        )

    def analyze_image(self, image, source: str = "unknown") -> AnalysisResult:
        """
        分析圖像

        Args:
            image: PIL Image 或 genai.File 物件
            source: 圖像來源（如攝影機 ID）

        Returns:
            AnalysisResult 物件
        """
        # 呼叫 Gemini
        response = self.model.generate_content([
            self.ANALYSIS_PROMPT,
            image
        ])

        # 解析回應
        result_dict = self._parse_response(response.text)

        # 建立結果物件
        return AnalysisResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            scene_description=result_dict.get("scene_description", ""),
            people_count=result_dict.get("people_count", 0),
            violations=result_dict.get("violations", []),
            safety_score=result_dict.get("safety_score", 0),
            recommendations=result_dict.get("recommendations", []),
            raw_response=response.text
        )

    def analyze_with_context(self, image, context: str, source: str = "unknown") -> AnalysisResult:
        """
        帶上下文的分析

        Args:
            image: 圖像
            context: 額外上下文（如：這是施工區、這是入口處）
            source: 圖像來源
        """
        prompt = f"""分析這張工地照片的安全狀況。

額外資訊：{context}

{self.ANALYSIS_PROMPT}"""

        response = self.model.generate_content([prompt, image])
        result_dict = self._parse_response(response.text)

        return AnalysisResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            scene_description=result_dict.get("scene_description", ""),
            people_count=result_dict.get("people_count", 0),
            violations=result_dict.get("violations", []),
            safety_score=result_dict.get("safety_score", 0),
            recommendations=result_dict.get("recommendations", []),
            raw_response=response.text
        )

    def _parse_response(self, text: str) -> dict:
        """解析 Gemini 回應"""
        text = text.strip()

        # 移除 markdown 標記
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "error": "JSON 解析失敗",
                "raw": text
            }


# === 批次處理器 ===

class BatchAnalyzer:
    """批次圖像分析"""

    def __init__(self, analyzer: SafetyAnalyzer):
        self.analyzer = analyzer
        self.results = []

    def process_folder(self, folder_path: str) -> list:
        """處理資料夾中的所有圖像"""
        from pathlib import Path
        from PIL import Image
        import time

        folder = Path(folder_path)
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

        print(f"找到 {len(image_files)} 張圖像")

        for img_path in image_files:
            print(f"分析: {img_path.name}")

            try:
                img = Image.open(img_path)
                result = self.analyzer.analyze_image(img, source=img_path.name)
                self.results.append(result)
            except Exception as e:
                print(f"  錯誤: {e}")

            # 避免超過 API 限制
            time.sleep(1)

        return self.results

    def generate_report(self) -> dict:
        """生成分析報告"""
        if not self.results:
            return {"error": "沒有分析結果"}

        total_violations = []
        total_people = 0
        scores = []

        for result in self.results:
            total_violations.extend(result.violations)
            total_people += result.people_count
            scores.append(result.safety_score)

        # 統計違規類型
        violation_types = {}
        for v in total_violations:
            vtype = v.get("type", "unknown")
            violation_types[vtype] = violation_types.get(vtype, 0) + 1

        return {
            "summary": {
                "images_analyzed": len(self.results),
                "total_people": total_people,
                "total_violations": len(total_violations),
                "average_safety_score": sum(scores) / len(scores) if scores else 0,
            },
            "violation_breakdown": violation_types,
            "high_severity": [
                v for v in total_violations
                if v.get("severity") == "high"
            ],
            "all_recommendations": list(set(
                r for result in self.results
                for r in result.recommendations
            ))
        }


# === 即時監控模擬 ===

def simulate_realtime_monitoring():
    """模擬即時監控流程"""
    print("=== 即時監控模擬 ===\n")

    print("""
# 即時監控流程

from PIL import Image
import time

analyzer = SafetyAnalyzer()

# 模擬從攝影機取得影像
def get_camera_frame(camera_id):
    # 實際應用會連接到攝影機
    return Image.open(f"camera_{camera_id}.jpg")

# 回調函數：處理違規
def on_violation(result, violation):
    print(f"[警報] {violation['type']} - {violation['description']}")
    # 發送通知、寫入資料庫等

# 監控迴圈
cameras = ["cam_01", "cam_02", "cam_03"]

while True:
    for cam_id in cameras:
        frame = get_camera_frame(cam_id)
        result = analyzer.analyze_image(frame, source=cam_id)

        # 檢查違規
        for violation in result.violations:
            if violation.get("severity") == "high":
                on_violation(result, violation)

        # 輸出狀態
        print(f"[{cam_id}] Score: {result.safety_score}, Violations: {len(result.violations)}")

    time.sleep(30)  # 每 30 秒分析一次
""")


# === 與系統整合 ===

def demo_system_integration():
    """系統整合範例"""
    print("\n=== 系統整合 ===\n")

    print("""
# 與工安監控系統整合

from event.schema import PPEDetectionEvent
from event.rules import RuleEngine

class GeminiPerception:
    '''Gemini 感知層'''

    def __init__(self):
        self.analyzer = SafetyAnalyzer()

    def process_frame(self, frame, camera_id: str) -> list[PPEDetectionEvent]:
        '''處理影格，輸出事件'''

        result = self.analyzer.analyze_image(frame, source=camera_id)

        events = []
        for violation in result.violations:
            # 轉換為系統事件格式
            event = PPEDetectionEvent(
                timestamp=datetime.now(timezone.utc),
                object=violation['type'],
                confidence=self._severity_to_confidence(violation['severity']),
                bbox=[0, 0, 100, 100],  # Gemini 不提供精確 bbox
                source=camera_id,
                metadata={
                    "description": violation['description'],
                    "gemini_raw": violation,
                    "safety_score": result.safety_score,
                }
            )
            events.append(event)

        return events

    def _severity_to_confidence(self, severity: str) -> float:
        return {"high": 0.9, "medium": 0.7, "low": 0.5}.get(severity, 0.5)


# 使用
perception = GeminiPerception()
events = perception.process_frame(camera_frame, "camera_01")

# 送入規則引擎
for event in events:
    result = rule_engine.evaluate(event)
    if result.matched:
        alert_manager.create_alert(result)
""")


# === 主程式 ===

def main():
    print("=" * 50)
    print("工安圖像分析範例")
    print("=" * 50)
    print()

    # 檢查 API Key
    if not os.getenv("GOOGLE_API_KEY"):
        print("錯誤：請設定 GOOGLE_API_KEY 環境變數")
        print("取得 API Key: https://makersuite.google.com/app/apikey")
        print()
        print("以下是程式碼示範（不執行實際分析）：")
        print()

    # 展示類別結構
    print("=== SafetyAnalyzer 類別 ===\n")
    print("SafetyAnalyzer 是核心分析器，用法：")
    print("""
from PIL import Image

analyzer = SafetyAnalyzer()
image = Image.open("construction_site.jpg")
result = analyzer.analyze_image(image, source="camera_01")

print(f"安全評分: {result.safety_score}")
print(f"違規數量: {len(result.violations)}")
for v in result.violations:
    print(f"  - {v['type']}: {v['description']}")
""")

    # 模擬即時監控
    simulate_realtime_monitoring()

    # 系統整合
    demo_system_integration()

    print("\n完成！")


if __name__ == "__main__":
    main()
