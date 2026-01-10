"""
多模態輸入轉結構化資料

學習目標：
1. 設計結構化輸出的 Prompt
2. JSON Schema 約束
3. 錯誤處理與驗證
4. 與 Pydantic 整合
"""

import os
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai

# 載入環境變數
load_dotenv()


def setup_gemini():
    """設定 Gemini API"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("錯誤：請設定 GOOGLE_API_KEY 環境變數")
        return None
    genai.configure(api_key=api_key)
    return True


def demo_basic_json():
    """基本 JSON 輸出"""
    print("=== 基本 JSON 輸出 ===\n")

    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = """分析以下違規描述，以 JSON 格式輸出：

描述：施工區 A 棟入口處發現一名工人未配戴安全帽，該工人正在搬運建材。

輸出格式：
```json
{
    "violation_type": "違規類型",
    "location": "地點",
    "activity": "正在進行的活動",
    "severity": "high/medium/low"
}
```

只輸出 JSON，不要其他文字。"""

    response = model.generate_content(prompt)

    print("原始輸出：")
    print(response.text)

    # 解析 JSON
    try:
        # 清理可能的 markdown 標記
        json_str = response.text.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]

        result = json.loads(json_str.strip())
        print("\n解析後的 Python 物件：")
        print(result)
    except json.JSONDecodeError as e:
        print(f"\nJSON 解析失敗: {e}")


def demo_json_schema():
    """使用 JSON Schema 約束"""
    print("\n=== JSON Schema 約束 ===\n")

    model = genai.GenerativeModel("gemini-1.5-flash")

    schema = {
        "type": "object",
        "properties": {
            "violation_id": {"type": "string", "pattern": "^VIO-[0-9]{4}$"},
            "timestamp": {"type": "string", "format": "date-time"},
            "location": {
                "type": "object",
                "properties": {
                    "zone": {"type": "string", "enum": ["construction", "office", "entrance", "warehouse"]},
                    "building": {"type": "string"},
                    "floor": {"type": "integer", "minimum": -2, "maximum": 50}
                },
                "required": ["zone"]
            },
            "violations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["no_helmet", "no_vest", "no_safety_belt", "other"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["type", "confidence"]
                }
            },
            "severity": {"type": "string", "enum": ["high", "medium", "low"]},
            "action_required": {"type": "boolean"}
        },
        "required": ["violation_id", "timestamp", "location", "violations", "severity"]
    }

    prompt = f"""你是一個 JSON 生成器。根據以下描述生成符合 schema 的 JSON。

Schema:
{json.dumps(schema, indent=2)}

描述：
今天下午 3 點，在 B 棟 12 樓施工區，發現兩名工人違規。
一人未戴安全帽（信心度 92%），一人未繫安全帶（信心度 85%）。
情況嚴重，需要立即處理。

只輸出 JSON，不要其他文字。"""

    response = model.generate_content(prompt)

    print("Schema 約束的輸出：")
    print(response.text)


def demo_pydantic_integration():
    """與 Pydantic 整合"""
    print("\n=== Pydantic 整合 ===\n")

    try:
        from pydantic import BaseModel, Field, field_validator
        from typing import List

        # 定義資料模型
        class ViolationItem(BaseModel):
            type: str = Field(..., description="違規類型")
            confidence: float = Field(..., ge=0, le=1)

        class Location(BaseModel):
            zone: str
            building: Optional[str] = None
            floor: Optional[int] = None

        class ViolationReport(BaseModel):
            violation_id: str
            timestamp: datetime
            location: Location
            violations: List[ViolationItem]
            severity: str = Field(..., pattern="^(high|medium|low)$")
            action_required: bool = True

            @field_validator('violation_id')
            @classmethod
            def validate_id(cls, v):
                if not v.startswith("VIO-"):
                    raise ValueError("ID must start with VIO-")
                return v

        print("Pydantic 模型定義完成")
        print()

        # 使用範例
        print("使用方式：")
        print("""
# 從 Gemini 取得 JSON
response = model.generate_content(prompt)
json_str = response.text

# 用 Pydantic 驗證
try:
    report = ViolationReport.model_validate_json(json_str)
    print(f"驗證成功: {report.violation_id}")
except ValidationError as e:
    print(f"驗證失敗: {e}")
""")

        # 示範 Pydantic 驗證
        valid_json = '''{
    "violation_id": "VIO-0001",
    "timestamp": "2024-01-15T15:30:00",
    "location": {"zone": "construction", "building": "B棟", "floor": 12},
    "violations": [
        {"type": "no_helmet", "confidence": 0.92},
        {"type": "no_safety_belt", "confidence": 0.85}
    ],
    "severity": "high",
    "action_required": true
}'''

        report = ViolationReport.model_validate_json(valid_json)
        print(f"驗證成功！")
        print(f"  違規 ID: {report.violation_id}")
        print(f"  地點: {report.location.zone} - {report.location.building}")
        print(f"  違規數: {len(report.violations)}")

    except ImportError:
        print("需要安裝 pydantic: pip install pydantic")


def demo_error_handling():
    """錯誤處理"""
    print("\n=== 錯誤處理 ===\n")

    def safe_parse_json(text: str) -> dict:
        """安全地解析 JSON，處理常見問題"""

        # 移除可能的 markdown 標記
        text = text.strip()

        if text.startswith("```"):
            # 提取 ``` 之間的內容
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                # 移除語言標記（如 json）
                if text.startswith("json"):
                    text = text[4:]
                elif text.startswith("JSON"):
                    text = text[4:]
            text = text.strip()

        # 嘗試解析
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"JSON 解析錯誤: {e}")

            # 嘗試修復常見問題
            # 1. 單引號改雙引號
            text = text.replace("'", '"')

            # 2. 移除尾部逗號
            import re
            text = re.sub(r',\s*}', '}', text)
            text = re.sub(r',\s*]', ']', text)

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"error": "無法解析", "raw": text}

    # 測試各種格式
    test_cases = [
        '{"name": "test", "value": 123}',  # 正常
        "```json\n{\"name\": \"test\"}\n```",  # Markdown
        "{'name': 'test'}",  # 單引號
        '{"name": "test", "items": [1, 2, 3,]}',  # 尾部逗號
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"測試 {i}:")
        print(f"  輸入: {test[:40]}...")
        result = safe_parse_json(test)
        print(f"  結果: {result}")
        print()


def demo_retry_strategy():
    """重試策略"""
    print("=== 重試策略 ===\n")

    print("""
def generate_with_retry(model, prompt, max_retries=3):
    '''帶重試的 JSON 生成'''

    for attempt in range(max_retries):
        response = model.generate_content(prompt)

        try:
            result = safe_parse_json(response.text)
            if "error" not in result:
                return result
        except Exception:
            pass

        # 如果失敗，加強提示
        if attempt < max_retries - 1:
            prompt = f'''
{prompt}

注意：請只輸出有效的 JSON，不要包含任何其他文字或 markdown 標記。
確保：
- 使用雙引號
- 沒有尾部逗號
- 所有括號配對正確
'''

    return {"error": "多次重試後仍無法生成有效 JSON"}

# 使用
result = generate_with_retry(model, prompt)
if "error" in result:
    # fallback 處理
    pass
""")


def demo_complex_extraction():
    """複雜資訊提取"""
    print("\n=== 複雜資訊提取 ===\n")

    model = genai.GenerativeModel("gemini-1.5-flash")

    # 複雜的違規報告文字
    report_text = """
今日（2024/1/15）安全巡查報告

早班巡查（08:00-12:00）：
- 09:15 在 A 區 3 樓發現承包商「建興工程」的兩名員工（工號 B-001、B-002）
  未配戴安全帽進行天花板作業，已口頭警告並要求改正。
- 10:30 在 B 區地下室停車場，發現一台堆高機未設置警示標誌。已通知管理員處理。
- 11:45 C 區 1 樓施工圍籬有破損，可能造成人員誤入，建議盡速修復。

午班巡查（13:00-17:00）：
- 14:20 A 區 3 樓已確認改善，員工已配戴安全帽。
- 15:00 D 區 5 樓高空作業，發現一名員工未繫安全帶，情況危急，
  已立即制止並通報工地主任。員工資訊：李明（工號 D-042，「永興工程」）。
- 16:30 全區複查完畢，無其他異常。

本日總結：
- 違規件數：3 件
- 已改善：1 件
- 待改善：2 件
- 需立即處理：1 件（D 區高空作業違規）

報告人：張安全
"""

    prompt = f"""分析以下安全巡查報告，提取所有違規事件，以 JSON 格式輸出。

報告內容：
{report_text}

輸出格式：
```json
{{
    "report_date": "報告日期",
    "reporter": "報告人",
    "violations": [
        {{
            "id": "編號 (V001, V002...)",
            "time": "發現時間",
            "location": {{
                "zone": "區域",
                "floor": "樓層",
                "detail": "詳細位置"
            }},
            "type": "違規類型",
            "description": "描述",
            "involved_parties": [
                {{
                    "name": "姓名（如有）",
                    "employee_id": "工號（如有）",
                    "company": "公司（如有）"
                }}
            ],
            "severity": "high/medium/low",
            "status": "pending/resolved/in_progress",
            "action_taken": "已採取措施"
        }}
    ],
    "summary": {{
        "total_violations": 總違規數,
        "resolved": 已改善數,
        "pending": 待改善數,
        "urgent": 需立即處理數
    }}
}}
```

只輸出 JSON。"""

    response = model.generate_content(prompt)

    print("複雜報告提取結果：")
    print(response.text)


# === 主程式 ===
if __name__ == "__main__":
    print("=" * 50)
    print("多模態輸入轉結構化資料")
    print("=" * 50)
    print()

    if not setup_gemini():
        exit(1)

    # 1. 基本 JSON
    demo_basic_json()

    # 2. JSON Schema
    demo_json_schema()

    # 3. Pydantic 整合
    demo_pydantic_integration()

    # 4. 錯誤處理
    demo_error_handling()

    # 5. 重試策略
    demo_retry_strategy()

    # 6. 複雜提取
    demo_complex_extraction()

    print("\n完成！")
