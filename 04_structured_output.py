import json
import os
from PIL import Image
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("錯誤：找不到 API Key，請檢查 .env 檔案或環境變數。")
else:
    genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

image = Image.open("workplace.jpg")

prompt = """
分析這張工地照片，輸出 JSON 格式：
{
    "person_count": 人數,
    "violations": [
        {
            "person_id": 編號,
            "type": "違規類型",
            "severity": "low/medium/high"
        }
    ],
    "recommendations": ["建議1", "建議2"]
}
只輸出 JSON，不要其他文字。
"""

response = model.generate_content([prompt, image])
print(response.text)
result = json.loads(response.text)
