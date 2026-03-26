import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# 1. 載入環境變數 (確保你的目錄下有 .env 檔案，且內容為 GOOGLE_API_KEY=你的KEY)
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("錯誤：找不到 API Key，請檢查 .env 檔案或環境變數。")
else:
    genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

try:
    image = Image.open("workplace.jpg")

    response = model.generate_content([
        "描述這張工地照片中的安全狀況，請用繁體中文回答。",
        image
    ])

    print("--- 辨識結果 ---")
    print(response.text)

except FileNotFoundError:
    print("錯誤：找不到 workplace.jpg 檔案。")
except Exception as e:
    print(f"發生錯誤：{e}")
