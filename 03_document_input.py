import os
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("錯誤：找不到 API Key，請檢查 .env 檔案或環境變數。")
else:
    genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

# 上傳 PDF
pdf_file = genai.upload_file("safety_regulation.pdf")

# 分析 PDF
response = model.generate_content([
    "請摘要這份工安法規的重點，特別是關於 PPE 的規定",
    pdf_file
])

print(response.text)
