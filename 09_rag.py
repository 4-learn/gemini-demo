"""
Demo：RAG（Retrieval-Augmented Generation）

法規有 200 頁，不可能全塞進 prompt。
RAG 的做法：先搜尋相關段落，只把相關的塞進 prompt。

執行方式：
  python 09_rag.py

需要：
  pip install google-genai scikit-learn python-dotenv numpy
  .env 裡設定 GOOGLE_API_KEY
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# === 法規資料（模擬切段後的結果） ===

REGULATION_CHUNKS = [
    {"id": "reg_001", "title": "安全帽規定",
     "content": "依據職業安全衛生設施規則第 281 條，雇主對於在高度 2 公尺以上之工作場所，應使勞工確實使用安全帽。安全帽應符合 CNS 國家標準。"},
    {"id": "reg_002", "title": "反光背心規定",
     "content": "依據職業安全衛生設施規則第 21 條，雇主應提供適當之反光標示或背心。於夜間或光線不足之場所作業，應提供高可見度服裝。"},
    {"id": "reg_003", "title": "護目鏡規定",
     "content": "依據職業安全衛生設施規則第 287 條，從事焊接、切割或研磨作業時，應配戴護目鏡或面罩，以防止飛濺物傷害眼睛。"},
    {"id": "reg_004", "title": "安全出口規定",
     "content": "依據建築技術規則第 97 條，安全出口寬度不得小於 1.2 公尺，不得堆放物品阻礙通行。緊急出口應設置明顯標示。"},
    {"id": "reg_005", "title": "高空作業安全帶",
     "content": "依據職業安全衛生設施規則第 281 條之 1，從事高度 2 公尺以上之高空作業，應使用安全帶或安全母索，並確認錨定點之強度。"},
    {"id": "reg_006", "title": "化學品標示",
     "content": "依據危害性化學品標示及通識規則第 5 條，雇主應於化學品容器上標示名稱、危害圖式、警示語及危害防範措施。"},
    {"id": "reg_007", "title": "噪音防護",
     "content": "依據職業安全衛生設施規則第 300 條，工作場所噪音超過 85 分貝時，雇主應提供耳塞或耳罩等聽力防護具。"},
    {"id": "reg_008", "title": "消防設備",
     "content": "依據消防法第 6 條，各類場所應設置滅火器、室內消防栓等消防安全設備，並定期檢修申報。"},
]


# === Embedding ===

def get_embeddings(texts):
    """取得文字的 embedding 向量"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
    )
    return [e.values for e in result.embeddings]


# === 向量搜尋 ===

def search(query_emb, chunk_embeddings, chunks, top_k=3):
    """向量搜尋：找出最相關的 top_k 段落"""
    sims = cosine_similarity([query_emb], chunk_embeddings)[0]
    sorted_idx = sims.argsort()[::-1][:top_k]
    return [{"chunk": chunks[i], "score": round(float(sims[i]), 4)} for i in sorted_idx]


# === RAG：搜尋 + 生成 ===

def rag_answer(question, chunks, chunk_embeddings):
    """完整 RAG 流程：搜尋相關段落 → 塞進 prompt → 生成回答"""

    # 1. 對問題做 embedding
    query_emb = get_embeddings([question])[0]

    # 2. 向量搜尋
    results = search(query_emb, chunk_embeddings, chunks, top_k=3)

    print(f"  搜尋結果（top 3）：")
    for r in results:
        print(f"    [{r['score']:.4f}] {r['chunk']['title']}")

    # 3. 組合 context
    context = "\n\n".join([
        f"【{r['chunk']['title']}】\n{r['chunk']['content']}"
        for r in results
    ])

    # 4. 生成回答
    prompt = f"""根據以下法規內容回答問題。只用提供的法規回答，不要編造。

法規內容：
{context}

問題：{question}

請用繁體中文回答。"""

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

    return {
        "question": question,
        "answer": response.text,
        "sources": [r["chunk"]["title"] for r in results],
    }


# === 主程式 ===

if __name__ == "__main__":
    print("=" * 55)
    print("  RAG：Retrieval-Augmented Generation")
    print("=" * 55)

    # 1. 建立法規 embedding
    print(f"\n1. 建立法規 embedding（{len(REGULATION_CHUNKS)} 段）...")
    chunk_texts = [c["content"] for c in REGULATION_CHUNKS]
    chunk_embeddings = get_embeddings(chunk_texts)
    print(f"   完成（向量維度: {len(chunk_embeddings[0])}）")

    # 2. 測試問題
    questions = [
        "安全帽有什麼規定？",
        "高空作業需要什麼防護？",
        "噪音太大怎麼辦？",
    ]

    for q in questions:
        print(f"\n{'─' * 55}")
        print(f"  問題：{q}")
        print(f"{'─' * 55}\n")

        result = rag_answer(q, REGULATION_CHUNKS, chunk_embeddings)
        print(f"\n  回答：{result['answer']}")
        print(f"  來源：{result['sources']}")
