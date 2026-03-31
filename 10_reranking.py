"""
Demo：Reranking（語意重排序）

向量搜尋的結果不一定精準。Reranking 用語意模型對結果做二次排序。

執行方式：
  python 10_reranking.py

需要：
  pip install google-genai scikit-learn python-dotenv numpy
  .env 裡設定 GOOGLE_API_KEY
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# === 法規資料 ===

REGULATION_CHUNKS = [
    {"id": "reg_001", "title": "安全帽規定",
     "content": "依據職業安全衛生設施規則第 281 條，雇主對於在高度 2 公尺以上之工作場所，應使勞工確實使用安全帽。"},
    {"id": "reg_002", "title": "反光背心規定",
     "content": "依據職業安全衛生設施規則第 21 條，雇主應提供適當之反光標示或背心。"},
    {"id": "reg_003", "title": "護目鏡規定",
     "content": "依據職業安全衛生設施規則第 287 條，從事焊接、切割或研磨作業時，應配戴護目鏡或面罩。"},
    {"id": "reg_004", "title": "安全出口規定",
     "content": "依據建築技術規則第 97 條，安全出口寬度不得小於 1.2 公尺，不得堆放物品阻礙通行。"},
    {"id": "reg_005", "title": "高空作業安全帶",
     "content": "依據職業安全衛生設施規則第 281 條之 1，從事高度 2 公尺以上之高空作業，應使用安全帶或安全母索。"},
    {"id": "reg_006", "title": "化學品標示",
     "content": "依據危害性化學品標示及通識規則第 5 條，雇主應於化學品容器上標示名稱、危害圖式、警示語。"},
    {"id": "reg_007", "title": "噪音防護",
     "content": "依據職業安全衛生設施規則第 300 條，工作場所噪音超過 85 分貝時，雇主應提供耳塞或耳罩。"},
    {"id": "reg_008", "title": "消防設備",
     "content": "依據消防法第 6 條，各類場所應設置滅火器、室內消防栓等消防安全設備。"},
]


# === Embedding ===

def get_embeddings(texts):
    result = client.models.embed_content(model="gemini-embedding-001", contents=texts)
    return [e.values for e in result.embeddings]


# === 向量搜尋（粗搜） ===

def vector_search(query_emb, chunk_embeddings, chunks, top_k=5):
    sims = sklearn_cosine([query_emb], chunk_embeddings)[0]
    sorted_idx = sims.argsort()[::-1][:top_k]
    return [{"chunk": chunks[i], "vector_score": round(float(sims[i]), 4)} for i in sorted_idx]


# === Reranking ===

def rerank_with_gemini(question, candidates):
    """用 Gemini 對搜尋結果做語意重排序"""

    candidate_text = ""
    for i, c in enumerate(candidates):
        candidate_text += f"\n段落 {i+1}（{c['chunk']['title']}）：\n{c['chunk']['content']}\n"

    prompt = f"""以下有 {len(candidates)} 段法規文字。
請根據問題的相關性，從最相關到最不相關排序。
只輸出排序後的編號，用逗號分隔。例如：3,1,5,2,4

問題：{question}
{candidate_text}
排序（最相關在前）："""

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = response.text.strip()

    try:
        order = [int(x.strip()) - 1 for x in text.split(",")]
        reranked = [candidates[i] for i in order if i < len(candidates)]
    except (ValueError, IndexError):
        reranked = candidates

    for i, item in enumerate(reranked):
        item["rerank_position"] = i + 1
        item["rerank_score"] = round(1.0 - i * 0.15, 2)

    return reranked


# === 主程式 ===

if __name__ == "__main__":
    print("=" * 55)
    print("  Reranking：語意重排序")
    print("=" * 55)

    # 1. 建立 embedding
    print(f"\n建立法規 embedding...")
    chunk_texts = [c["content"] for c in REGULATION_CHUNKS]
    chunk_embeddings = get_embeddings(chunk_texts)

    # 2. 測試問題
    question = "工人在屋頂施工掉下來，違反什麼法規？"
    print(f"\n問題：{question}")

    # 3. 向量搜尋（粗搜 top 5）
    query_emb = get_embeddings([question])[0]
    search_results = vector_search(query_emb, chunk_embeddings, REGULATION_CHUNKS, top_k=5)

    print(f"\n{'─' * 55}")
    print(f"  Step 1：向量搜尋（粗搜 top 5）")
    print(f"{'─' * 55}")
    for i, r in enumerate(search_results):
        print(f"  {i+1}. [{r['vector_score']:.4f}] {r['chunk']['title']}")

    # 4. Reranking（精排）
    reranked = rerank_with_gemini(question, search_results)

    print(f"\n{'─' * 55}")
    print(f"  Step 2：Reranking（語意精排）")
    print(f"{'─' * 55}")
    for r in reranked:
        print(f"  {r['rerank_position']}. [{r['rerank_score']:.2f}] {r['chunk']['title']}")

    # 5. 比較
    print(f"\n{'─' * 55}")
    print(f"  比較結果")
    print(f"{'─' * 55}")

    old_top3 = [r["chunk"]["title"] for r in search_results[:3]]
    new_top3 = [r["chunk"]["title"] for r in reranked[:3]]

    print(f"  向量搜尋 top 3: {old_top3}")
    print(f"  Reranking top 3: {new_top3}")

    if old_top3 != new_top3:
        print(f"\n  Reranking 改變了排序！更相關的結果排到前面。")
    else:
        print(f"\n  排序沒變，Reranking 確認了向量搜尋的結果是合理的。")
