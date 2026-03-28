"""
Demo：Reranking（語意重排序）

向量搜尋的結果不一定精準。Reranking 用語意模型對結果做二次排序。

執行方式：
  python 10_reranking.py
  python 10_reranking.py --mock

需要：
  pip install google-generativeai python-dotenv numpy
"""

import json
import os
import sys
import numpy as np
from dotenv import load_dotenv

# 共用 RAG 的資料和函式
from importlib import import_module

load_dotenv()


# === 法規資料（跟 09_rag.py 一樣） ===

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


# === Embedding + 向量搜尋（跟 09_rag.py 一樣） ===

def get_embeddings(texts, mock=False):
    if mock:
        keywords = ["安全帽", "反光", "護目鏡", "出口", "高空", "化學", "噪音", "消防",
                    "helmet", "vest", "帽", "背心", "焊接", "作業", "違規"]
        vectors = []
        for text in texts:
            vec = [1.0 if kw in text else 0.0 for kw in keywords]
            np.random.seed(hash(text) % 2**31)
            vec = np.array(vec) + np.random.normal(0, 0.1, len(keywords))
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors

    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    result = genai.embed_content(model="models/text-embedding-004", content=texts)
    return result["embedding"]


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def vector_search(query_emb, chunk_embeddings, chunks, top_k=5):
    """向量搜尋（粗搜）"""
    scores = []
    for i, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_emb, emb)
        scores.append((score, i))
    scores.sort(reverse=True)
    return [{"chunk": chunks[idx], "vector_score": round(sc, 4)} for sc, idx in scores[:top_k]]


# === Reranking ===

def rerank_with_gemini(question, candidates, mock=False):
    """用 Gemini 對搜尋結果做語意重排序"""

    if mock:
        return _mock_rerank(question, candidates)

    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")

    # 組合候選段落
    candidate_text = ""
    for i, c in enumerate(candidates):
        candidate_text += f"\n段落 {i+1}（{c['chunk']['title']}）：\n{c['chunk']['content']}\n"

    prompt = f"""以下有 {len(candidates)} 段法規文字。
請根據問題的相關性，從最相關到最不相關排序。
只輸出排序後的編號，用逗號分隔。例如：3,1,5,2,4

問題：{question}
{candidate_text}
排序（最相關在前）："""

    response = model.generate_content(prompt)
    text = response.text.strip()

    # 解析排序
    try:
        order = [int(x.strip()) - 1 for x in text.split(",")]
        reranked = [candidates[i] for i in order if i < len(candidates)]
    except (ValueError, IndexError):
        reranked = candidates  # 解析失敗就不動

    # 加上 rerank score
    for i, item in enumerate(reranked):
        item["rerank_position"] = i + 1
        item["rerank_score"] = round(1.0 - i * 0.15, 2)

    return reranked


def _mock_rerank(question, candidates):
    """模擬 reranking"""
    # 簡單規則：標題包含問題關鍵字的排前面
    keywords = question.replace("？", "").replace("有什麼", "").replace("規定", "")

    scored = []
    for c in candidates:
        title = c["chunk"]["title"]
        content = c["chunk"]["content"]
        relevance = sum(1 for kw in keywords if kw in title or kw in content)
        scored.append((relevance, c))

    scored.sort(key=lambda x: -x[0])
    reranked = [item for _, item in scored]

    for i, item in enumerate(reranked):
        item["rerank_position"] = i + 1
        item["rerank_score"] = round(1.0 - i * 0.15, 2)

    return reranked


# === 比較：有/沒有 Reranking ===

if __name__ == "__main__":
    use_mock = "--mock" in sys.argv

    print("=" * 55)
    print("  Reranking：語意重排序")
    print("=" * 55)

    # 1. 建立 embedding
    print(f"\n建立法規 embedding...")
    chunk_texts = [c["content"] for c in REGULATION_CHUNKS]
    chunk_embeddings = get_embeddings(chunk_texts, mock=use_mock)

    # 2. 測試問題
    question = "在高處工作需要什麼安全裝備？"

    print(f"\n問題：{question}")

    # 3. 向量搜尋（粗搜 top 5）
    query_emb = get_embeddings([question], mock=use_mock)[0]
    search_results = vector_search(query_emb, chunk_embeddings, REGULATION_CHUNKS, top_k=5)

    print(f"\n{'─' * 55}")
    print(f"  Step 1：向量搜尋（粗搜 top 5）")
    print(f"{'─' * 55}")
    for i, r in enumerate(search_results):
        print(f"  {i+1}. [{r['vector_score']:.4f}] {r['chunk']['title']}")

    # 4. Reranking（精排）
    reranked = rerank_with_gemini(question, search_results, mock=use_mock)

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
        print(f"\n  ✅ Reranking 改變了排序！更相關的結果排到前面。")
    else:
        print(f"\n  排序沒變，但 Reranking 確認了向量搜尋的結果是合理的。")

    # 重點
    print(f"\n{'=' * 55}")
    print("  Reranking 流程")
    print(f"{'=' * 55}")
    print("""
  1. 向量搜尋：快速粗搜 top N（便宜、快）
  2. Reranker：對 top N 做語意精排（慢、但精準）
  3. 取精排後的 top K 送給 LLM

  向量搜尋 → 「大概相關」
  Reranking → 「真的相關」

  常見 Reranker：
  - Gemini（用 prompt 排序，簡單）
  - Cohere Rerank API（專用模型，效果好）
  - Cross-encoder（HuggingFace，可離線）
""")
