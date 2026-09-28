# 作用：读取 docs/*.txt，切分成 chunk，构建 embedding，并保存 FAISS 索引
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DOCS_DIR = Path(__file__).resolve().parent / "docs"
OUT_DIR = Path(__file__).resolve().parent / "data" / "index"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def split_text_into_chunks(text: str, chunk_size: int = 400, overlap: int = 80):
    """按段落和长度切分文本。
    若段落为空则忽略。
    """
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        words = para.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
    return chunks


def load_documents():
    documents = []
    for file in sorted(DOCS_DIR.glob("*.txt")):
        text = file.read_text(encoding="utf-8")
        chunks = split_text_into_chunks(text)
        for idx, chunk in enumerate(chunks):
            documents.append({
                "id": f"{file.stem}_{idx}",
                "source": file.name,
                "text": chunk,
            })
    return documents


def build_index():
    docs = load_documents()
    if not docs:
        raise FileNotFoundError(f"未在 {DOCS_DIR} 中找到任何 .txt 文档。请先放入文档。")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    # 保存索引与元数据
    faiss.write_index(index, str(OUT_DIR / "docs.index"))
    np.save(OUT_DIR / "embeddings.npy", embeddings.astype("float32"))
    with open(OUT_DIR / "docs_metadata.pkl", "wb") as f:
        pickle.dump(docs, f)

    print(f"成功构建索引：{len(docs)} 个 chunk，维度：{dim}")
    print(f"索引保存到：{OUT_DIR}")


if __name__ == "__main__":
    build_index()
