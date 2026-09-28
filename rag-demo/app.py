# 作用：从 FAISS 索引中检索相关文本，并用小型生成模型给出回答
import pickle
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "data" / "index" / "docs.index"
METADATA_PATH = BASE_DIR / "data" / "index" / "docs_metadata.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "gpt2"
TOP_K = 3


if not INDEX_PATH.exists() or not METADATA_PATH.exists():
    raise FileNotFoundError(
        "未找到索引文件。请先运行：python rag-demo/build_index.py"
    )

index = faiss.read_index(str(INDEX_PATH))
with open(METADATA_PATH, "rb") as f:
    docs = pickle.load(f)

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
# device=-1 表示使用 CPU；若有 GPU，可设置为 0
generator = pipeline("text-generation", model=GEN_MODEL_NAME, device=-1, max_new_tokens=160)


def retrieve(query: str, top_k: int = TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    q_emb = q_emb.astype("float32")
    distances, indices = index.search(q_emb, top_k)
    results = []
    for idx in indices[0]:
        if idx < 0:
            continue
        results.append(docs[int(idx)])
    return results


def answer_question(question: str):
    hits = retrieve(question)
    if not hits:
        return "未找到相关资料。请先建索引并检查文档内容。", ""

    context = "\n\n".join([item["text"] for item in hits])
    prompt = (
        "请基于下面的资料回答问题。如果资料中没有答案，请明确说明无法从资料中得出结论。\n\n"
        f"资料：\n{context}\n\n问题：{question}\n\n回答："
    )

    response = generator(prompt, do_sample=True, temperature=0.2, top_p=0.9, num_return_sequences=1)[0]["generated_text"]
    answer_text = response.replace(prompt, "").strip()
    return answer_text, context


with gr.Blocks(title="RAG Demo") as demo:
    gr.MarkDown("# 最小 RAG Demo\n\n基于 FAISS + sentence-transformers + gpt2 的问答示例。")
    with gr.Row():
        question = gr.Textbox(label="请输入问题", placeholder="例如：什么是 Transformer？")
        submit = gr.Button("生成回答")

    output_text = gr.Textbox(label="回答结果", lines=8)
    context_box = gr.Textbox(label="检索到的上下文", lines=8)

    submit.click(fn=answer_question, inputs=question, outputs=[output_text, context_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
