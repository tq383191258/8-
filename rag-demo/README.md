# RAG Demo 模板

这是一个最小可运行的 RAG（Retrieval-Augmented Generation）示例，用于帮助你快速理解大模型在知识库问答场景中的工作流程。

## 目录结构

```text
rag-demo/
  README.md
  requirements.txt
  build_index.py
  app.py
  docs/
    doc1.txt
    doc2.txt
  data/
    index/
```

## 依赖

```bash
pip install -r rag-demo/requirements.txt
```

## 运行步骤

1. 把文档放在 `rag-demo/docs/` 中。
2. 运行：

```bash
python rag-demo/build_index.py
```

3. 启动 demo：

```bash
python rag-demo/app.py
```

4. 打开浏览器，访问 `http://localhost:7860`

## 示例代码

### requirements.txt

```text
transformers>=4.30.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.3
gradio>=3.30
torch>=2.0.0
numpy
scikit-learn
tqdm
```

### build_index.py

```python
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from pathlib import Path
import pickle

DOCS_DIR = Path("rag-demo/docs")
OUT_DIR = Path("rag-demo/data/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_docs():
    docs = []
    for f in DOCS_DIR.glob("*.txt"):
        text = f.read_text(encoding="utf-8")
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
        docs.extend(chunks)
    return docs


def build():
    model = SentenceTransformer(EMB_MODEL)
    docs = load_docs()
    embeddings = model.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(OUT_DIR / "docs.index"))
    np.save(OUT_DIR / "docs_embeddings.npy", embeddings)
    with open(OUT_DIR / "docs.pkl", "wb") as f:
        pickle.dump({"docs": docs}, f)
    print("Index built successfully.")


if __name__ == "__main__":
    build()
```

### app.py

```python
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import gradio as gr
from pathlib import Path

OUT_DIR = Path("rag-demo/data/index")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "gpt2"
TOP_K = 3

index = faiss.read_index(str(OUT_DIR / "docs.index"))
with open(OUT_DIR / "docs.pkl", "rb") as f:
    data = pickle.load(f)
docs = data["docs"]

embed_model = SentenceTransformer(EMB_MODEL)
gen = pipeline("text-generation", model=GEN_MODEL, device=-1, max_new_tokens=150)


def answer(question):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, TOP_K)
    context = "\n\n".join([docs[i] for i in I[0]])
    prompt = f"根据以下资料回答问题：\n\n{context}\n\n问题：{question}\n回答："
    out = gen(prompt, do_sample=True, temperature=0.2, top_p=0.9, num_return_sequences=1)[0]["generated_text"]
    answer_text = out.replace(prompt, "").strip()
    return answer_text, context

with gr.Blocks() as demo:
    gr.Markdown("# Minimal RAG Demo")
    question = gr.Textbox(label="输入问题")
    answer_text = gr.Textbox(label="回答")
    context_box = gr.Textbox(label="检索到的上下文")
    btn = gr.Button("提交")
    btn.click(fn=answer, inputs=question, outputs=[answer_text, context_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

## 注意事项

- 这个 demo 适合用于学习流程，不是生产级解决方案。
- 真实项目中建议用更强的模型、加入安全过滤、持久化索引和更稳定的调优方式。
- 可替换为 `Qwen`, `Llama`, `Mistral` 等更强大模型进行更好的效果。
