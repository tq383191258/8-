# RAG Demo

这是一个最小可运行的 RAG（Retrieval-Augmented Generation）演示，用于帮助学习者理解：

- 如何用 embedding 对文档做向量化；
- 如何用 FAISS 做相似度搜索；
- 如何把检索结果拼接进 prompt；
- 如何用一个小模型生成答案；
- 如何用 Gradio 做一个简单交互界面。

## 运行步骤

1. 安装依赖：

```bash
pip install -r rag-demo/requirements.txt
```

2. 构建索引：

```bash
python rag-demo/build_index.py
```

3. 启动 Web UI：

```bash
python rag-demo/app.py
```

4. 打开浏览器访问：

```text
http://localhost:7860
```

## 说明

这个 demo 适合用于学习大模型的典型工作流，不是生产级实现。
它的重点是让你足够快地跑通一个完整流程，理解 RAG 的“检索 + 生成”逻辑。

## 扩展建议

- 替换成更强的文本模型（如 Qwen / Llama / Mistral）
- 使用更大规模文档库
- 使用 LangChain 或 LlamaIndex 做更高级的框架封装
- 增加评估指标或日志记录
