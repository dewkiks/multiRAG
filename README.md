# 📚 MultiRAG PDF Summarizer and Retriever

This project processes academic or research PDFs using LangChain, Google Gemini, and ChromaDB. It performs intelligent chunking, summarizes both **text and images**, and stores them for **retrieval-based querying**. Useful for building AI-powered study tools, semantic search systems, or revision companions.

---

## 🚀 Features

- 🧠 Extracts and summarizes text and images from PDFs  
- 🖼️ Supports base64-encoded image blocks (e.g., diagrams/graphs)  
- 📊 Table detection scaffolded (extensible)  
- 🔍 Stores both raw content and summaries with `MultiVectorRetriever`  
- 🧾 Queries the document like: _"What are the types of transformers?"_

---

## 🛠️ Tech Stack

- 🦜 LangChain (Chains, Prompts, Chroma Retriever)  
- 🧠 Google Gemini (Text and Image LLMs)  
- 📄 `unstructured` (for advanced PDF parsing)  
- 💾 ChromaDB + InMemoryStore (vector + doc storage)  
- 🔐 dotenv for API key handling

---

## 📦 Installation

1. Create a conda env with GPU-enabled PyTorch (CUDA 12.1):

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Add your `.env` file:

    ```
    GEMINI_API_KEY=your_google_gemini_api_key_here
    ```

---

## 📂 Folder Structure

```
Data/
├── transformer.pdf       # Your input research PDF
main.py                   # Runs the Multirag class
README.md
.env                      # Gemini API key
```

---

## 🧪 Usage

```python
from main import Multirag

tt = Multirag("Data/transformer.pdf")
```

This will:
- Chunk & extract text and images  
- Summarize both with Gemini  
- Store original + summaries in ChromaDB  
- Perform sample retrieval with a test query

---

## 📝 Notes

- Table summarization logic is scaffolded but not implemented yet.  
- Image summarization sends base64-encoded images via Gemini's multimodal API.  
- You can extend this into a PDF-to-chatbot, podcast script generator, or AI study tool.

---

## 🧑‍💻 Author

Made with ❤️ by dewkiks

---

