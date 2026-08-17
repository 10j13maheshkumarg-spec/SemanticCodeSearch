# Semantic Code Search Engine & Web IDE

A powerful, AI-driven Semantic Code Search Engine built directly into a Web IDE. This project moves beyond traditional keyword-matching by using Machine Learning vector embeddings to interpret the *meaning* and *intent* behind natural language queries.

## 🚀 The Core Innovation

Most traditional search tools (like `grep` or standard GitHub search) fail if you don't know the exact variable name or syntax. This engine understands plain English context. By converting source code and search queries into mathematical vectors, it finds code snippets based on meaning.

To make this deployable on ultra-cheap, free-tier cloud infrastructure, the entire ML embedding pipeline was engineered to use **ONNX Runtime**, shrinking the memory footprint to under 150MB and eliminating the need for heavy PyTorch dependencies.

## ✨ Features

- **Instant GitHub Ingestion:** Paste a GitHub repository URL to automatically clone, chunk, generate embeddings, and index the entire codebase in seconds.
- **Tri-Mode Search Engine:** Toggle seamlessly between:
  - **Semantic Mode:** Uses AI to find code based on meaning.
  - **Keyword Mode:** Exact word matching using the BM25 algorithm.
  - **Hybrid Mode:** Combines both for maximum accuracy.
- **Integrated Web IDE Layout:** 
  - Dynamic **File Tree Explorer**
  - **Monaco Editor** for syntax-highlighted code viewing (VS Code engine).
  - Real-time **WebSocket Terminal** for executing commands in the browser.
- **Context-Aware AI Chat:** A built-in LLM side-panel (powered by Groq) that allows you to instantly chat with, debug, or refactor the code you just searched for.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, WebSockets
- **Machine Learning & Embeddings:** ONNX Runtime, `all-MiniLM-L6-v2` (running entirely locally)
- **Vector Database:** ChromaDB
- **Keyword Engine:** `rank_bm25` (Okapi BM25)
- **LLM Integration:** Groq API
- **Frontend:** Vanilla JS/HTML/CSS, Monaco Editor, xterm.js

## 🏁 Getting Started

### Prerequisites
- Python 3.9+
- A [Groq API Key](https://console.groq.com/) for the AI Chat Assistant.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/10j13maheshkumarg-spec/SemanticCodeSearch.git
   cd SemanticCodeSearch
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

4. **Run the Application:**
   ```bash
   python app.py
   ```
   *The application will start on `http://localhost:8001`.*

## 💡 Usage
1. Open the Web IDE in your browser.
2. Paste a target GitHub URL into the "Folder path to index..." field and click **Index**.
3. Once indexed, type a natural language query (e.g., *"where is the user authentication logic?"*) into the search bar.
4. Click on a search result to open it in the Monaco editor.
5. Use the AI Assistant panel on the right to ask questions about the code you are viewing!
