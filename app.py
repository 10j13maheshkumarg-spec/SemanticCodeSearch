import os
import ast
import json
import hashlib
import pickle
import re
import threading
import time
import asyncio
import subprocess
import requests
from pathlib import Path
from fastapi import FastAPI, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import chromadb
from rank_bm25 import BM25Okapi
import zipfile
import io
import shutil

from dotenv import load_dotenv
load_dotenv()
try:
    from groq import Groq
    groq_client = Groq()
except Exception as e:
    groq_client = None
    print(f"Groq Client failed to initialize: {e}")

app = FastAPI(title="Semantic Code Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow mobile apps to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

def get_embeddings(texts):
    if not HF_TOKEN:
        print("HF_TOKEN missing. Using dummy embeddings.")
        return [[0.0]*384 for _ in texts]
        
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Send request with retry logic for model loading
    for attempt in range(3):
        try:
            response = requests.post(HF_API_URL, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                time.sleep(2) # wait for model to load on HF side
            else:
                print(f"HF API Error {response.status_code}: {response.text}")
                break # 401/404 errors won't fix themselves with retries
        except Exception as e:
            print(f"HF API request failed: {e}")
            break
            
    print("WARNING: HuggingFace API failed. Falling back to zero-embeddings so Keyword Search still works.")
    return [[0.0]*384 for _ in texts]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chroma_path = os.path.join(BASE_DIR, "chroma_db")
client = chromadb.PersistentClient(path=chroma_path)

templates_dir = os.path.join(BASE_DIR, "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

PROJECTS_FILE = os.path.join(BASE_DIR, "projects.json")
active_collection_name = None
active_folder_path = None
observer = None

def get_projects():
    if os.path.exists(PROJECTS_FILE):
        with open(PROJECTS_FILE, 'r') as f:
            projects = json.load(f)
            # Filter out old local paths, keeping only valid HTTP/GitHub URLs
            valid_projects = [p for p in projects if p.startswith("http")]
            return valid_projects
    return []

def save_project(folder_path):
    projects = get_projects()
    if folder_path not in projects:
        projects.append(folder_path)
        with open(PROJECTS_FILE, 'w') as f:
            json.dump(projects, f)

def get_collection_name(folder_path):
    return "proj_" + hashlib.md5(folder_path.encode()).hexdigest()

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def download_github_repo(github_url):
    # Parse github URL (e.g. https://github.com/user/repo)
    clean_url = github_url.strip().rstrip('/')
    if not clean_url.startswith("https://github.com/"):
        raise ValueError("Invalid GitHub URL. Must start with https://github.com/")
        
    parts = clean_url.split('/')
    if len(parts) < 5:
        raise ValueError("Invalid GitHub URL format.")
        
    owner = parts[-2]
    repo = parts[-1]
    
    # Try downloading main branch first, then master
    branches = ['main', 'master']
    zip_url = None
    resp = None
    
    for branch in branches:
        url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            zip_url = url
            resp = r
            break
            
    if not resp or resp.status_code != 200:
        raise ValueError(f"Could not download repository. Ensure it is public and has a main/master branch.")
        
    # Save and extract to repos/owner/repo
    repo_dir = os.path.join(BASE_DIR, "repos", owner, f"{repo}-{branch}")
    
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
        
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(os.path.join(BASE_DIR, "repos", owner))
        
    # The extracted folder is usually named repo-branch
    return repo_dir

def extract_python_chunks(source, filepath):
    chunks = []
    try:
        tree = ast.parse(source)
    except Exception:
        return []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', start_line)
            source_lines = source.splitlines()[start_line-1:end_line]
            
            type_name = "Class" if isinstance(node, ast.ClassDef) else "Function"
            header = f"{type_name}: {node.name}"
            
            lines_per_chunk = 15
            overlap = 5
            start_idx = 0
            while start_idx < len(source_lines):
                end_idx = min(start_idx + lines_per_chunk, len(source_lines))
                chunk_code = "\n".join(source_lines[start_idx:end_idx])
                enriched_text = f"File: {filepath}\nContext: {header}\nCode:\n{chunk_code}"
                
                chunks.append({
                    "code": chunk_code,
                    "enriched_text": enriched_text,
                    "header": header,
                    "filepath": filepath,
                    "start_line": start_line + start_idx,
                    "end_line": start_line + end_idx - 1
                })
                start_idx += (lines_per_chunk - overlap)
    return chunks

def extract_generic_chunks(source, filepath, lines_per_chunk=15, overlap=5):
    lines = source.splitlines()
    chunks = []
    start = 0
    ext = os.path.splitext(filepath)[1]
    while start < len(lines):
        end = min(start + lines_per_chunk, len(lines))
        chunk_code = "\n".join(lines[start:end])
        if chunk_code.strip():
            header = f"Text Block ({ext})"
            enriched_text = f"File: {filepath}\nContext: {header}\nText:\n{chunk_code}"
            chunks.append({
                "code": chunk_code,
                "enriched_text": enriched_text,
                "header": header,
                "filepath": filepath,
                "start_line": start + 1,
                "end_line": end
            })
        start += (lines_per_chunk - overlap)
    return chunks

def extract_code_chunks(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            source = file.read()
    except Exception:
        return []
        
    if filepath.endswith('.py'):
        py_chunks = extract_python_chunks(source, filepath)
        if py_chunks: return py_chunks
        return extract_generic_chunks(source, filepath)
    elif filepath.endswith(('.js', '.html', '.css', '.cpp', '.c', '.java', '.go', '.rs', '.ts', '.jsx', '.tsx', '.md', '.txt')):
        return extract_generic_chunks(source, filepath)
    return []

# --- Extension to Monaco language ID mapping ---
EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".html": "html",
    ".css": "css",
    ".cpp": "cpp",
    ".c": "c",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".txt": "plaintext",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
}

# --- Directories to skip when walking the file tree ---
SKIP_DIRS = {'.git', 'venv', 'env', 'node_modules', 'chroma_db', '__pycache__'}

class SearchQuery(BaseModel):
    query: str
    n_results: int = 5
    search_mode: str = "semantic" # can be semantic, keyword, or hybrid

class IngestRequest(BaseModel):
    github_url: str
    
class SetProjectRequest(BaseModel):
    github_url: str

class ChatRequest(BaseModel):
    query: str
    results: list
    history: list = []

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/projects")
async def api_get_projects():
    global active_collection_name, active_folder_path
    projects = get_projects()
    if active_collection_name is None and projects:
        active_folder_path = projects[-1]
        active_collection_name = get_collection_name(active_folder_path)
    return {"projects": projects, "active": active_folder_path}

@app.post("/api/projects/set")
async def api_set_project(request: SetProjectRequest):
    global active_collection_name, active_folder_path
    folder_path = request.github_url
    if folder_path not in get_projects():
        return {"error": "Project not found in history."}
        
    coll_name = get_collection_name(folder_path)
    try:
        client.get_collection(name=coll_name)
        active_collection_name = coll_name
        active_folder_path = folder_path
        return {"message": "Project switched successfully.", "active": folder_path}
    except Exception:
        return {"error": "Project database not found. Please re-index."}

@app.post("/api/ingest")
async def ingest_folder(request: IngestRequest):
    global active_collection_name, active_folder_path
    github_url = request.github_url.strip().strip('"').strip("'")
    
    try:
        folder_path = download_github_repo(github_url)
    except Exception as e:
        return {"error": str(e)}
        
    coll_name = get_collection_name(github_url)
    try:
        client.delete_collection(name=coll_name)
    except Exception:
        pass
        
    collection = client.create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})
    
    raw_documents, enriched_documents, metadatas, ids = [], [], [], []
    
    for root, _, files in os.walk(folder_path):
        if any(part.startswith('.') or part in ['venv', 'env', 'node_modules', 'chroma_db', '__pycache__'] for part in root.split(os.sep)):
            continue
            
        for file in files:
            filepath = os.path.join(root, file)
            chunks = extract_code_chunks(filepath)
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{filepath}_{chunk['start_line']}_{i}"
                raw_documents.append(chunk['code'])
                enriched_documents.append(chunk['enriched_text'])
                metadatas.append({
                    "filepath": chunk['filepath'],
                    "header": chunk['header'],
                    "start_line": chunk['start_line'],
                    "end_line": chunk['end_line']
                })
                ids.append(doc_id)
                    
    if not raw_documents:
        return {"error": "No valid source code found in this folder."}
        
    try:
        # Build BM25 Keyword Index using enriched texts to include paths
        bm25_data = {
            "bm25": BM25Okapi([tokenize(d) for d in enriched_documents]),
            "documents": raw_documents,
            "metadatas": metadatas,
            "ids": ids
        }
        with open(os.path.join(BASE_DIR, f"bm25_{coll_name}.pkl"), 'wb') as f:
            pickle.dump(bm25_data, f)

        # Build Chroma Semantic Index using enriched texts
        batch_size = 50 # HF API limits
        for i in range(0, len(raw_documents), batch_size):
            batch_docs = enriched_documents[i:i+batch_size]
            batch_embeddings = get_embeddings(batch_docs)
            
            collection.upsert(
                embeddings=batch_embeddings,
                documents=raw_documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
            
        save_project(github_url)
        active_collection_name = coll_name
        active_folder_path = github_url
        return {"message": f"Successfully indexed {len(raw_documents)} code snippets from {github_url}!"}
    except Exception as e:
        return {"error": f"Error during indexing: {str(e)}"}

@app.post("/api/search")
async def search_code(search_query: SearchQuery):
    if not active_collection_name:
        return {"error": "No project is currently active or indexed."}
            
    try:
        collection = client.get_collection(name=active_collection_name)
        if collection.count() == 0:
            return {"error": "The active project has no indexed code."}
            
        # 1. Semantic Search
        semantic_map = {}
        if search_query.search_mode in ["semantic", "hybrid"]:
            try:
                query_embedding = get_embeddings([search_query.query])
                results = collection.query(query_embeddings=query_embedding, n_results=search_query.n_results * 2)
            except Exception as e:
                return {"error": f"HF API Error: {str(e)}"}
            
            if results['documents'] and len(results['documents']) > 0:
                for doc, meta, dist, doc_id in zip(results['documents'][0], results['metadatas'][0], results['distances'][0], results['ids'][0]):
                    similarity = max(0.0, 1.0 - dist)
                    semantic_map[doc_id] = {
                        "code": doc, "filepath": meta['filepath'], "header": meta['header'], 
                        "start_line": meta['start_line'], "end_line": meta['end_line'],
                        "semantic_score": similarity, "bm25_score": 0.0
                    }

        # 2. BM25 Exact Keyword Search
        # Auto-fallback to keyword if semantic failed (e.g. HF API returned zero-vectors)
        force_keyword = search_query.search_mode == "semantic" and not semantic_map
        
        if search_query.search_mode in ["keyword", "hybrid"] or force_keyword:
            bm25_path = os.path.join(BASE_DIR, f"bm25_{active_collection_name}.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                query_tokens = tokenize(search_query.query)
                # Filter very common stop words to improve BM25 for natural language
                stop_words = {"what", "where", "how", "is", "the", "a", "an", "in", "on", "at", "to", "for", "of", "do", "does", "did"}
                query_tokens = [t for t in query_tokens if t not in stop_words]
                
                if query_tokens:
                    bm25 = bm25_data["bm25"]
                    doc_scores = bm25.get_scores(query_tokens)
                    top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:search_query.n_results * 2]
                    for i in top_indices:
                        score = doc_scores[i]
                        if score > 0:
                            doc_id = bm25_data["ids"][i]
                            if doc_id in semantic_map:
                                semantic_map[doc_id]["bm25_score"] = score
                            else:
                                meta = bm25_data["metadatas"][i]
                                semantic_map[doc_id] = {
                                    "code": bm25_data["documents"][i],
                                    "filepath": meta['filepath'], "header": meta['header'],
                                    "start_line": meta['start_line'], "end_line": meta['end_line'],
                                    "semantic_score": 0.0, "bm25_score": score
                                }

        # 3. Mode Filtering & Merge
        final_results = []
        for doc_id, data in semantic_map.items():
            if search_query.search_mode == "semantic" and not force_keyword:
                if data["semantic_score"] >= 0.25:
                    data["similarity_score"] = data["semantic_score"] * 100
                    final_results.append(data)
            elif search_query.search_mode == "keyword" or force_keyword:
                if data["bm25_score"] >= 1.0:
                    data["similarity_score"] = data["bm25_score"] * 10
                    final_results.append(data)
            else: # hybrid
                if data["semantic_score"] >= 0.30 or data["bm25_score"] >= 2.0:
                    data["similarity_score"] = round((data["semantic_score"] * 100) + data["bm25_score"], 4)
                    final_results.append(data)

        final_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return {"results": final_results[:search_query.n_results]}
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# NEW FEATURE 1: File Tree Endpoint
# ============================================================

def build_file_tree(root_path):
    """Recursively build a JSON-serializable file tree structure."""
    tree = []
    try:
        entries = os.listdir(root_path)
    except PermissionError:
        return tree

    dirs = []
    files = []

    for entry in sorted(entries, key=lambda e: e.lower()):
        full_path = os.path.join(root_path, entry)
        if os.path.isdir(full_path):
            # Skip hidden dirs and common exclusions
            if entry.startswith('.') or entry in SKIP_DIRS:
                continue
            children = build_file_tree(full_path)
            dirs.append({
                "name": entry,
                "path": full_path,
                "type": "dir",
                "children": children
            })
        else:
            files.append({
                "name": entry,
                "path": full_path,
                "type": "file"
            })

    # Directories first, then files (both already alphabetically sorted)
    return dirs + files


def get_repo_dir(github_url):
    clean_url = github_url.strip().rstrip('/')
    parts = clean_url.split('/')
    if len(parts) < 5: return None
    owner = parts[-2]
    repo = parts[-1]
    base = os.path.join(BASE_DIR, "repos", owner)
    for branch in ['main', 'master']:
        path = os.path.join(base, f"{repo}-{branch}")
        if os.path.exists(path):
            return path
    return None

@app.get("/api/file_tree")
async def api_file_tree():
    if not active_folder_path:
        return JSONResponse(content={"error": "No project active"}, status_code=400)
        
    repo_dir = get_repo_dir(active_folder_path)
    if not repo_dir:
        return JSONResponse(content={"error": "Repository not downloaded"}, status_code=400)
    
    tree = build_file_tree(repo_dir)
    return {"tree": tree}


# ============================================================
# NEW FEATURE 2: File Content Endpoint
# ============================================================

@app.get("/api/file_content")
async def api_file_content(path: str):
    if not active_folder_path:
        return JSONResponse(content={"error": "No project active"}, status_code=400)
        
    repo_dir = get_repo_dir(active_folder_path)
    if not repo_dir:
        return JSONResponse(content={"error": "Repository not downloaded"}, status_code=400)

    # Security: resolve the path and verify it's within the active project
    try:
        resolved = os.path.realpath(path)
        project_root = os.path.realpath(repo_dir)
        if not resolved.startswith(project_root):
            return JSONResponse(
                content={"error": "Access denied: path is outside the active project."},
                status_code=403
            )
    except Exception:
        return JSONResponse(content={"error": "Invalid path."}, status_code=400)

    if not os.path.isfile(resolved):
        return JSONResponse(content={"error": "File not found."}, status_code=404)

    # Determine language from extension
    ext = os.path.splitext(resolved)[1].lower()
    language = EXTENSION_LANGUAGE_MAP.get(ext, "plaintext")
    filename = os.path.basename(resolved)

    try:
        with open(resolved, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return JSONResponse(
            content={"error": "Cannot read file: not a text file."},
            status_code=400
        )
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error reading file: {str(e)}"},
            status_code=500
        )

    return {"content": content, "language": language, "filename": filename}


# ============================================================
# NEW FEATURE 3: Upgraded Chat with Conversation Memory
# ============================================================

@app.post("/api/chat")
async def chat_with_code(request: ChatRequest):
    if not groq_client:
        return {"error": "Groq API key not configured or failed to initialize. Please check your .env file."}

    # Build code context from search results
    context = ""
    for r in request.results:
        context += f"\nFile: {r['filepath']} ({r['header']})\n```\n{r['code']}\n```\n"

    # Build the messages array with conversation history
    messages = []

    # System message
    messages.append({
        "role": "system",
        "content": "You are a Senior Software Engineer analyzing code. Answer based strictly on the provided code snippets. Do not hallucinate."
    })

    # Code context as a user message
    messages.append({
        "role": "user",
        "content": f"Here are the relevant code snippets for reference:\n{context}"
    })

    # Append conversation history
    for msg in request.history:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })

    # Current user question
    messages.append({
        "role": "user",
        "content": request.query
    })

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        return {"answer": completion.choices[0].message.content}
    except Exception as e:
        return {"error": f"LLM Error: {str(e)}"}


# ============================================================
# NEW FEATURE 4: WebSocket Terminal
# ============================================================

@app.websocket("/ws/terminal")
async def websocket_terminal(websocket: WebSocket):
    await websocket.accept()

    # Determine the working directory
    repo_dir = get_repo_dir(active_folder_path) if active_folder_path else None
    cwd = repo_dir if repo_dir else BASE_DIR

    try:
        # Spawn a shell subprocess (bash on linux, cmd on windows)
        shell_cmd = "bash" if os.name == "posix" else "cmd.exe"
        process = subprocess.Popen(
            shell_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            shell=True,
            bufsize=0
        )

        loop = asyncio.get_event_loop()

        async def read_stream(stream, label="stdout"):
            """Read from a subprocess stream line by line using run_in_executor."""
            try:
                while True:
                    line = await loop.run_in_executor(None, stream.readline)
                    if not line:
                        break
                    try:
                        decoded = line.decode('utf-8', errors='replace')
                    except Exception:
                        decoded = line.decode('latin-1', errors='replace')
                    try:
                        await websocket.send_text(decoded)
                    except Exception:
                        break
            except Exception:
                pass

        # Launch background tasks to read stdout and stderr
        stdout_task = asyncio.ensure_future(read_stream(process.stdout, "stdout"))
        stderr_task = asyncio.ensure_future(read_stream(process.stderr, "stderr"))

        # Read input from WebSocket and write to stdin
        try:
            while True:
                data = await websocket.receive_text()
                if process.stdin:
                    try:
                        process.stdin.write((data + "\n").encode('utf-8'))
                        process.stdin.flush()
                    except Exception:
                        break
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            # Cleanup
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.kill()
            except Exception:
                pass
            stdout_task.cancel()
            stderr_task.cancel()

    except Exception as e:
        try:
            await websocket.send_text(f"Terminal error: {str(e)}\n")
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
