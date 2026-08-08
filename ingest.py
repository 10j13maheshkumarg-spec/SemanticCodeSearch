import os
import ast
import argparse
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize Sentence Transformer model
MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "code_embeddings"
# Use cosine similarity for better semantic matching
collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

def extract_code_chunks(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        source = file.read()
    
    try:
        tree = ast.parse(source)
    except Exception as e:
        print(f"Skipping {filepath} due to parsing error: {e}")
        return []

    chunks = []
    
    # Extract functions and classes
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = node.end_lineno
            chunk_code = "\n".join(source.splitlines()[start_line-1:end_line])
            
            # Create a descriptive header
            if isinstance(node, ast.ClassDef):
                type_name = "Class"
            else:
                type_name = "Function"
                
            header = f"{type_name}: {node.name}"
            chunks.append({
                "code": chunk_code,
                "header": header,
                "filepath": filepath,
                "start_line": start_line,
                "end_line": end_line
            })
            
    return chunks

def ingest_folder(folder_path):
    print(f"Scanning folder: {folder_path}...")
    documents = []
    metadatas = []
    ids = []
    
    for root, _, files in os.walk(folder_path):
        # Skip hidden directories, venv, chroma_db
        if any(part.startswith('.') or part in ['venv', 'env', 'chroma_db', '__pycache__'] for part in root.split(os.sep)):
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                print(f"Processing: {filepath}")
                chunks = extract_code_chunks(filepath)
                
                for i, chunk in enumerate(chunks):
                    doc_id = f"{filepath}_{chunk['header']}_{i}"
                    documents.append(chunk['code'])
                    metadatas.append({
                        "filepath": chunk['filepath'],
                        "header": chunk['header'],
                        "start_line": chunk['start_line'],
                        "end_line": chunk['end_line']
                    })
                    ids.append(doc_id)
                    
    if documents:
        print(f"Embedding and storing {len(documents)} chunks...")
        # explicitly embed using SentenceTransformers
        embeddings = model.encode(documents).tolist()
        
        # Batch upload to ChromaDB
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            collection.upsert(
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print("Ingestion complete!")
    else:
        print("No Python files or parsable code chunks found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a folder of Python files into the semantic search engine.")
    parser.add_argument("folder", type=str, help="The folder path to scan and index.")
    args = parser.parse_args()
    
    if os.path.exists(args.folder):
        ingest_folder(args.folder)
    else:
        print(f"Error: Folder '{args.folder}' does not exist.")
