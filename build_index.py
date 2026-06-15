import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(".", "data")
COLLECTION_PATH = os.path.join(DATA_DIR, "collection", "collection.tsv")
INDEX_DIR = os.path.join(".", "index")
os.makedirs(INDEX_DIR, exist_ok=True)

MAX_PASSAGES = 100_000
BATCH_SIZE = 512  # tune up if you have VRAM to spare

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

print(f"Reading {MAX_PASSAGES} passages...")
chunk_ids = []
chunks = []

with open(COLLECTION_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            chunk_ids.append(parts[0])
            chunks.append(parts[1])
        if len(chunks) >= MAX_PASSAGES:
            break

print(f"Encoding {len(chunks)} passages in batches of {BATCH_SIZE}...")
embeddings = model.encode(
    chunks,
    batch_size=BATCH_SIZE,
    normalize_embeddings=True,   # unit vectors → cosine sim = dot product
    show_progress_bar=True,
    device='cuda'
)

print("Building FAISS index...")
dim = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
index = faiss.IndexFlatIP(dim)  # inner product = cosine sim on unit vectors
index.add(embeddings.astype(np.float32))

print("Saving index and IDs...")
faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
np.save(os.path.join(INDEX_DIR, "ids.npy"), np.array(chunk_ids))
np.save(os.path.join(INDEX_DIR, "texts.npy"), np.array(chunks))

print(f"Done. Index contains {index.ntotal} vectors.")