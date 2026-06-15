import os
import random
import numpy as np
import faiss
import neal
from sentence_transformers import SentenceTransformer

random.seed(42)
np.random.seed(42)

DATA_DIR     = os.path.join(".", "data")
INDEX_DIR    = os.path.join(".", "index")
QUERIES_PATH = os.path.join(DATA_DIR, "queries", "queries.dev.tsv")
QRELS_PATH   = os.path.join(DATA_DIR, "qrels", "qrels.dev.tsv")

TOP_K   = 100
K_FINAL = 5
ALPHA   = 0.95
penalty = 1.0

def ndcg_at_k(selected_ids, relevant_ids, k):
    dcg = 0.0
    for rank, pid in enumerate(selected_ids[:k], 1):
        if str(pid) in relevant_ids:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

print("Loading index and model...")
index     = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
chunk_ids = np.load(os.path.join(INDEX_DIR, "ids.npy"))
chunks    = np.load(os.path.join(INDEX_DIR, "texts.npy"))
model     = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')


# # --- Load qrels ---
# print("Loading qrels...")
# qrels = {}
# with open(QRELS_PATH, 'r') as f:
#     for line in f:
#         parts = line.strip().split('\t')
#         qid, _, pid, rel = parts[0], parts[1], parts[2], int(parts[3])
#         if rel > 0:
#             if qid not in qrels:
#                 qrels[qid] = set()
#             qrels[qid].add(pid)

# --- Encode query and retrieve top-k candidates ---

print("Processing query and retrieving candidates...")
with open(QUERIES_PATH, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f, start=1):
        if random.random() < 1 /idx:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                query_id, query =parts[0], parts[1]


query_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
scores, indices = index.search(query_emb, TOP_K)

cand_embs  = np.array([index.reconstruct(int(i)) for i in indices[0]])
cand_ids   = chunk_ids[indices[0]]
cand_texts = chunks[indices[0]]

print(f"Retrieved {len(cand_texts)} candidates via cosine search.")

# --- Build QUBO ---
n = len(cand_embs)
QUBO = np.zeros((n, n))

for i in range(n):
    QUBO[i][i] = -ALPHA * float(cand_embs[i] @ query_emb[0])
    QUBO[i][i] += penalty * (1 - 2 * K_FINAL)
    for j in range(i+1, n):
        QUBO[i][j] = (1 - ALPHA) * float(cand_embs[i] @ cand_embs[j])
        QUBO[i][j] += 2 * penalty

# --- Solve ---
sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample_qubo(QUBO, num_reads=100)
best_sample = sampleset.first.sample

print(f"\nMinimum energy: {sampleset.first.energy:.4f}")

# --- Rank selected chunks by cosine score for nDCG ---
selected = [(node, cand_ids[node], cand_texts[node], float(cand_embs[node] @ query_emb[0]))
            for node, val in best_sample.items() if val == 1]
selected.sort(key=lambda x: x[3], reverse=True)

# --- Display ---
print("\n--- Query ---")
print(f"Query ID: {query_id}")
print(f"Query: {query}")
print(f"\n--- Final RAG context ({K_FINAL} chunks) ---")
for rank, (node, cid, text, cos_score) in enumerate(selected, 1):
    print(f"\n[{rank}] Chunk ID {cid} (score: {cos_score:.4f}): {text}")

# print(f"\nnDCG@{K_FINAL}: {score:.4f}")