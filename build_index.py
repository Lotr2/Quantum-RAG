import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Setup paths based on your D: drive structure
CRUX_ROOT = r"D:\crux_datasets\crux"
os.environ["CRUX_ROOT"] = CRUX_ROOT

# Add CRUX code to Python path so 'import crux' works without pip install
sys.path.append(r"D:\crux")

from crux.tools.mds.ir_utils import load_data

# 2. Load the CRUX DUC04 dataset metadata (Topics, Qrels)
print("Loading CRUX DUC04 evaluation data...")
data = load_data(subset="duc04")

# Grab the FIRST bundle (Topic 0)
bundle = data.iloc[0]
topic_id = bundle.name  # e.g., 'duc04-test-0'
query_text = bundle['topic']

print(f"\n--- Target Bundle ---")
print(f"Topic ID: {topic_id}")
print(f"Query: {query_text[:80]}...")

# 3. Load the specific candidate passages for this bundle
# Note: You mentioned cloning crux-mds-corpus earlier. Update this path if it's stored elsewhere!
CORPUS_PATH = r"D:\crux_datasets\crux-mds-corpus\crux-mds-duc04\corpus.jsonl" 

chunk_ids = []
chunks = []

print(f"\nExtracting candidate passages for {topic_id} from corpus...")
with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        doc = json.loads(line)
        # We only want chunks belonging to this specific topic
        # CRUX chunk IDs look like "duc04-test-0#56"
        if doc['id'].startswith(topic_id):
            chunk_ids.append(doc['id'])
            chunks.append(doc['contents']) 

print(f"Found {len(chunks)} candidate passages for this bundle.")

# 4. Embed the Query and the Passages
print("\nLoading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

print("Embedding query and candidate passages...")
# Embed the user prompt (Topic)
query_embedding = model.encode(query_text, normalize_embeddings=True)

# Embed the candidate chunks
passage_embeddings = model.encode(chunks, normalize_embeddings=True)

# 5. Compute the Matrices for your QUBO!
print("\nComputing matrices for QUBO formulation...")

# Relevance Matrix (Query-to-Passage) -> LINEAR TERMS
# Shape: (num_passages,) - Higher score means more relevant
relevance_scores = np.dot(passage_embeddings, query_embedding)

# Redundancy Matrix (Passage-to-Passage) -> QUADRATIC TERMS
# Shape: (num_passages, num_passages) - Higher score means they are duplicates
similarity_matrix = np.dot(passage_embeddings, passage_embeddings.T)

print(f"Linear terms (Relevance) shape: {relevance_scores.shape}")
print(f"Quadratic terms (Redundancy) shape: {similarity_matrix.shape}")

# Optional: Save these arrays so you don't have to re-embed while tweaking your QUBO math
# np.save("linear_terms.npy", relevance_scores)
# np.save("quadratic_terms.npy", similarity_matrix)

print("\nSuccess! You can now map `relevance_scores` and `similarity_matrix` directly to your Quantum Annealer variables.")