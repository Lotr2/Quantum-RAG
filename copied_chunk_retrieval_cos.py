import os
import sys
import json
import pickle 
import neal.sampler
import numpy as np
import neal
import openjij as oj
import subprocess
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from utils import parse_crux_output, save_results, plot_metric, plot_all_metrics, plot_chunk_count, print_full_matrix

CRUX_ROOT = r"D:\lunacy\Em#loyed\Quantum\Code\Version 2.0\crux_datasets\crux"
CRUX_CODE = r"D:\lunacy\Em#loyed\Quantum\Code\Version 2.0\crux"
CORPUS_PATH = r"D:\lunacy\Em#loyed\Quantum\Code\Version 2.0\crux_datasets\crux-mds-corpus-fast\collections\duc04-test-docs.jsonl" 

ITTERATIONS= 10  # Limit to first 10 topics for testing

os.environ["CRUX_ROOT"] = CRUX_ROOT
sys.path.append(CRUX_CODE)
from crux.tools.mds.ir_utils import load_data

RUN_FILE_PATH = "qubo_duc04_run.txt"
QREL_PATH = os.path.join(CRUX_ROOT, "crux-mds-duc04", "qrels", "div_qrels-tau3.txt")
JUDGE_PATH = os.path.join(CRUX_ROOT, "crux-mds-duc04", "judge", "ratings.Llama-3.1-70B-Instruct.0-1.jsonl")

CACHE_FILE_PATH = "score_cache.pkl" 

# QUBO Hyperparameters
K_FINAL = 5
ALPHA   = 0.85

# ==========================================
# 2. LOAD DATA, PREP CORPUS, & LOAD CACHE
# ==========================================
print("Loading CRUX DUC04 evaluation data...")
data = load_data(subset="duc04")

print("Pre-loading corpus into memory...")
corpus_by_topic = {}
with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        doc = json.loads(line)
        tid = doc['id'].split(':')[0]  
        
        if tid not in corpus_by_topic:
            corpus_by_topic[tid] = []
        corpus_by_topic[tid].append((doc['id'], doc['contents']))

if os.path.exists(CACHE_FILE_PATH):
    print("Found existing cache file. Loading saved calculations...")
    with open(CACHE_FILE_PATH, 'rb') as f:
        score_cache = pickle.load(f)
else:
    print("No cache found. Creating a new one...")
    score_cache = {"relevance": {}, "redundancy": {}, "bi_encoder": {}}

print("Loading SentenceTransformer model + cross encoder...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
sampler = neal.SimulatedAnnealingSampler()

# ==========================================
# 3. QUBO SELECTION LOOP
# ==========================================
print(f"Processing {len(data)} topics...")
alpha_grid = [round(a, 2) for a in np.linspace(0.8, 1, 10)]
results = []
for ALPHA in alpha_grid:
    with open(RUN_FILE_PATH, 'w', encoding='utf-8') as run_file:
        total_chunks=0
        n_topics=0

        for index, row in data.head(ITTERATIONS).iterrows():
            topic_id = str(index) if 'id' not in row else str(row['id'])
            query_text = row['topic']
            
            candidates = corpus_by_topic.get(topic_id, [])
            if not candidates:
                continue
                
            cand_ids = [c[0] for c in candidates]
            cand_texts = [c[1] for c in candidates]
            n = len(cand_texts)
            
            cache_updated = False # Track if we need to save the file
            
            if topic_id in score_cache["bi_encoder"]:
                cand_embs = score_cache["bi_encoder"][topic_id]
            else:
                cand_embs = model.encode(cand_texts, normalize_embeddings=True).astype(np.float32)
                score_cache["bi_encoder"][topic_id] = cand_embs
                cache_updated = True

            # NEW: embed the query with the same bi-encoder so we can get cosine similarity
            query_emb = model.encode([query_text], normalize_embeddings=True).astype(np.float32)[0]

            # --- CROSS-ENCODER RELEVANCE ---
            if topic_id in score_cache["relevance"]:
                raw_ce_scores = score_cache["relevance"][topic_id]
            else:
                cross_inp = [[query_text, cand] for cand in cand_texts]
                raw_ce_scores = cross_encoder.predict(cross_inp)
                score_cache["relevance"][topic_id] = raw_ce_scores
                cache_updated = True

            norm_ce_scores = 1 / (1 + np.exp(-raw_ce_scores))

            # --- COSINE SIMILARITY (commented out for now) ---
            # raw_ce_scores = cand_embs @ query_emb
            # norm_ce_scores = (raw_ce_scores + 1) / 2   # map to [0, 1] to match old sigmoid range
            
            pair_indices = []
            for i in range(n):
                for j in range(i + 1, n):
                    pair_indices.append((i, j))

            # --- CROSS-ENCODER REDUNDANCY ---
            if topic_id in score_cache["redundancy"]:
                raw_ce_scores_redun = score_cache["redundancy"][topic_id]
            else:
                cross_inp_redun = [[cand_texts[i], cand_texts[j]] for (i, j) in pair_indices]
                if cross_inp_redun:
                    raw_ce_scores_redun = cross_encoder.predict(cross_inp_redun)
                else:
                    raw_ce_scores_redun = np.array([])

                score_cache["redundancy"][topic_id] = raw_ce_scores_redun
                cache_updated = True

            # --- COSINE SIMILARITY (commented out for now) ---
            # if pair_indices:
            #     raw_ce_scores_redun = np.array([cand_embs[i] @ cand_embs[j] for (i, j) in pair_indices])
            # else:
            #     raw_ce_scores_redun = np.array([])
                
            if len(raw_ce_scores_redun) > 0:
                norm_ce_scores_redun = 1 / (1 + np.exp(-raw_ce_scores_redun))    # map to [0, 1]
                
            # Save to disk immediately if new calculations were made
            if cache_updated:
                with open(CACHE_FILE_PATH, 'wb') as f:
                    pickle.dump(score_cache, f)
            
            QUBO_matrix = np.zeros((n, n))
            
            for i in range(n):
                QUBO_matrix[i][i] = -ALPHA * float(norm_ce_scores[i])
                
            if len(raw_ce_scores_redun) > 0:
                for idx, (i, j) in enumerate(pair_indices):
                    QUBO_matrix[i][j] = (1 - ALPHA) * float(norm_ce_scores_redun[idx])
            
            qubo_dict = {}
            for i in range(n):
                qubo_dict[(i, i)] = QUBO_matrix[i][i]
                for j in range(i + 1, n):
                    qubo_dict[(i, j)] = QUBO_matrix[i][j]
                    
            sampleset = sampler.sample_qubo(qubo_dict, num_reads=100)
            best_sample = sampleset.first.sample
            
            # Rank selected chunks by similarity score
            selected = [(node, cand_ids[node], float(raw_ce_scores[node]))
                        for node, val in best_sample.items() if val == 1]
            selected.sort(key=lambda x: x[2], reverse=True)

            total_chunks += len(selected)
            n_topics += 1
            print(f"Topic {topic_id}: Selected {len(selected)} chunks from {(n)} (Target: {K_FINAL})")

            for rank, (node, cid, ce_score) in enumerate(selected, 1):
                run_file.write(f"{topic_id} Q0 {cid} {rank} {ce_score:.4f} QUBO_Annealer\n")
            np.set_printoptions(precision=2, suppress=True)
    print(f"\nFinished processing. Results saved to {RUN_FILE_PATH}")

    # ==========================================
    # 4. RUN CRUX EVALUATION
    # ==========================================
    print("\n--- Running CRUX Answerability Evaluation ---\n")

    eval_command = [
        sys.executable, "-m", "crux.evaluation.rac_eval",
        "--run", RUN_FILE_PATH,
        "--qrel", QREL_PATH,
        "--filter_by_oracle",
        "--judge", JUDGE_PATH,
        "--k", "5"
    ]

    result = subprocess.run(eval_command, capture_output=True, text=True)
    print("alpha: ",ALPHA)
    print("average selected chunks:", total_chunks / ITTERATIONS)
    print(result.stdout)
    if result.stderr:
        print("Evaluation Logs/Errors:", result.stderr)
    metrics = parse_crux_output(result.stdout)               # <-- THIS is "what goes here"
    results.append({"alpha": ALPHA, "mean_chunks": total_chunks / n_topics, **metrics})
    print("iterations:", ITTERATIONS)

df = save_results(results, "alpha_sweep_results.csv")
# plot_metric(df, "alpha_nDCG@5", save_path="alpha_vs_alpha_ndcg.png")
plot_all_metrics(df, save_path="alpha_all_metrics.png")
plot_chunk_count(df, k_target=K_FINAL, save_path="alpha_vs_chunk_count.png")