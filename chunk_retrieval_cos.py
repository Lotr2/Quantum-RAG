import os
import sys
import json
import pickle
import numpy as np
import openjij as oj
import subprocess
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils import parse_crux_output, save_results, plot_all_metrics, plot_chunk_count

# ─── Configuration ───────────────────────────────────────────────────────

CRUX_ROOT = r"D:\lunacy\Em#loyed\Quantum\Code\Version 2.0\crux_datasets\crux"
CRUX_CODE = r"D:\lunacy\Em#loyed\Quantum\Code\Version 2.0\crux"
CORPUS_PATH = r"D:\lunacy\Em#loyed\Quantum\Code\Version 2.0\crux_datasets\crux-mds-corpus-fast\collections\duc04-test-docs.jsonl"

os.environ["CRUX_ROOT"] = CRUX_ROOT
sys.path.append(CRUX_CODE)

RUN_FILE_PATH = "qubo_duc04_run.txt"
QREL_PATH = os.path.join(CRUX_ROOT, "crux-mds-duc04", "qrels", "div_qrels-tau3.txt")
JUDGE_PATH = os.path.join(CRUX_ROOT, "crux-mds-duc04", "judge", "ratings.Llama-3.1-70B-Instruct.0-1.jsonl")
CACHE_FILE_PATH = "score_cache.pkl"

K_FINAL = 5
N_SA_READS = 100
N_ITERATIONS = 50
ALPHA_GRID = [round(a, 2) for a in np.linspace(0.75, 1, 10)]

# ─── Data Loading ────────────────────────────────────────────────────────


def load_corpus(path):
    corpus_by_topic = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            tid = doc["id"].split(":")[0]
            corpus_by_topic.setdefault(tid, []).append((doc["id"], doc["contents"]))
    return corpus_by_topic


def load_or_create_cache(path):
    if os.path.exists(path):
        print("Found existing cache file. Loading saved calculations...")
        with open(path, "rb") as f:
            return pickle.load(f)
    print("No cache found. Creating a new one...")
    return {"bi_encoder": {}}


def save_cache(cache, path):
    with open(path, "wb") as f:
        pickle.dump(cache, f)

# ─── Embedding & Scoring ────────────────────────────────────────────────


def encode_candidates(model, texts, topic_id, cache):
    if topic_id in cache["bi_encoder"]:
        return cache["bi_encoder"][topic_id], False
    embs = model.encode(texts, normalize_embeddings=True).astype(np.float32)
    cache["bi_encoder"][topic_id] = embs
    return embs, True


def compute_relevance(cand_embs, query_emb):
    raw = cand_embs @ query_emb
    norm = abs(raw)
    return raw, norm


def compute_redundancy(cand_embs, n):
    pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pair_indices:
        return np.array([]), np.array([]), pair_indices
    raw = np.array([cand_embs[i] @ cand_embs[j] for i, j in pair_indices])
    norm = abs(raw)
    return raw, norm, pair_indices

# ─── QUBO Construction & Sampling ────────────────────────────────────────


def build_qubo_dict(n, alpha, rel_norm, redun_norm, pair_indices):
    qubo = {}
    for i in range(n):
        qubo[(i, i)] = -alpha * float(rel_norm[i])
    for idx, (i, j) in enumerate(pair_indices):
        qubo[(i, j)] = (1 - alpha) * float(redun_norm[idx])
    return qubo


def sample_qubo(sampler, qubo_dict):
    sampleset = sampler.sample_qubo(qubo_dict, num_reads=N_SA_READS)
    return sampleset.first.sample

# ─── Selection & Output ──────────────────────────────────────────────────


def extract_selected(best_sample, cand_ids, raw_rel_scores):
    selected = [
        (cand_ids[node], float(raw_rel_scores[node]))
        for node, val in best_sample.items() if val == 1
    ]
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected


def write_run_entries(selected, run_file, topic_id):
    for rank, (cid, score) in enumerate(selected, 1):
        run_file.write(f"{topic_id} Q0 {cid} {rank} {score:.4f} QUBO_Annealer\n")

# ─── CRUX Evaluation ─────────────────────────────────────────────────────


def run_crux_eval(run_path, qrel_path, judge_path, k=5):
    command = [
        sys.executable, "-m", "crux.evaluation.rac_eval",
        "--run", run_path,
        "--qrel", qrel_path,
        "--filter_by_oracle",
        "--judge", judge_path,
        "--k", str(k),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Evaluation Logs/Errors:", result.stderr)
    return parse_crux_output(result.stdout)

# ─── Per-Topic Processing ────────────────────────────────────────────────


def process_topic(model, sampler, topic_id, query_text, candidates, alpha, cache):
    cand_ids = [c[0] for c in candidates]
    cand_texts = [c[1] for c in candidates]
    n = len(cand_texts)

    cand_embs, cache_updated = encode_candidates(model, cand_texts, topic_id, cache)
    query_emb = model.encode([query_text], normalize_embeddings=True).astype(np.float32)[0]

    raw_rel, norm_rel = compute_relevance(cand_embs, query_emb)
    raw_redun, norm_redun, pair_indices = compute_redundancy(cand_embs, n)

    qubo_dict = build_qubo_dict(n, alpha, norm_rel, norm_redun, pair_indices)
    best = sample_qubo(sampler, qubo_dict)

    selected = extract_selected(best, cand_ids, raw_rel)
    print(f"Topic {topic_id}: Selected {len(selected)} chunks from {n} (Target: {K_FINAL})")

    return selected, cache_updated

# ─── Alpha Sweep ──────────────────────────────────────────────────────────


def run_alpha_sweep(data, corpus_by_topic, model, sampler, cache):
    results = []

    for alpha in ALPHA_GRID:
        total_chunks = 0
        n_topics = 0

        with open(RUN_FILE_PATH, "w", encoding="utf-8") as run_file:
            for _, row in data.head(N_ITERATIONS).iterrows():
                topic_id = str(row["id"]) if "id" in row else str(_)
                query_text = row["topic"]
                candidates = corpus_by_topic.get(topic_id, [])

                if not candidates:
                    continue

                selected, cache_updated = process_topic(
                    model, sampler, topic_id, query_text, candidates, alpha, cache
                )
                if cache_updated:
                    save_cache(cache, CACHE_FILE_PATH)

                total_chunks += len(selected)
                n_topics += 1
                write_run_entries(selected, run_file, topic_id)

        print(f"\nFinished processing. Results saved to {RUN_FILE_PATH}")
        print(f"alpha: {alpha}")
        print(f"average selected chunks: {total_chunks / n_topics:.1f}")

        metrics = run_crux_eval(RUN_FILE_PATH, QREL_PATH, JUDGE_PATH)
        results.append({"alpha": alpha, "mean_chunks": total_chunks / n_topics, **metrics})
        print("iterations:", N_ITERATIONS)

    return save_results(results, "alpha_sweep_results.csv")

# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from crux.tools.mds.ir_utils import load_data

    print("Loading CRUX DUC04 evaluation data...")
    data = load_data(subset="duc04")

    print("Pre-loading corpus into memory...")
    corpus_by_topic = load_corpus(CORPUS_PATH)
    cache = load_or_create_cache(CACHE_FILE_PATH)

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    sampler = oj.SASampler()

    print(f"Processing {len(data)} topics...")
    df = run_alpha_sweep(data, corpus_by_topic, model, sampler, cache)

    plot_all_metrics(df, save_path="alpha_all_metrics.png")
    plot_chunk_count(df, k_target=K_FINAL, save_path="alpha_vs_chunk_count.png")
