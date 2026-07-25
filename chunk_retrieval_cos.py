import os
import sys
import json
import pickle
import numpy as np
import openjij as oj
import subprocess
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils import parse_crux_output, save_results, plot_all_metrics, plot_chunk_count, plot_mmr_vs_alpha, plot_mmr_comparison

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
GRADED_QREL_PATH = os.path.join(CRUX_ROOT, "crux-mds-duc04", "qrels", "legacy", "qrels.txt")
JUDGE_V2_PATH = os.path.join(CRUX_ROOT, "crux-mds-duc04", "judge", "v2", "ratings.Llama-3.3-70B-Instruct.0-1.jsonl")
MAX_REL_GRADE = 3

K_FINAL = 5
N_SA_READS = 100
N_ITERATIONS = 50
ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.70, 0.8, 0.9]
# ALPHA_GRID = [0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9]
MMR_LAMBDA = 0.5

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


def load_graded_qrels(path):
    qrels = {}
    if not os.path.exists(path):
        return qrels
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                topic_id, _, chunk_id, grade = parts[0], parts[1], parts[2], int(parts[3])
                qrels.setdefault(topic_id, {})[chunk_id] = grade
    return qrels


def load_judge_v2(path):
    judge = {}
    if not os.path.exists(path):
        return judge
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            topic_id = entry["id"]
            chunk_id = entry["docid"]
            mean_rating = float(np.mean(entry["rating"])) / 5.0
            judge.setdefault(topic_id, {})[chunk_id] = mean_rating
    return judge

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
    selected_emb = [cand_embs[cand_ids.index(cid)] for cid, _ in selected]
    print(f"Topic {topic_id}: Selected {len(selected)} chunks from {n} (Target: {K_FINAL})")

    return selected, cache_updated, selected_emb, query_emb

# ─── Alpha Sweep ──────────────────────────────────────────────────────────


def run_alpha_sweep(data, corpus_by_topic, model, sampler, cache, judge_scores=None):
    results = []

    for alpha in ALPHA_GRID:
        total_chunks = 0
        n_topics = 0
        mmr_scores = []
        mmr_raw_scores = []

        with open(RUN_FILE_PATH, "w", encoding="utf-8") as run_file:
            for _, row in data.head(N_ITERATIONS).iterrows():
                topic_id = str(row["id"]) if "id" in row else str(_)
                query_text = row["topic"]
                candidates = corpus_by_topic.get(topic_id, [])

                if not candidates:
                    continue

                selected, cache_updated, cand_embs, query_embds = process_topic(
                    model, sampler, topic_id, query_text, candidates, alpha, cache
                )
                print(selected)
                if cache_updated:
                    save_cache(cache, CACHE_FILE_PATH)
                total_chunks += len(selected)
                n_topics += 1
                score = accumuate_MMR(query_embds, cand_embs)
                mmr_scores.append(score)
                if judge_scores:
                    topic_judge = judge_scores.get(topic_id, {})
                    score_raw = accumuate_MMR_raw_rel(topic_judge, selected, cand_embs)
                    mmr_raw_scores.append(score_raw)
                write_run_entries(selected, run_file, topic_id)

        print(f"\nFinished processing. Results saved to {RUN_FILE_PATH}")
        print(f"alpha: {alpha}")
        print(f"average selected chunks: {total_chunks / n_topics:.1f}")

        metrics = run_crux_eval(RUN_FILE_PATH, QREL_PATH, JUDGE_PATH)
        mean_mmr = float(np.mean(mmr_scores)) if mmr_scores else 0.0
        result = {"alpha": alpha, "mean_chunks": total_chunks / n_topics, "MMR": mean_mmr}
        if mmr_raw_scores:
            result["MMR_raw"] = float(np.mean(mmr_raw_scores))
        result.update(metrics)
        results.append(result)
        print("iterations:", N_ITERATIONS)

    return save_results(results, "alpha_sweep_results.csv")



# ─── MMR Evaluation Metric ─────────────────────────────────────────────────────────────────
def evaluate_MMR(query_emp, selected_chunks_emp):
    if(selected_chunks_emp == []):
        return 0.0
    recent_selected_chunk_emp = selected_chunks_emp[-1]
    rel = query_emp @ recent_selected_chunk_emp
    redun = 0
    for i in range(len(selected_chunks_emp) - 1):
        redun = max(redun, recent_selected_chunk_emp @ selected_chunks_emp[i])
    score = MMR_LAMBDA * rel - MMR_LAMBDA * redun
    return score

def accumuate_MMR(query_emp, selected_chunks_emp):
    score =0
    for i in range(len(selected_chunks_emp)):
        score = score+ evaluate_MMR(query_emp, selected_chunks_emp[:i+1])
        print(f"MMR score after selecting {i+1} chunks: {score:.4f}")
    print(f"Total Cumulative MMR Metric: {score:.4f}")
    return score


def evaluate_MMR_raw_rel(judge_scores, chunk_id, selected_chunks_emp):
    if not selected_chunks_emp:
        return 0.0
    rel = judge_scores.get(chunk_id, 0.0)
    current_emb = selected_chunks_emp[-1]
    redun = 0
    for i in range(len(selected_chunks_emp) - 1):
        redun = max(redun, current_emb @ selected_chunks_emp[i])
    score = MMR_LAMBDA * rel - MMR_LAMBDA * redun
    return score


def accumuate_MMR_raw_rel(judge_scores, selected, selected_chunks_emp):
    score = 0
    for i in range(len(selected)):
        chunk_id = selected[i][0]
        step = evaluate_MMR_raw_rel(judge_scores, chunk_id, selected_chunks_emp[:i+1])
        score += step
        print(f"MMR_raw score after selecting {i+1} chunks: {score:.4f}")
    print(f"Total Cumulative MMR_raw Metric: {score:.4f}")
    return score

# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from crux.tools.mds.ir_utils import load_data

    print("Loading CRUX DUC04 evaluation data...")
    data = load_data(subset="duc04")

    print("Pre-loading corpus into memory...")
    corpus_by_topic = load_corpus(CORPUS_PATH)
    cache = load_or_create_cache(CACHE_FILE_PATH)
    graded_qrels = load_graded_qrels(GRADED_QREL_PATH)
    judge_scores = load_judge_v2(JUDGE_V2_PATH)
    print(f"Loaded graded relevance for {len(graded_qrels)} topics")
    print(f"Loaded judge v2 ratings for {len(judge_scores)} topics")

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    sampler = oj.SASampler()

    print(f"Processing {len(data)} topics...")
    df = run_alpha_sweep(data, corpus_by_topic, model, sampler, cache, judge_scores)

    plot_all_metrics(df, save_path="alpha_all_metrics.png")
    plot_chunk_count(df, k_target=K_FINAL, save_path="alpha_vs_chunk_count.png")
    plot_mmr_vs_alpha(df, save_path="alpha_vs_mmr.png")
    if "MMR_raw" in df.columns:
        plot_mmr_comparison(df, save_path="alpha_mmr_comparison.png")
