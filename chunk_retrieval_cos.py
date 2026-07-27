import os
import sys
import json
import pickle
import importlib.util
import numpy as np
import openjij as oj
import subprocess
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils import parse_crux_output, save_results, plot_all_metrics, plot_chunk_count, plot_mmr_vs_alpha, plot_mmr_comparison, plot_ice_vs_alpha, plot_snu_vs_alpha

# ─── Configuration ───────────────────────────────────────────────────────

CRUX_ROOT = r"D:\lunacy\Em#loyed\Quantum\Code\Version 2.0\crux_datasets\crux"
CRUX_CODE = r"D:\lunacy\Em#loyed\Quantum\Code\Version 2.0\crux"
CORPUS_PATH = r"D:\lunacy\Em#loyed\Quantum\Code\Version 2.0\crux_datasets\crux-mds-corpus-fast\collections\duc04-test-docs.jsonl"

os.environ["CRUX_ROOT"] = CRUX_ROOT
sys.path.append(CRUX_CODE)

# ─── Direct import for redundancy metrics (lives in Version 2.1 crux) ─────

_REDUndancy_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'crux', 'crux', 'evaluation', 'redundancy_metrics.py'
)
_spec = importlib.util.spec_from_file_location("redundancy_metrics", _REDUndancy_METRICS_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["redundancy_metrics"] = _mod
_spec.loader.exec_module(_mod)
compute_redundancy_metrics = _mod.compute_redundancy_metrics

RUN_FILE_PATH = "qubo_duc04_run.txt"
QREL_PATH = os.path.join(CRUX_ROOT, "crux-mds-duc04", "qrels", "div_qrels-tau3.txt")
JUDGE_PATH = os.path.join(CRUX_ROOT, "crux-mds-duc04", "judge", "ratings.Llama-3.1-70B-Instruct.0-1.jsonl")
CACHE_FILE_PATH = "score_cache.pkl"
GRADED_QREL_PATH = os.path.join(CRUX_ROOT, "crux-mds-duc04", "qrels", "legacy", "qrels.txt")
JUDGE_V2_PATH = os.path.join(CRUX_ROOT, "crux-mds-duc04", "judge", "v2", "ratings.Llama-3.3-70B-Instruct.0-1.jsonl")
MAX_REL_GRADE = 3

K_FINAL = 5
N_SA_READS = 100
N_ITERATIONS = 1
# ALPHA_GRID = [round(a, 3) for a in np.linspace(0.6, 0.9, 20)]
# # ALPHA_GRID = [0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9]
# ALPHA_GRID = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
ALPHA_GRID = [0.5]
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
    print("cand_ids: ",cand_ids)
    selected = [
        (cand_ids[node], float(raw_rel_scores[node]))
        for node, val in best_sample.items() if val == 1
    ]
    # selected.sort(key=lambda x: x[1], reverse=True)
    #this line apparently sorts b
    return selected


def write_run_entries(selected, run_file, topic_id):
    for rank, (cid, score) in enumerate(selected, 1):
        run_file.write(f"{topic_id} Q0 {cid} {rank} {score:.4f} QUBO_Annealer\n")

# ─── SNU / ICE Helpers ──────────────────────────────────────────────────────

def build_passage_answer_map(selected, topic_ratings, tau=3):
    """Build a passage_answer_map for one topic from raw per-subquestion ratings.

    Args:
        selected: List of (chunk_id, score) tuples from QUBO selection.
        topic_ratings: Dict mapping chunk_id -> list of per-subquestion ratings.
        tau: Threshold above which a rating counts as "answered".

    Returns:
        List[Set[int]] where element k is the set of sub-question indices
        answered by the k-th selected passage.
    """
    answer_map = []
    for chunk_id, _ in selected:
        ratings = topic_ratings.get(chunk_id)
        if ratings is None:
            answer_map.append(set())
        else:
            answer_map.append({i for i, r in enumerate(ratings) if r >= tau})
    return answer_map

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

    print("printing embs structure",cand_embs)
    print("cand_embs len: ", len(cand_embs))
    print("cand len: ", len(candidates))

    raw_rel, norm_rel = compute_relevance(cand_embs, query_emb)
    raw_redun, norm_redun, pair_indices = compute_redundancy(cand_embs, n)

    qubo_dict = build_qubo_dict(n, alpha, norm_rel, norm_redun, pair_indices)
    best = sample_qubo(sampler, qubo_dict)
    print("best sample from QUBO:", best)
    selected = extract_selected(best, cand_ids, raw_rel)
    selected_emb = [cand_embs[cand_ids.index(cid)] for cid, _ in selected]
    # print(f"Topic {topic_id}: Selected {len(selected)} chunks from {n} (Target: {K_FINAL})")

    return selected, cache_updated, selected_emb, query_emb

# ─── Alpha Sweep ──────────────────────────────────────────────────────────


def run_alpha_sweep(data, corpus_by_topic, model, sampler, cache, judge_scores=None,
                    raw_ratings=None, subquestions=None, snu_tau=3, snu_lambda=0.5):
    results = []

    for alpha in ALPHA_GRID:
        total_chunks = 0
        n_topics = 0
        mmr_scores = []
        mmr_raw_scores = []
        all_snu = []
        all_ice = []
        all_coverage = []

        with open(RUN_FILE_PATH, "w", encoding="utf-8") as run_file:
            for _, row in data.head(N_ITERATIONS).iterrows():
                topic_id = str(row["id"]) if "id" in row else str(_)
                query_text = row["topic"]
                candidates = corpus_by_topic.get(topic_id, [])

                if not candidates:
                    continue

                selected, cache_updated, selected_embs, query_embds = process_topic(
                    model, sampler, topic_id, query_text, candidates, alpha, cache
                )
                print(selected)
                if cache_updated:
                    save_cache(cache, CACHE_FILE_PATH)
                total_chunks += len(selected)
                n_topics += 1
                score = accumuate_MMR(query_embds, selected_embs)
                mmr_scores.append(score)
                if judge_scores:
                    topic_judge = judge_scores.get(topic_id, {})
                    score_raw = accumuate_MMR_raw_rel(topic_judge, selected, selected_embs)
                    mmr_raw_scores.append(score_raw)
                # write_run_entries(selected, run_file, topic_id)

                if raw_ratings is not None and subquestions is not None:
                    topic_ratings = raw_ratings.get(topic_id, {})
                    topic_sqs = subquestions.get(topic_id, [])
                    answer_map = build_passage_answer_map(selected, topic_ratings, tau=snu_tau)
                    metrics = compute_redundancy_metrics(topic_sqs, answer_map, lam=snu_lambda)
                    all_snu.append(metrics.snu)
                    all_ice.append(metrics.ice)
                    all_coverage.append(metrics.coverage)

        print(f"\nFinished processing. Results saved to {RUN_FILE_PATH}")
        print(f"alpha: {alpha}")
        print(f"average selected chunks: {total_chunks / n_topics:.1f}")

        crux_metrics = run_crux_eval(RUN_FILE_PATH, QREL_PATH, JUDGE_PATH)
        mean_mmr = float(np.mean(mmr_scores)) if mmr_scores else 0.0
        result = {"alpha": alpha, "mean_chunks": total_chunks / n_topics, "MMR": mean_mmr}
        if mmr_raw_scores:
            result["MMR_raw"] = float(np.mean(mmr_raw_scores))
        if all_snu:
            result["SNU"] = float(np.mean(all_snu))
            result["ICE"] = float(np.mean(all_ice))
            result["Coverage"] = float(np.mean(all_coverage))
        result.update(crux_metrics)
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
    score = MMR_LAMBDA * rel - (1-MMR_LAMBDA) * redun
    return score

def accumuate_MMR(query_emp, selected_chunks_emp):
    score =0
    for i in range(len(selected_chunks_emp)):
        score = score+ evaluate_MMR(query_emp, selected_chunks_emp[:i+1])
        print(f"MMR score after selecting {i+1} chunks: {score:.4f}")
    print(f"Total Cumulative MMR Metric: {score:.4f}")
    return score

def find_MMR_Solution(cands_emb, query_emp, k=10):
    selected={}
    for _ in range(k):
        for i in range(len(cands_emb)):
            if i in selected:
                continue
            score = evaluate_MMR(query_emp, [cands_emb[j] for j in selected] + [cands_emb[i]])
            if score > best_score:
                best_score = score
                best_index = i  
        selected.append(best_index)
    return selected
def extract_MMR_solution(cands_emb, query_emp, k=10):
    selected_indices = find_MMR_Solution(cands_emb, query_emp, k)
    selected_chunks = [cands_emb[i] for i in selected_indices]
    return selected_chunks



def evaluate_MMR_raw_rel(judge_scores, chunk_id, selected_chunks_emp):
    if not selected_chunks_emp:
        return 0.0
    rel = judge_scores.get(chunk_id, 0.0)
    current_emb = selected_chunks_emp[-1]
    redun = 0
    for i in range(len(selected_chunks_emp) - 1):
        redun = max(redun, current_emb @ selected_chunks_emp[i])
    score = MMR_LAMBDA * rel - (1-MMR_LAMBDA) * redun
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
# 

# ─── Main ─────────────────────────────────────────────────────────────────

def print_random_topic(corpus_by_topic, subquestions=None, raw_ratings=None, data=None, seed=None):
    """Print one random topic with all its candidate passages, subquestions, and ratings."""
    import random
    if seed is not None:
        random.seed(seed)

    topic_id = random.choice(list(corpus_by_topic.keys()))
    passages = corpus_by_topic[topic_id]

    print("=" * 80)
    print(f"TOPIC: {topic_id}")
    print("=" * 80)

    if data is not None:
        if "id" in data.columns:
            matching = data[data["id"].astype(str) == topic_id]
        else:
            matching = data[data.index.astype(str) == topic_id]
        if not matching.empty:
            print(f"\nQUERY:\n  {matching.iloc[0]['topic']}")

    if subquestions and topic_id in subquestions:
        sqs = subquestions[topic_id]
        print(f"\nSUBQUESTIONS ({len(sqs)}):")
        for i, sq in enumerate(sqs):
            print(f"  [{i}] {sq}")

    print(f"\nPASSAGES ({len(passages)}):")
    print("-" * 80)
    for i, (chunk_id, text) in enumerate(passages):
        preview = " ".join(text.split()[:120])
        print(f"\n  [{i}] {chunk_id}")
        print(f"      {preview}{'...' if len(text.split()) > 120 else ''}")

        if raw_ratings and topic_id in raw_ratings:
            ratings = raw_ratings[topic_id].get(chunk_id)
            if ratings is not None:
                answered = [j for j, r in enumerate(ratings) if r >= 3]
                print(f"      ratings: {ratings}  (answers sq: {answered})")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    from crux.tools.mds.ir_utils import load_data
    from crux.tools import load_ratings
    from crux.tools.mds.ir_utils import load_subtopics

    print("Loading CRUX DUC04 evaluation data...")
    data = load_data(subset="duc04")

    print("Pre-loading corpus into memory...")
    corpus_by_topic = load_corpus(CORPUS_PATH)
    cache = load_or_create_cache(CACHE_FILE_PATH)
    graded_qrels = load_graded_qrels(GRADED_QREL_PATH)
    judge_scores = load_judge_v2(JUDGE_V2_PATH)
    print(f"Loaded graded relevance for {len(graded_qrels)} topics")
    print(f"Loaded judge v2 ratings for {len(judge_scores)} topics")

    print("Loading raw per-subquestion ratings for SNU/ICE...")
    raw_ratings = load_ratings(JUDGE_V2_PATH)
    subquestions = load_subtopics("duc04")
    print(f"Loaded raw ratings for {len(raw_ratings)} topics, subquestions for {len(subquestions)} topics")


    print("raw_ratings:", judge_scores)
    # print_random_topic(corpus_by_topic, subquestions, raw_ratings, data, seed=42)

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    sampler = oj.SASampler()

    print(f"Processing {len(data)} topics...")
    df = run_alpha_sweep(data, corpus_by_topic, model, sampler, cache, judge_scores,
                         raw_ratings=raw_ratings, subquestions=subquestions)

    plot_all_metrics(df, save_path="alpha_all_metrics.png")
    plot_chunk_count(df, k_target=K_FINAL, save_path="alpha_vs_chunk_count.png")
    plot_mmr_vs_alpha(df, save_path="alpha_vs_mmr.png")
    plot_snu_vs_alpha(df, save_path="alpha_vs_snu.png")
    plot_ice_vs_alpha(df, save_path="alpha_vs_ice.png")
    if "MMR_raw" in df.columns:
        plot_mmr_comparison(df, save_path="alpha_mmr_comparison.png")



