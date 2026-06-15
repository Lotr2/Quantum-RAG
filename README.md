# Quantum-RAG
A two-stage retrieval pipeline that combines dense vector search (FAISS +
SBERT) with QUBO/Ising-based reranking (simulated annealing via `neal`) to
select a diverse, relevant set of context chunks for RAG.

## How it works

1. **Retrieval**: A query is encoded with `all-MiniLM-L6-v2` and matched
   against a FAISS index of pre-encoded passages using cosine similarity,
   returning the top-`K` candidates.
2. **Reranking**: The candidates are formulated as a QUBO problem, where the
   diagonal terms reward relevance to the query and the off-diagonal terms
   penalize redundancy between candidates. A cardinality constraint enforces
   selecting exactly `K_FINAL` chunks. The problem is solved via simulated
   annealing.

The two-stage design exists to accommodate future implementation on quantum
technology that is limited by hardware constraints. The QUBO matrix grows as
O(n^2), which increases the hardware demand and is not scalable for large n.
Dense retrieval narrows the field first; QUBO then optimizes for relevance
*and* diversity jointly, which similarity-only ranking can't do.

## Setup

This repo does **not** include the MS MARCO data or a pre-built FAISS index —
both are too large for GitHub. You'll need to download the data and build the
index yourself (takes a few minutes on GPU, longer on CPU).

1. Download the MS MARCO Passage Ranking dataset from the
   [official source](https://microsoft.github.io/msmarco/Datasets.html#passage-ranking-dataset).
   You'll need:
   - `collection.tsv` (the full passage collection, ~8.8M passages)
   - `queries.dev.tsv` (dev query set)

2. Place these under `data/` matching this structure:
data/
├── collection/collection.tsv
└── queries/queries.dev.tsv
3. Install dependencies and build the index:
```bash
   pip install -r requirements.txt
   python build_index.py
```

   By default this encodes the **first 100,000 passages** (`MAX_PASSAGES`
   in the script) — adjust this constant to index more or fewer. On a CUDA
   GPU, encoding 100K passages takes roughly a few minutes; on CPU this can
   take significantly longer (expect 30+ minutes depending on hardware).

   This produces `index/index.faiss`, `index/ids.npy`, and `index/texts.npy`.

4. Run the reranker:
```bash
   python chunk_retrieval_cos.py
```

   This picks a random query from `data/queries/queries.dev.tsv`, retrieves
   the top 100 candidates from the index, and prints the final reranked
   chunks with their cosine scores.

## A note on result quality

By default, the index covers only the **first 100,000 passages** of MS MARCO
(~1.1% of the full 8.8M-passage collection). This means:
- The top-`K` candidates retrieved for a given query may not match what
  you'd get from the full corpus — the true most-relevant passage for a
  query might simply not be in this subset.
- Results from this repo should be treated as a **pipeline demonstration** and
  a research step towards quantum implementation, not a benchmark of
  retrieval or reranking quality on MS MARCO.

To get more representative results, increase `MAX_PASSAGES` in
`build_index.py` and rebuild the index — though note this requires
significant disk space and encoding time as it approaches the full
collection.

## Known issues / in progress

- **Evaluation (nDCG@k) is not yet functional.** The code includes a
  commented-out qrels-loading and nDCG computation block. The MS MARCO
  qrels file (`qrels.dev.tsv`) does not reliably contain relevance
  judgments for queries/passages within the 100K-passage subset, so nDCG
  scores against it would be misleading. This will be revisited once
  evaluation is run against the full collection or a qrels subset aligned
  with the indexed passages.

## Configuration

Key parameters in `chunk_retrieval_cos.py`:

| Parameter | Description | Default |
|---|---|---|
| `TOP_K` | Number of candidates retrieved before reranking | 100 |
| `K_FINAL` | Number of chunks selected by the QUBO solver | 5 |
| `ALPHA` | Weight balancing relevance vs. redundancy | 0.95 |
| `penalty` | Strength of the cardinality constraint | 1.0 |