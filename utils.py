import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Single source of truth for the cutoff used across parsing/plotting.
# Change this ONE value when you switch @5 <-> @10 <-> whatever, and every
# function below picks it up automatically.
K = 5

def metric_name(base: str, k: int | None = None) -> str:
    """Build a metric column name like 'nDCG@5' from a base name ('nDCG')."""
    return f"{base}@{k if k is not None else K}"


def print_full_matrix(matrix, precision=4, suppress_scientific=True):
    """
    Prints a NumPy matrix in its entirety without truncation.
    
    Parameters:
    - matrix: The NumPy array/matrix to print.
    - precision: Number of decimal places to show (default: 4).
    - suppress_scientific: If True, prints 0.0001 instead of 1e-4 (default: True).
    """
    with np.printoptions(
        threshold=sys.maxsize,
        precision=precision,
        suppress=suppress_scientific,
        linewidth=50
    ):
        print(np.asarray(matrix)) 
        
 
def parse_crux_output(stdout_text: str) -> dict:
    """
    Parse a CRUX eval stdout line, e.g.:
    'run.txt | run.txt | P@5 | 0.4280 | nDCG@5 | 0.5573 | alpha_nDCG@5 | 0.4661 | Cov@5 | 0.5191 |'
    into {'P@5': 0.4280, 'nDCG@5': 0.5573, 'alpha_nDCG@5': 0.4661, 'Cov@5': 0.5191}.

    This is already cutoff-agnostic -- it just reads whatever key/value pairs
    are actually in the line, so it needs no change when K changes.
    """
    parts = [p.strip() for p in stdout_text.split("|") if p.strip()]
    metrics = {}
    i = 0
    while i < len(parts) - 1:
        key, val = parts[i], parts[i + 1]
        try:
            metrics[key] = float(val)
            i += 2
        except ValueError:
            i += 1
    return metrics
 
 
def save_results(results: list[dict], path: str = "alpha_sweep_results.csv") -> pd.DataFrame:
    """
    Convert a list of per-alpha result dicts into a wide-format DataFrame
    (one row per alpha, one column per metric), sorted by alpha, and write it to CSV.
    """
    df = pd.DataFrame(results).sort_values("alpha").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df
 
 
def plot_metric(df: pd.DataFrame, metric: str | None = None, save_path: str | None = None, higher_is_better: bool = True):
    """
    Plot alpha (x) against a single chosen metric (y), marking the best alpha.
    This is the plot to use for the actual alpha* decision.

    If `metric` is omitted, defaults to 'alpha_nDCG@{K}' using the module-level K.
    """
    if metric is None:
        metric = metric_name("alpha_nDCG")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["alpha"], df[metric], marker="o", color="#2563eb")
 
    best_idx = df[metric].idxmax() if higher_is_better else df[metric].idxmin()
    best_alpha, best_val = df.loc[best_idx, "alpha"], df.loc[best_idx, metric]
    ax.scatter([best_alpha], [best_val], color="#dc2626", zorder=5,
               label=f"best: α={best_alpha:.2f}, {metric}={best_val:.4f}")
 
    ax.set_xlabel("alpha")
    ax.set_ylabel(metric)
    ax.set_title(f"alpha vs {metric}")
    ax.grid(alpha=0.3)
    ax.legend()
 
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
 
 
def plot_all_metrics(df: pd.DataFrame, metrics: list[str] | None = None, save_path: str | None = None):
    """
    Grid of alpha vs every metric, for the full trade-off picture (e.g. for a paper figure).
    Defaults to the four CRUX metrics at cutoff K if present.
    """
    if metrics is None:
        metrics = [m for m in
                   [metric_name("P"), metric_name("nDCG"), metric_name("alpha_nDCG"), metric_name("Cov"),
                    "SNU", "ICE"]
                   if m in df.columns]
 
    n = len(metrics)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
 
    for ax, metric in zip(axes, metrics):
        ax.plot(df["alpha"], df[metric], marker="o", color="#2563eb")
        best_idx = df[metric].idxmax()
        ax.scatter([df.loc[best_idx, "alpha"]], [df.loc[best_idx, metric]], color="#dc2626", zorder=5)
        ax.set_xlabel("alpha")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.3)
 
    for ax in axes[n:]:
        ax.axis("off")
 
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
 
 
def plot_chunk_count(df: pd.DataFrame, k_target: float | None = None, save_path: str | None = None):
    """
    Sanity-check plot: mean selected chunk count vs alpha. If you've added the
    cardinality penalty, this should sit roughly flat near k_target across the
    whole sweep -- if it drifts, your metric comparisons across alpha aren't
    apples-to-apples.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["alpha"], df["mean_chunks"], marker="o", color="#059669")
    if k_target is not None:
        ax.axhline(k_target, color="#dc2626", linestyle="--", label=f"K_FINAL = {k_target}")
        ax.legend()
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean selected chunks per topic")
    ax.set_title("Cardinality stability across alpha")
    ax.grid(alpha=0.3)
 
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
 
 
def plot_mmr_vs_alpha(df: pd.DataFrame, k_target: float | None = None, save_path: str | None = None):
    """
    Plot cumulative MMR score (y1) and mean chunks (y2) against alpha (x).
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # --- Primary Axis (MMR) ---
    line1 = ax.plot(df["alpha"], df["MMR"], marker="o", color="#7c3aed", label="MMR")
    best_idx = df["MMR"].idxmax()
    best_alpha, best_val = df.loc[best_idx, "alpha"], df.loc[best_idx, "MMR"]
    scat = ax.scatter([best_alpha], [best_val], color="#dc2626", zorder=5,
                      label=f"best: α={best_alpha:.2f}, MMR={best_val:.4f}")

    ax.set_xlabel("alpha")
    ax.set_ylabel("Cumulative MMR", color="#7c3aed")
    ax.tick_params(axis="y", labelcolor="#7c3aed")
    ax.set_title("alpha vs MMR & Mean Chunks")
    ax.grid(alpha=0.3)

    # --- Secondary Axis (Mean Chunks) ---
    ax2 = ax.twinx()  # Fixed: twinx() instead of twin()
    color2 = "tab:red"
    ax2.set_ylabel("Number of Chunks", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    
    # Fixed: passed df["alpha"] as x-axis
    line2 = ax2.plot(df["alpha"], df["mean_chunks"], color=color2, marker="s", label="Mean Chunks")

    lines_2, labels_2 = [], []
    if k_target is not None:
        line_k = ax2.axhline(k_target, color="#dc2626", linestyle="--", label=f"K_FINAL = {k_target}")
        lines_2 = [line_k]
        labels_2 = [f"K_FINAL = {k_target}"]
    
    ax.set_xticks(df["alpha"])

    # --- Combined Legend ---
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2_all, labels_2_all = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2_all, labels_1 + labels_2_all, loc="upper left")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_snu_vs_alpha(df: pd.DataFrame, k_target: float | None = None, save_path: str | None = None):
    """
    Plot Sub-question Net Utility (y1) and mean chunks (y2) against alpha (x).
    """
    if "SNU" not in df.columns:
        print("SNU column not found in DataFrame — skipping plot_snu_vs_alpha.")
        return None

    fig, ax = plt.subplots(figsize=(14, 10))

    # --- Primary Axis (SNU) ---
    line1 = ax.plot(df["alpha"], df["SNU"], marker="o", color="#7c3aed", label="SNU")
    best_idx = df["SNU"].idxmax()
    best_alpha, best_val = df.loc[best_idx, "alpha"], df.loc[best_idx, "SNU"]
    scat = ax.scatter([best_alpha], [best_val], color="#dc2626", zorder=5,
                      label=f"best: α={best_alpha:.2f}, SNU={best_val:.4f}")

    ax.set_xlabel("alpha")
    ax.set_ylabel("SNU", color="#7c3aed")
    ax.tick_params(axis="y", labelcolor="#7c3aed")
    ax.set_title("alpha vs Sub-question Net Utility & Mean Chunks")
    ax.grid(alpha=0.3)

    # --- Secondary Axis (Mean Chunks) ---
    ax2 = ax.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Number of Chunks", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    
    line2 = ax2.plot(df["alpha"], df["mean_chunks"], color=color2, marker="s", label="Mean Chunks")

    if k_target is not None:
        ax2.axhline(k_target, color="#dc2626", linestyle="--", label=f"K_FINAL = {k_target}")

    ax.set_xticks(df["alpha"])

    # --- Combined Legend ---
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2_all, labels_2_all = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2_all, labels_1 + labels_2_all, loc="upper left")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig



def plot_redundancy_metrics(df: pd.DataFrame, save_path: str | None = None):
    """
    Grid of alpha vs SNU, ICE, and Coverage — the three sub-question
    redundancy / coverage metrics.
    """
    available = [m for m in ["SNU", "ICE", "Coverage"] if m in df.columns]
    if not available:
        print("No redundancy metrics (SNU, ICE, Coverage) found in DataFrame columns.")
        return None

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes = axes if n > 1 else [axes]

    colors = {"SNU": "#7c3aed", "ICE": "#ea580c", "Coverage": "#059669"}
    for ax, metric in zip(axes, available):
        ax.plot(df["alpha"], df[metric], marker="o", color=colors.get(metric, "#2563eb"))
        best_idx = df[metric].idxmax()
        ax.scatter([df.loc[best_idx, "alpha"]], [df.loc[best_idx, metric]],
                   color="#dc2626", zorder=5,
                   label=f"best: alpha={df.loc[best_idx, 'alpha']:.2f}")
        ax.set_xlabel("alpha")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ice_vs_alpha(df: pd.DataFrame, k_target: float | None = None, save_path: str | None = None):
    """
    Plot Incremental Coverage Efficiency (y1) and mean chunks (y2) against alpha (x).
    """
    if "ICE" not in df.columns:
        print("ICE column not found in DataFrame — skipping plot_ice_vs_alpha.")
        return None

    fig, ax = plt.subplots(figsize=(14, 10))

    # --- Primary Axis (ICE) ---
    line1 = ax.plot(df["alpha"], df["ICE"], marker="o", color="#ea580c", label="ICE")
    best_idx = df["ICE"].idxmax()
    best_alpha, best_val = df.loc[best_idx, "alpha"], df.loc[best_idx, "ICE"]
    scat = ax.scatter([best_alpha], [best_val], color="#dc2626", zorder=5,
                      label=f"best: α={best_alpha:.2f}, ICE={best_val:.4f}")

    ax.set_xlabel("alpha")
    ax.set_ylabel("ICE", color="#ea580c")
    ax.tick_params(axis="y", labelcolor="#ea580c")
    ax.set_title("alpha vs Incremental Coverage Efficiency & Mean Chunks")
    ax.grid(alpha=0.3)

    # --- Secondary Axis (Mean Chunks) ---
    ax2 = ax.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Number of Chunks", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    line2 = ax2.plot(df["alpha"], df["mean_chunks"], color=color2, marker="s", label="Mean Chunks")

    if k_target is not None:
        ax2.axhline(k_target, color="#dc2626", linestyle="--", label=f"K_FINAL = {k_target}")

    ax.set_xticks(df["alpha"])

    # --- Combined Legend ---
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2_all, labels_2_all = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2_all, labels_1 + labels_2_all, loc="upper left")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

if __name__ == "__main__":
    result_path = "result.csv"
    df = pd.read_csv(result_path)
    print(df)

    andcg = metric_name("alpha_nDCG")

    # plot_metric(df, andcg, save_path="results/alpha_vs_alpha_ndcg.png")
    # plot_all_metrics(df, save_path="results/alpha_all_metrics.png")
    plot_chunk_count(df, save_path="results/alpha_vs_chunk_count.png")
    plot_mmr_vs_alpha(df, save_path="results/alpha_vs_mmr.png")
    # plot_redundancy_metrics(df, save_path="results/redundancy_metrics.png")
    plot_snu_vs_alpha(df, save_path="results/alpha_vs_snu.png")
    plot_ice_vs_alpha(df, save_path="results/alpha_vs_ice.png")
    # if "MMR_raw" in df.columns:
    #     plot_mmr_comparison(df, save_path="mmr_comparison.png")
    print("All plots generated successfully.")