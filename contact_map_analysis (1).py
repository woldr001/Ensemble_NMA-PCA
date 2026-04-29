"""
contact_map_analysis.py
=======================
Computes Cα-Cα distance-based contact maps for two PDB structures and
identifies residue positions that show the greatest conformational change
between them.

Dependencies
------------
    numpy, scipy, matplotlib, biopython

    Install via:
        pip install numpy scipy matplotlib biopython

Usage
-----
    python contact_map_analysis.py

    Edit the constants in the "USER SETTINGS" block below to point to your
    PDB files and adjust thresholds as needed.

Output files
------------
    contact_map_<label>.png          – Cα distance matrix heatmap
    delta_contact_map.png            – |ΔD| heatmap
    top_changing_residues.png        – bar chart of residues ranked by change
    contact_map_results.npz          – all numerical arrays for downstream use
    top_changing_residues.csv        – ranked residue list (CSV)

References
----------
    Müller, C. W., Schlauderer, G. J., Reinstein, J., & Schulz, G. E. (1996).
    Adenylate kinase motions during catalysis: an energetic counterweight
    balancing substrate binding. Structure, 4(2), 147-156.
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
PDB_1 = "1ake_bare.pdb"          # closed conformation (with inhibitor removed)
PDB_2 = "4ake_bare.pdb"          # open conformation
LABEL_1 = "1AKE (closed)"
LABEL_2 = "4AKE (open)"

CONTACT_THRESHOLD_ANG = 8.0      # Å — residues closer than this are "in contact"
TOP_N_RESIDUES = 20              # how many top-changing positions to highlight
CHAIN_ID = None                  # None = use first chain found; or e.g. "A"

OUTPUT_MAP_1   = "contact_map_1ake.png"
OUTPUT_MAP_2   = "contact_map_4ake.png"
OUTPUT_DELTA   = "delta_contact_map.png"
OUTPUT_BAR     = "top_changing_residues.png"
OUTPUT_NPZ     = "contact_map_results.npz"
OUTPUT_CSV     = "top_changing_residues.csv"
# ---------------------------------------------------------------------------


# ── 1. PDB parsing ──────────────────────────────────────────────────────────

def extract_ca_coords(pdb_path: str, chain_id: str | None = None) -> tuple[np.ndarray, list[str]]:
    """
    Parse a PDB file and return Cα coordinates and residue labels.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    chain_id : str or None
        If None the first chain encountered is used; otherwise the specified
        chain is extracted.

    Returns
    -------
    coords : np.ndarray, shape (N, 3)
        Cα XYZ coordinates in Å.
    labels : list of str
        Residue labels formatted as "<chain>:<resname><resseq>" (e.g. "A:ALA14").
    """
    coords: list[list[float]] = []
    labels: list[str] = []
    selected_chain: str | None = chain_id
    seen_residues: set[tuple[str, int, str]] = set()

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            chain = line[21]
            if selected_chain is None:
                selected_chain = chain          # latch onto first chain
            if chain != selected_chain:
                continue

            resname = line[17:20].strip()
            resseq  = int(line[22:26].strip())
            icode   = line[26].strip()          # insertion code
            res_key = (chain, resseq, icode)

            if res_key in seen_residues:         # skip alternate conformations
                continue
            seen_residues.add(res_key)

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
            label_str = f"{chain}:{resname}{resseq}"
            if icode:
                label_str += icode
            labels.append(label_str)

    if not coords:
        sys.exit(
            f"[ERROR] No Cα atoms found in '{pdb_path}' "
            f"(chain filter = '{selected_chain}'). Check CHAIN_ID setting."
        )

    return np.array(coords, dtype=np.float64), labels


# ── 2. Distance matrix ───────────────────────────────────────────────────────

def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric pairwise Euclidean distance matrix for Cα atoms.

    Uses scipy.spatial.distance.cdist for fully vectorised computation
    (no Python-level loops).

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)

    Returns
    -------
    dist_matrix : np.ndarray, shape (N, N)
        Symmetric matrix of pairwise Cα–Cα distances in Å.
    """
    return cdist(coords, coords, metric="euclidean")


# ── 3. Plotting helpers ──────────────────────────────────────────────────────

def _residue_tick_params(n: int, labels: list[str], ax: plt.Axes, step: int = 20):
    """Set axis ticks every `step` residues with sequence-number labels."""
    ticks = np.arange(0, n, step)
    # Extract just the numeric part for brevity
    tick_labels = [labels[i].split(":")[1] for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=7)


def plot_distance_map(
    dist_matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: str,
    contact_threshold: float = CONTACT_THRESHOLD_ANG,
) -> None:
    """
    Save a heatmap of the Cα–Cα distance matrix with a contact-threshold contour.

    Parameters
    ----------
    dist_matrix : np.ndarray, shape (N, N)
    labels : list of str
    title : str
    output_path : str
    contact_threshold : float
        Residues within this distance (Å) are considered in contact and are
        highlighted by a contour line.
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(dist_matrix, cmap="viridis_r", origin="upper",
                   vmin=0, vmax=np.percentile(dist_matrix, 95))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cα–Cα distance (Å)", fontsize=10)

    # Overlay contact boundary
    ax.contour(dist_matrix, levels=[contact_threshold],
               colors="white", linewidths=0.8, linestyles="--")

    _residue_tick_params(n, labels, ax)
    ax.set_xlabel("Residue", fontsize=11)
    ax.set_ylabel("Residue", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"  Saved → {output_path}")


def plot_delta_map(
    delta: np.ndarray,
    labels: list[str],
    output_path: str,
) -> None:
    """
    Save a heatmap of the absolute distance change |ΔD| = |D1 – D2|.

    Parameters
    ----------
    delta : np.ndarray, shape (N, N)
        Absolute difference matrix.
    labels : list of str
    output_path : str
    """
    n = len(labels)
    vmax = np.percentile(delta, 99)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(delta, cmap="hot_r", origin="upper", vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|ΔCα–Cα distance| (Å)", fontsize=10)

    _residue_tick_params(n, labels, ax)
    ax.set_xlabel("Residue", fontsize=11)
    ax.set_ylabel("Residue", fontsize=11)
    ax.set_title("|ΔD| Contact Map  (1AKE vs 4AKE)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"  Saved → {output_path}")


def plot_top_residues(
    scores: np.ndarray,
    labels: list[str],
    top_n: int,
    output_path: str,
) -> None:
    """
    Bar chart of the top_n residues ranked by their mean |ΔD| across all pairs.

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Per-residue mean absolute distance change.
    labels : list of str
    top_n : int
    output_path : str
    """
    ranked_idx = np.argsort(scores)[::-1][:top_n]
    ranked_scores = scores[ranked_idx]
    ranked_labels = [labels[i] for i in ranked_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(top_n), ranked_scores, color="steelblue", edgecolor="white")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(ranked_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean |ΔCα–Cα distance| (Å)", fontsize=11)
    ax.set_title(
        f"Top {top_n} residues with greatest Cα distance change\n"
        f"between {LABEL_1} and {LABEL_2}",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bar values
    for bar, val in zip(bars, ranked_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=7, color="black",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"  Saved → {output_path}")


# ── 4. Residue alignment ─────────────────────────────────────────────────────

def align_residues(
    labels_1: list[str], labels_2: list[str]
) -> tuple[list[int], list[int]]:
    """
    Find the common residue set present in both structures (by label) and
    return the corresponding index arrays for each structure.

    This handles cases where the two PDB files have slightly different residue
    ranges (e.g. disordered termini missing in one structure).

    Parameters
    ----------
    labels_1, labels_2 : list of str

    Returns
    -------
    idx_1, idx_2 : list of int
        Parallel index lists into coords_1 and coords_2 respectively.
    """
    set_2 = {lbl: i for i, lbl in enumerate(labels_2)}
    idx_1, idx_2 = [], []
    for i, lbl in enumerate(labels_1):
        if lbl in set_2:
            idx_1.append(i)
            idx_2.append(set_2[lbl])

    n_common = len(idx_1)
    if n_common == 0:
        sys.exit(
            "[ERROR] No common residues found between the two structures. "
            "Check that both files use the same chain ID and residue numbering."
        )
    print(
        f"  Aligned {n_common} common residues "
        f"({len(labels_1) - n_common} unique to structure 1, "
        f"{len(labels_2) - n_common} unique to structure 2)"
    )
    return idx_1, idx_2


# ── 5. Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Adenylate Kinase — Cα Contact Map Analysis")
    print("=" * 60)

    # --- Parse PDB files ---
    print(f"\n[1/5] Parsing '{PDB_1}' ...")
    coords_1, labels_1 = extract_ca_coords(PDB_1, chain_id=CHAIN_ID)
    print(f"      Found {len(labels_1)} Cα atoms  (chain {labels_1[0].split(':')[0]})")

    print(f"\n[2/5] Parsing '{PDB_2}' ...")
    coords_2, labels_2 = extract_ca_coords(PDB_2, chain_id=CHAIN_ID)
    print(f"      Found {len(labels_2)} Cα atoms  (chain {labels_2[0].split(':')[0]})")

    # --- Align to common residue set ---
    print("\n[3/5] Aligning residue sets ...")
    idx_1, idx_2 = align_residues(labels_1, labels_2)
    coords_1_aln  = coords_1[idx_1]
    coords_2_aln  = coords_2[idx_2]
    labels_common = [labels_1[i] for i in idx_1]
    n = len(labels_common)

    # --- Compute distance matrices ---
    print(f"\n[4/5] Computing {n}×{n} distance matrices ...")
    D1 = compute_distance_matrix(coords_1_aln)
    D2 = compute_distance_matrix(coords_2_aln)
    delta = np.abs(D1 - D2)

    # Per-residue score: mean |ΔD| across all pairwise interactions
    # (diagonal is always 0, so it does not inflate the mean)
    per_residue_score = delta.mean(axis=1)

    # Ranked results
    ranked_idx    = np.argsort(per_residue_score)[::-1]
    ranked_labels = [labels_common[i] for i in ranked_idx]
    ranked_scores = per_residue_score[ranked_idx]

    # --- Save numerical outputs ---
    np.savez_compressed(
        OUTPUT_NPZ,
        D1=D1, D2=D2, delta=delta,
        per_residue_score=per_residue_score,
        labels=np.array(labels_common),
    )
    print(f"  Saved → {OUTPUT_NPZ}")

    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "residue_label", "mean_abs_delta_dist_ang"])
        for rank, (lbl, score) in enumerate(zip(ranked_labels, ranked_scores), start=1):
            writer.writerow([rank, lbl, f"{score:.4f}"])
    print(f"  Saved → {OUTPUT_CSV}")

    # --- Plots ---
    print("\n[5/5] Generating plots ...")
    plot_distance_map(D1, labels_common, f"Cα Distance Map — {LABEL_1}",
                      OUTPUT_MAP_1, CONTACT_THRESHOLD_ANG)
    plot_distance_map(D2, labels_common, f"Cα Distance Map — {LABEL_2}",
                      OUTPUT_MAP_2, CONTACT_THRESHOLD_ANG)
    plot_delta_map(delta, labels_common, OUTPUT_DELTA)
    plot_top_residues(per_residue_score, labels_common, TOP_N_RESIDUES, OUTPUT_BAR)

    # --- Summary ---
    print(f"\n{'─'*60}")
    print(f"  Top {min(TOP_N_RESIDUES, 10)} residues by mean |ΔCα–Cα distance|:\n")
    print(f"  {'Rank':>4}  {'Residue':<18}  {'Mean |ΔD| (Å)':>14}")
    print(f"  {'─'*4}  {'─'*18}  {'─'*14}")
    for rank in range(min(TOP_N_RESIDUES, 10)):
        print(f"  {rank+1:>4}  {ranked_labels[rank]:<18}  {ranked_scores[rank]:>14.3f}")
    print(f"\n  Full ranking saved to '{OUTPUT_CSV}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
