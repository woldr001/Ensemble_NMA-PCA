"""
anm_gnm_analysis.py
===================
Performs Anisotropic Network Model (ANM) and Gaussian Network Model (GNM)
analysis on an ensemble of protein conformers to identify concerted motions,
and cross-references results with per-residue flexibility from contact-map
analysis.

For each conformer the script:
  • Builds the GNM Kirchhoff matrix and extracts N_SLOW_MODES slow modes.
  • Builds the ANM Hessian matrix and extracts N_SLOW_MODES slow modes.

Ensemble averages (mean ± std) are computed across all conformers for:
  • Square fluctuations per mode (mobility profiles).
  • Inter-residue cross-correlation matrices.

Dynamic domains are identified from the ensemble-mean GNM cross-correlation
via hierarchical clustering.

Dependencies
------------
    prody, numpy, matplotlib, scipy

    Install via:  pip install prody numpy matplotlib scipy

Usage
-----
    python anm_gnm_analysis.py

    Run contact_map_analysis.py first to produce contact_map_results.npz.
    Edit the "USER SETTINGS" block below.

Output files
------------
    gnm_ensemble_fluctuations.png    – GNM slow-mode ensemble mean ± std
    anm_ensemble_fluctuations.png    – ANM slow-mode ensemble mean ± std
    gnm_ensemble_crosscorr.png       – ensemble-mean GNM cross-correlation
    anm_ensemble_crosscorr.png       – ensemble-mean ANM cross-correlation
    dynamic_domains_ensemble.png     – GNM domain assignment
    gnm_anm_vs_flexibility.png       – GNM/ANM modes vs contact-map flexibility
    concerted_summary.png            – 4-panel summary figure
    anm_gnm_results.npz              – all numerical arrays for downstream use
    conformer_<k>_anm_modes.nmd      – NMD files for VMD / NMWiz
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
PDB_DIR      = "/mnt/home/woldring/TopoFormer/Contact_Maps/cd1_seq1"
N_CONFORMERS = 10
PDB_PATTERN  = "cluster_repr_{i}.pdb"
PROTEIN_NAME = "IPNS (isopenicillin N-synthase)"
CHAIN        = "A"       # chain to analyse; set to None to auto-detect

GNM_CUTOFF   = 7.3      # Å  – Kirchhoff matrix contact cutoff
ANM_CUTOFF   = 15.0     # Å  – Hessian contact cutoff
N_SLOW_MODES = 3        # number of slow modes to analyse
N_DOMAINS    = 3        # dynamic domains to identify from GNM clustering
TOP_N        = 20       # top-flexible residues to highlight

OUT_NPZ      = "anm_gnm_results.npz"
# ---------------------------------------------------------------------------


# ── Colours ─────────────────────────────────────────────────────────────────

PALETTE = {
    "gnm"    : "#2E86AB",
    "anm"    : "#E84855",
    "flex"   : "#333333",
    "mark"   : "#FF006E",
    "domain" : ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"],
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mark_top_residues(ax, resids, top_resids, alpha: float = 0.15) -> None:
    """Shade columns corresponding to top-flexible residues."""
    top_set = set(top_resids)
    for i, r in enumerate(resids):
        if r in top_set:
            ax.axvspan(i - 0.5, i + 0.5, color=PALETTE["mark"],
                       alpha=alpha, linewidth=0, zorder=0)


def _style_ax(ax, xlabel: str = "Residue index",
              ylabel: str = "", title: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


# ── ProDy helpers ────────────────────────────────────────────────────────────

def _import_prody():
    try:
        import prody as pd
        pd.confProDy(verbosity="none")
        return pd
    except ImportError:
        sys.exit("[ERROR] ProDy is not installed.  Run:  pip install prody")


def load_ca(pdb_path: str, chain: str | None):
    """Load a PDB and return the Cα selection."""
    pd = _import_prody()
    struct = pd.parsePDB(pdb_path)
    if struct is None:
        sys.exit(f"[ERROR] Could not parse '{pdb_path}'.")
    sel_str = f"calpha and chain {chain}" if chain else "calpha"
    ca = struct.select(sel_str)
    if ca is None or len(ca) == 0:
        ca = struct.select("calpha")   # fallback: ignore chain filter
    if ca is None or len(ca) == 0:
        sys.exit(f"[ERROR] No Cα atoms found in '{pdb_path}'.")
    return ca


def run_gnm(ca, cutoff: float, n_modes: int):
    from prody import GNM
    gnm = GNM()
    gnm.buildKirchhoff(ca, cutoff=cutoff)
    gnm.calcModes(n_modes + 1)   # mode 0 is trivial (zero eigenvalue)
    return gnm


def run_anm(ca, cutoff: float, n_modes: int):
    from prody import ANM
    anm = ANM()
    anm.buildHessian(ca, cutoff=cutoff)
    anm.calcModes(n_modes * 3 + 6)   # skip 6 rigid-body modes
    return anm


def get_sq_fluctuations(model, mode_indices: list[int]) -> np.ndarray:
    from prody import calcSqFlucts
    return np.array([calcSqFlucts(model[i]) for i in mode_indices])


def get_cross_correlation(model, mode_indices: list[int]) -> np.ndarray:
    from prody import calcCrossCorr
    cc = sum(calcCrossCorr(model[i]) for i in mode_indices)
    return cc / len(mode_indices)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_ensemble_fluctuations(
    flucts_all: np.ndarray,
    resids: np.ndarray,
    top_resids: np.ndarray,
    model_type: str,
    out_path: str,
    color: str,
) -> None:
    """
    Multi-panel figure showing ensemble mean ± std for each slow mode.
    Individual conformer traces are shown as thin semi-transparent lines.

    Parameters
    ----------
    flucts_all : (N_conformers, N_modes, N_res)
    """
    n_conf, n_modes, n_res = flucts_all.shape
    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 2.8 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        flucts_k = flucts_all[:, k, :]       # (N_conf, N_res)
        mean_k   = flucts_k.mean(axis=0)
        std_k    = flucts_k.std(axis=0)
        x        = np.arange(n_res)

        _mark_top_residues(ax, resids, top_resids)

        for f in flucts_k:
            ax.plot(x, f, color=color, lw=0.7, alpha=0.25)

        ax.plot(x, mean_k, color=color, lw=2.0, label="Ensemble mean")
        ax.fill_between(x, mean_k - std_k, mean_k + std_k,
                        alpha=0.25, color=color, label="±1 std")

        ax.set_xticks(range(0, n_res, 20))
        ax.set_xticklabels(resids[::20], rotation=45, ha="right", fontsize=7)
        _style_ax(ax,
                  ylabel="Sq. flucts. (Å²)",
                  title=f"{model_type} – Slow mode {k+1}  |  "
                        f"Ensemble of {n_conf} conformers")
        ax.legend(fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Residue number", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Saved → {out_path}")


def plot_cross_correlation(
    cc_matrix: np.ndarray,
    resids: np.ndarray,
    top_resids: np.ndarray,
    model_type: str,
    out_path: str,
) -> None:
    """Heatmap of the ensemble-mean cross-correlation matrix."""
    n = len(resids)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    top_idx = [i for i, r in enumerate(resids) if r in set(top_resids)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cc_matrix, cmap="RdBu_r", origin="upper",
                   norm=norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cross-correlation", fontsize=10)

    for idx in top_idx:
        ax.axhline(idx, color=PALETTE["mark"], lw=0.6, ls="--", alpha=0.7)
        ax.axvline(idx, color=PALETTE["mark"], lw=0.6, ls="--", alpha=0.7)

    step  = 20
    ticks = np.arange(0, n, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels(resids[ticks], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(resids[ticks], fontsize=7)
    ax.set_xlabel("Residue", fontsize=10)
    ax.set_ylabel("Residue", fontsize=10)
    ax.set_title(f"{model_type} Ensemble-Mean Cross-Correlation — {PROTEIN_NAME}\n"
                 f"(pink dashes = top-{TOP_N} flexible residues)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Saved → {out_path}")


def plot_dynamic_domains(
    cc_matrix: np.ndarray,
    resids: np.ndarray,
    top_resids: np.ndarray,
    n_domains: int,
    out_path: str,
) -> None:
    """
    Identify dynamic domains via hierarchical clustering of the ensemble-mean
    GNM cross-correlation matrix, then plot domains alongside the CC heatmap.
    """
    dist_matrix    = 1.0 - np.abs(cc_matrix)
    np.fill_diagonal(dist_matrix, 0.0)
    dist_condensed = np.clip(squareform(dist_matrix, checks=False), 0, None)
    linkage_matrix = linkage(dist_condensed, method="ward")
    domain_labels  = fcluster(linkage_matrix, n_domains, criterion="maxclust")

    n       = len(resids)
    top_idx = [i for i, r in enumerate(resids) if r in set(top_resids)]

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1],
                            hspace=0.05, wspace=0.05)

    # CC heatmap
    ax_cc = fig.add_subplot(gs[0, 0])
    norm  = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im    = ax_cc.imshow(cc_matrix, cmap="RdBu_r", origin="upper",
                         norm=norm, aspect="auto")
    for idx in top_idx:
        ax_cc.axhline(idx, color=PALETTE["mark"], lw=0.6, ls="--", alpha=0.8)
        ax_cc.axvline(idx, color=PALETTE["mark"], lw=0.6, ls="--", alpha=0.8)
    step  = 20
    ticks = np.arange(0, n, step)
    ax_cc.set_xticks(ticks)
    ax_cc.set_xticklabels(resids[ticks], rotation=45, ha="right", fontsize=7)
    ax_cc.set_yticks(ticks)
    ax_cc.set_yticklabels(resids[ticks], fontsize=7)
    ax_cc.set_title(f"GNM Ensemble Cross-Correlation & Dynamic Domains — {PROTEIN_NAME}",
                    fontsize=11, fontweight="bold")
    ax_cc.set_ylabel("Residue", fontsize=10)

    # Domain colour strip
    ax_strip = fig.add_subplot(gs[0, 1])
    cmap_domains = matplotlib.colors.ListedColormap(
        [PALETTE["domain"][d % len(PALETTE["domain"])] for d in range(n_domains)]
    )
    ax_strip.imshow(domain_labels.reshape(-1, 1),
                    cmap=cmap_domains, aspect="auto", origin="upper",
                    vmin=1, vmax=n_domains)
    ax_strip.set_xticks([])
    ax_strip.set_yticks(ticks)
    ax_strip.set_yticklabels(resids[ticks], fontsize=7)
    ax_strip.set_title("Domain", fontsize=9)

    # Domain profile
    ax_prof = fig.add_subplot(gs[1, 0])
    for d in range(1, n_domains + 1):
        mask = domain_labels == d
        ax_prof.scatter(
            np.where(mask)[0], domain_labels[mask],
            color=PALETTE["domain"][(d - 1) % len(PALETTE["domain"])],
            s=20, label=f"Domain {d}", zorder=3,
        )
    _mark_top_residues(ax_prof, resids, top_resids)
    ax_prof.set_xlim(-1, n)
    ax_prof.set_ylim(0.3, n_domains + 0.7)
    ax_prof.set_yticks(range(1, n_domains + 1))
    ax_prof.set_xticks(ticks)
    ax_prof.set_xticklabels(resids[ticks], rotation=45, ha="right", fontsize=7)
    ax_prof.set_xlabel("Residue number", fontsize=10)
    ax_prof.set_ylabel("Domain", fontsize=10)
    ax_prof.legend(fontsize=8, ncol=n_domains, loc="upper right")
    ax_prof.grid(axis="x", linestyle="--", alpha=0.3)
    ax_prof.spines[["top", "right"]].set_visible(False)

    # Colorbar
    ax_cb = fig.add_subplot(gs[1, 1])
    ax_cb.axis("off")
    cbar = fig.colorbar(im, ax=ax_cb, fraction=0.8, pad=0.05)
    cbar.set_label("Cross-correlation", fontsize=9)

    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Saved → {out_path}")


def plot_overlap_with_flexibility(
    gnm_flucts_mean: np.ndarray,
    anm_flucts_mean: np.ndarray,
    per_residue_flexibility: np.ndarray,
    resids: np.ndarray,
    top_resids: np.ndarray,
    out_path: str,
) -> None:
    """
    Dual-axis comparison of ensemble-mean GNM and ANM slow-mode mobility
    against the per-residue contact-map flexibility signal.

    Pearson r quantifies how well each model captures the structural
    variability seen in the ensemble distance matrices.
    """
    n_modes = gnm_flucts_mean.shape[0]
    fig, axes = plt.subplots(n_modes, 2, figsize=(14, 3.2 * n_modes), sharex=True)
    if n_modes == 1:
        axes = axes.reshape(1, 2)

    x = np.arange(len(resids))

    for k in range(n_modes):
        for col, (flucts, model_name, color) in enumerate([
            (gnm_flucts_mean[k], "GNM", PALETTE["gnm"]),
            (anm_flucts_mean[k], "ANM", PALETTE["anm"]),
        ]):
            ax = axes[k, col]
            f_norm = _norm01(flucts)
            d_norm = _norm01(per_residue_flexibility)
            corr   = float(np.corrcoef(flucts, per_residue_flexibility)[0, 1])

            _mark_top_residues(ax, resids, top_resids)
            ax.plot(x, f_norm, color=color, lw=1.8,
                    label=f"{model_name} mode {k+1} (norm.)")
            ax.fill_between(x, f_norm, alpha=0.12, color=color)
            ax.plot(x, d_norm, color=PALETTE["flex"], lw=1.4, ls="--",
                    label="Contact flexibility (norm.)")
            ax.fill_between(x, d_norm, alpha=0.08, color=PALETTE["flex"])
            ax.set_yticks([0, 0.5, 1])
            ax.set_ylim(-0.05, 1.15)
            _style_ax(ax,
                      ylabel="Normalised value",
                      title=f"{model_name} mode {k+1} vs flexibility  "
                            f"[Pearson r = {corr:+.3f}]")
            ax.legend(fontsize=8, loc="upper right")
            ax.set_xticks(range(0, len(resids), 20))
            ax.set_xticklabels(resids[::20], rotation=45, ha="right", fontsize=7)

    for col in range(2):
        axes[-1, col].set_xlabel("Residue number", fontsize=10)

    fig.suptitle(
        f"GNM & ANM Ensemble-Mean Fluctuations vs Contact-Map Flexibility\n"
        f"{PROTEIN_NAME}  |  pink shading = top-{TOP_N} flexible residues",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_summary(
    gnm_flucts_mean: np.ndarray,
    anm_flucts_mean: np.ndarray,
    gnm_flucts_std: np.ndarray,
    anm_flucts_std: np.ndarray,
    per_residue_flexibility: np.ndarray,
    resids: np.ndarray,
    top_resids: np.ndarray,
    out_path: str,
) -> None:
    """Four-panel summary: GNM mode 1, ANM mode 1, flexibility, and overlay."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    x = np.arange(len(resids))

    panels = [
        (axes[0], gnm_flucts_mean[0], gnm_flucts_std[0], PALETTE["gnm"],
         "GNM – Ensemble-mean slow mode 1", "Sq. flucts. (Å²)"),
        (axes[1], anm_flucts_mean[0], anm_flucts_std[0], PALETTE["anm"],
         "ANM – Ensemble-mean slow mode 1", "Sq. flucts. (Å²)"),
    ]
    for ax, mean_f, std_f, color, title, ylabel in panels:
        _mark_top_residues(ax, resids, top_resids)
        ax.plot(x, mean_f, color=color, lw=1.8)
        ax.fill_between(x, mean_f - std_f, mean_f + std_f,
                        alpha=0.25, color=color, label="±1 std")
        delta_scaled = _norm01(per_residue_flexibility) * mean_f.max()
        ax.plot(x, delta_scaled, color=PALETTE["flex"], lw=1.0, ls=":",
                alpha=0.7, label="Contact flexibility (scaled)")
        _style_ax(ax, ylabel=ylabel, title=title)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xticks(range(0, len(resids), 20))
        ax.set_xticklabels(resids[::20], rotation=45, ha="right", fontsize=7)

    # Panel 3: raw flexibility profile
    ax3 = axes[2]
    _mark_top_residues(ax3, resids, top_resids)
    ax3.bar(x, per_residue_flexibility, color="#888888", width=1.0, alpha=0.7)
    _style_ax(ax3, ylabel="Mean dist std (Å)",
              title="Contact-Map Per-Residue Flexibility")
    ax3.set_xticks(range(0, len(resids), 20))
    ax3.set_xticklabels(resids[::20], rotation=45, ha="right", fontsize=7)

    # Panel 4: normalised overlay of GNM, ANM, flexibility
    ax4 = axes[3]
    _mark_top_residues(ax4, resids, top_resids)
    ax4.plot(x, _norm01(gnm_flucts_mean[0]), color=PALETTE["gnm"],
             lw=1.6, label="GNM mode 1")
    ax4.plot(x, _norm01(anm_flucts_mean[0]), color=PALETTE["anm"],
             lw=1.6, label="ANM mode 1")
    ax4.plot(x, _norm01(per_residue_flexibility), color=PALETTE["flex"],
             lw=1.4, ls="--", label="Contact flexibility")
    _style_ax(ax4, ylabel="Normalised value",
              title="GNM / ANM / Flexibility Overlay (normalised)")
    ax4.set_yticks([0, 0.5, 1])
    ax4.set_ylim(-0.05, 1.15)
    ax4.legend(fontsize=9, loc="upper right")
    ax4.set_xticks(range(0, len(resids), 20))
    ax4.set_xticklabels(resids[::20], rotation=45, ha="right", fontsize=7)

    axes[-1].set_xlabel("Residue number", fontsize=10)
    fig.suptitle(
        f"Concerted Motions: GNM & ANM vs Contact-Map Flexibility\n"
        f"{PROTEIN_NAME}  ({N_CONFORMERS} conformers)  |  "
        f"pink shading = top-{TOP_N} flexible residues",
        fontsize=12, fontweight="bold", y=1.005,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print(f"  {PROTEIN_NAME}")
    print(f"  Ensemble ANM / GNM Concerted Motion Analysis  ({N_CONFORMERS} conformers)")
    print("=" * 65)

    pdb_files = [
        os.path.join(PDB_DIR, PDB_PATTERN.replace("{i}", str(i)))
        for i in range(N_CONFORMERS)
    ]
    for p in pdb_files:
        if not os.path.isfile(p):
            sys.exit(f"[ERROR] File not found: {p}\n"
                     "        Check PDB_DIR and PDB_PATTERN in USER SETTINGS.")

    # ── Reference structure to determine n_res ───────────────────────────────
    print(f"\n[1/6] Loading reference structure ({os.path.basename(pdb_files[0])}) ...")
    ca_ref   = load_ca(pdb_files[0], CHAIN)
    n_res    = len(ca_ref)
    resids   = ca_ref.getResnums()
    print(f"      {n_res} Cα atoms detected (chain filter = '{CHAIN}')")

    gnm_mode_idx = list(range(1, N_SLOW_MODES + 1))
    anm_mode_idx = list(range(6, 6 + N_SLOW_MODES))

    gnm_flucts_all = np.zeros((N_CONFORMERS, N_SLOW_MODES, n_res))
    anm_flucts_all = np.zeros((N_CONFORMERS, N_SLOW_MODES, n_res))
    gnm_cc_all     = np.zeros((N_CONFORMERS, n_res, n_res))
    anm_cc_all     = np.zeros((N_CONFORMERS, n_res, n_res))

    # ── GNM + ANM on every conformer ─────────────────────────────────────────
    print(f"\n[2/6] Running GNM (cutoff={GNM_CUTOFF} Å) and "
          f"ANM (cutoff={ANM_CUTOFF} Å) on {N_CONFORMERS} conformers ...")

    for k, pdb_path in enumerate(pdb_files):
        ca_k = load_ca(pdb_path, CHAIN)
        if len(ca_k) != n_res:
            print(f"  [WARN] Conformer {k} has {len(ca_k)} Cα atoms "
                  f"(expected {n_res}). Skipping.")
            continue

        gnm_k = run_gnm(ca_k, GNM_CUTOFF, N_SLOW_MODES)
        anm_k = run_anm(ca_k, ANM_CUTOFF, N_SLOW_MODES)

        gnm_flucts_all[k] = get_sq_fluctuations(gnm_k, gnm_mode_idx)
        anm_flucts_all[k] = get_sq_fluctuations(anm_k, anm_mode_idx)
        gnm_cc_all[k]     = get_cross_correlation(gnm_k, gnm_mode_idx)
        anm_cc_all[k]     = get_cross_correlation(anm_k, anm_mode_idx)

        # NMD file for VMD / NMWiz
        try:
            from prody import writeNMD
            nmd_path = f"conformer_{k}_anm_modes.nmd"
            writeNMD(nmd_path, anm_k[:N_SLOW_MODES], ca_k)
        except Exception:
            pass

        print(f"      [{k}] {os.path.basename(pdb_path)}: done")

    # ── Ensemble averages ────────────────────────────────────────────────────
    print("\n[3/6] Computing ensemble averages ...")
    gnm_flucts_mean = gnm_flucts_all.mean(axis=0)
    gnm_flucts_std  = gnm_flucts_all.std(axis=0)
    gnm_cc_mean     = gnm_cc_all.mean(axis=0)

    anm_flucts_mean = anm_flucts_all.mean(axis=0)
    anm_flucts_std  = anm_flucts_all.std(axis=0)
    anm_cc_mean     = anm_cc_all.mean(axis=0)

    # ── Load contact-map flexibility signal ──────────────────────────────────
    print("\n[4/6] Loading contact-map flexibility signal ...")
    try:
        npz    = np.load("contact_map_results.npz", allow_pickle=True)
        per_residue_flexibility = npz["per_residue_flexibility"]
        top_flexible_resids     = npz["top_flexible_resids"]
        labels_npz              = npz["labels"].tolist()
        npz_resids = np.array([
            int("".join(filter(str.isdigit, lbl.split(":")[1])))
            for lbl in labels_npz
        ])
        # Align onto the ProDy residue numbering
        flex_aligned = np.zeros(n_res)
        for i, r in enumerate(resids):
            match = np.where(npz_resids == r)[0]
            if len(match):
                flex_aligned[i] = per_residue_flexibility[match[0]]
        per_residue_flexibility = flex_aligned
        print(f"      Loaded flexibility signal for {n_res} residues.")
    except FileNotFoundError:
        print("      contact_map_results.npz not found — using uniform placeholder.")
        print("      Run contact_map_analysis.py first for full overlap plots.")
        per_residue_flexibility = np.ones(n_res)
        top_flexible_resids = resids[:TOP_N]

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n[5/6] Generating plots ...")

    plot_ensemble_fluctuations(
        gnm_flucts_all, resids, top_flexible_resids, "GNM",
        "gnm_ensemble_fluctuations.png", PALETTE["gnm"],
    )
    plot_ensemble_fluctuations(
        anm_flucts_all, resids, top_flexible_resids, "ANM",
        "anm_ensemble_fluctuations.png", PALETTE["anm"],
    )
    plot_cross_correlation(
        gnm_cc_mean, resids, top_flexible_resids, "GNM",
        "gnm_ensemble_crosscorr.png",
    )
    plot_cross_correlation(
        anm_cc_mean, resids, top_flexible_resids, "ANM",
        "anm_ensemble_crosscorr.png",
    )
    plot_dynamic_domains(
        gnm_cc_mean, resids, top_flexible_resids, N_DOMAINS,
        "dynamic_domains_ensemble.png",
    )
    plot_overlap_with_flexibility(
        gnm_flucts_mean, anm_flucts_mean,
        per_residue_flexibility, resids, top_flexible_resids,
        "gnm_anm_vs_flexibility.png",
    )
    plot_summary(
        gnm_flucts_mean, anm_flucts_mean,
        gnm_flucts_std,  anm_flucts_std,
        per_residue_flexibility, resids, top_flexible_resids,
        "concerted_summary.png",
    )

    # ── Save numerical results ────────────────────────────────────────────────
    print("\n[6/6] Saving numerical results ...")
    np.savez_compressed(
        OUT_NPZ,
        gnm_flucts_all=gnm_flucts_all,
        gnm_flucts_mean=gnm_flucts_mean,
        gnm_flucts_std=gnm_flucts_std,
        gnm_cc_mean=gnm_cc_mean,
        anm_flucts_all=anm_flucts_all,
        anm_flucts_mean=anm_flucts_mean,
        anm_flucts_std=anm_flucts_std,
        anm_cc_mean=anm_cc_mean,
        per_residue_flexibility=per_residue_flexibility,
        resids=resids,
        top_flexible_resids=top_flexible_resids,
    )
    print(f"  Saved → {OUT_NPZ}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  PEARSON CORRELATIONS  (ensemble-mean sq. flucts. vs contact flexibility)\n")
    print(f"  {'Model':<6}  {'Mode':>6}  {'Pearson r':>10}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*10}")
    for model_name, flucts_mean in [("GNM", gnm_flucts_mean), ("ANM", anm_flucts_mean)]:
        for k, flucts in enumerate(flucts_mean):
            r = float(np.corrcoef(flucts, per_residue_flexibility)[0, 1])
            print(f"  {model_name:<6}  {k+1:>6}  {r:>+10.4f}")

    print(f"\n  Interpretation:")
    print(f"  • High |r| → the slow mode captures the flexibility")
    print(f"    pattern seen in the ensemble contact-map variability.")
    print(f"  • gnm_ensemble_crosscorr.png: residue-pair concerted motions.")
    print(f"  • dynamic_domains_ensemble.png: domain assignments.")
    print(f"  • conformer_<k>_anm_modes.nmd: load into VMD (NMWiz plugin)")
    print(f"    to animate slow modes as 3-D arrows on the structure.")
    print("=" * 65)


if __name__ == "__main__":
    main()
