"""
anm_gnm_analysis.py
===================
Performs Anisotropic Network Model (ANM) and Gaussian Network Model (GNM)
analysis on two adenylate kinase conformations to identify concerted motions,
and cross-references results with user-supplied contact-map change residues.

Dependencies
------------
    prody, numpy, matplotlib, scipy

    Install via:
        pip install prody numpy matplotlib scipy

Usage
-----
    python anm_gnm_analysis.py

    Edit the "USER SETTINGS" block to point to your PDB files and update
    TOP_CHANGING_RESIDS with your ranked residue list.

Output files
------------
    gnm_slowmode_<label>.png         – GNM slow-mode mobility profile
    gnm_cross_correlation.png        – GNM inter-residue cross-correlation heatmap
    gnm_contact_overlap.png          – Overlap of GNM slow modes with |ΔD| signal
    anm_slowmode_<label>.png         – ANM slow-mode square fluctuations
    anm_cross_correlation.png        – ANM cross-correlation heatmap
    anm_contact_overlap.png          – Overlap of ANM slow modes with |ΔD| signal
    dynamic_domains.png              – GNM domain assignment from slow modes
    concerted_summary.png            – Combined summary figure
    anm_gnm_results.npz              – All numerical arrays for downstream use
    <label>_anm_modes.nmd           – NMD file for ProDy/VMD visualisation

References
----------
    Bahar, I., Atilgan, A.R., Erman, B. (1997). Direct evaluation of thermal
        fluctuations in proteins using a single-parameter harmonic potential.
        Folding & Design, 2(3), 173-181.  [GNM]

    Atilgan, A.R. et al. (2001). Anisotropy of fluctuation dynamics of proteins
        with an elastic network model. Biophysical Journal, 80(1), 505-515. [ANM]

    Bakan, A., Meireles, L.M., Bahar, I. (2011). ProDy: Protein Dynamics
        Inferred from Theory and Experiments. Bioinformatics, 27(11), 1575-1577.
"""

import sys
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")  # suppress ProDy INFO messages to stdout

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
PDB_1 = "1ake_bare.pdb"          # closed conformation
PDB_2 = "4ake_bare.pdb"          # open conformation
LABEL_1 = "1AKE"
LABEL_2 = "4AKE"
CHAIN   = "A"                    # chain to analyse

# Top-changing residues from contact map difference analysis (ranked 1→20)
TOP_CHANGING_RESIDS = [
    149, 148, 150, 129, 128, 151, 147, 146, 127, 130,
    131, 126, 145, 152, 142, 141, 153, 140, 125, 132,
]

# ANM / GNM parameters
GNM_CUTOFF  = 7.3   # Å  — default contact cutoff for GNM
ANM_CUTOFF  = 15.0  # Å  — default contact cutoff for ANM
N_SLOW_MODES = 3    # number of slow (lowest-frequency) modes to visualise
N_DOMAINS    = 3    # number of dynamic domains to identify from slow modes

# Output paths
OUT_NPZ     = "anm_gnm_results.npz"
# ---------------------------------------------------------------------------


# ── Colour helpers ───────────────────────────────────────────────────────────

PALETTE = {
    "1AKE"   : "#2E86AB",
    "4AKE"   : "#E84855",
    "overlap": "#F4A261",
    "domain" : ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"],
    "mark"   : "#FF006E",
}


def _mark_top_residues(ax, resids_all, top_resids, ymin, ymax, alpha=0.15):
    """Shade columns corresponding to top-changing residues."""
    top_set = set(top_resids)
    for i, r in enumerate(resids_all):
        if r in top_set:
            ax.axvspan(i - 0.5, i + 0.5, color=PALETTE["mark"],
                       alpha=alpha, linewidth=0, zorder=0)


def _style_ax(ax, xlabel="Residue index", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


# ── ProDy helpers ────────────────────────────────────────────────────────────

def load_ca(pdb_path: str, chain: str):
    """Load a PDB file and return the Cα selection via ProDy."""
    try:
        import prody as pd
        pd.confProDy(verbosity="none")
    except ImportError:
        sys.exit("[ERROR] ProDy is not installed.  Run:  pip install prody")

    struct = pd.parsePDB(pdb_path, chain=chain)
    if struct is None:
        sys.exit(f"[ERROR] Could not parse '{pdb_path}'. Check the path and chain ID.")
    ca = struct.select("calpha")
    if ca is None or len(ca) == 0:
        sys.exit(f"[ERROR] No Cα atoms found in chain {chain} of '{pdb_path}'.")
    return ca


def run_gnm(ca, cutoff: float, n_modes: int):
    """Fit GNM and return model + slow modes."""
    from prody import GNM
    gnm = GNM()
    gnm.buildKirchhoff(ca, cutoff=cutoff)
    gnm.calcModes(n_modes + 1)        # +1 because mode 0 is trivial (zero eigenvalue)
    return gnm


def run_anm(ca, cutoff: float, n_modes: int):
    """Fit ANM and return model + slow modes."""
    from prody import ANM
    anm = ANM()
    anm.buildHessian(ca, cutoff=cutoff)
    anm.calcModes(n_modes * 3 + 6)    # extra modes to skip the 6 rigid-body modes
    return anm


def get_sq_fluctuations(model, mode_indices):
    """Compute square fluctuations from selected mode indices (1-based in ProDy)."""
    from prody import calcSqFlucts
    modes = [model[i] for i in mode_indices]
    return np.array([calcSqFlucts(m) for m in modes])


def get_cross_correlation(model, mode_indices):
    """Compute cross-correlation matrix from selected modes."""
    from prody import calcCrossCorr
    modes = [model[i] for i in mode_indices]
    # stack contributions
    cc = None
    for m in modes:
        c = calcCrossCorr(m)
        cc = c if cc is None else cc + c
    return cc / len(modes)


# ── Plotting functions ───────────────────────────────────────────────────────

def plot_slow_mode_profile(sq_flucts_list, resids, top_resids, label, model_type, out_path):
    """
    Line plot of square fluctuations for the N slowest modes.
    Pink shading marks top-changing residues from contact map analysis.
    """
    n_modes = len(sq_flucts_list)
    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 2.8 * n_modes),
                              sharex=True)
    if n_modes == 1:
        axes = [axes]

    for k, (ax, flucts) in enumerate(zip(axes, sq_flucts_list)):
        mode_num = k + 1
        _mark_top_residues(ax, resids, top_resids,
                           flucts.min(), flucts.max())
        ax.plot(range(len(resids)), flucts,
                color=PALETTE[label], lw=1.5, label=f"Mode {mode_num}")
        ax.fill_between(range(len(resids)), flucts,
                        alpha=0.15, color=PALETTE[label])
        ax.set_xticks(range(0, len(resids), 20))
        ax.set_xticklabels(resids[::20], rotation=45, ha="right", fontsize=7)
        _style_ax(ax,
                  ylabel="Sq. flucts. (Å²)",
                  title=f"{model_type} – {label}  |  Slow mode {mode_num}")
        ax.legend(fontsize=9, loc="upper right")

    fig.text(0.5, 0.01, "Residue number", ha="center", fontsize=10)
    ax.set_xlabel("Residue number", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Saved → {out_path}")


def plot_cross_correlation(cc_matrix, resids, top_resids, label, model_type, out_path):
    """
    Heatmap of the cross-correlation matrix.
    Dashed lines and tick annotations mark top-changing residues.
    """
    n = len(resids)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    top_set  = set(top_resids)
    top_idx  = [i for i, r in enumerate(resids) if r in top_set]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cc_matrix, cmap="RdBu_r", origin="upper",
                   norm=norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cross-correlation", fontsize=10)

    # Mark top-changing residues with dashed lines
    for idx in top_idx:
        ax.axhline(idx, color=PALETTE["mark"], lw=0.6, ls="--", alpha=0.7)
        ax.axvline(idx, color=PALETTE["mark"], lw=0.6, ls="--", alpha=0.7)

    step = 20
    ticks = np.arange(0, n, step)
    tick_labels = resids[ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=45,
                                              ha="right", fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Residue", fontsize=10)
    ax.set_ylabel("Residue", fontsize=10)
    ax.set_title(f"{model_type} Cross-Correlation  –  {label}\n"
                 f"(pink dashes = top-|ΔD| residues)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Saved → {out_path}")


def plot_overlap_with_contact_change(sq_flucts_list, per_residue_delta,
                                     resids, top_resids, label, model_type, out_path):
    """
    Dual-axis plot comparing slow-mode mobility (ANM/GNM) with the per-residue
    |ΔD| signal from contact map analysis. High overlap means the motion
    captured by the slow mode correlates with the structural change observed
    between the two conformations.

    Also computes and prints the Pearson correlation between each slow mode
    and the |ΔD| signal as a quantitative overlap metric.
    """
    n_modes = len(sq_flucts_list)
    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 3.2 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    top_set = set(top_resids)

    for k, (ax, flucts) in enumerate(zip(axes, sq_flucts_list)):
        mode_num = k + 1

        # Normalise both signals to [0, 1] for visual overlay
        def norm01(x):
            xmin, xmax = x.min(), x.max()
            return (x - xmin) / (xmax - xmin + 1e-12)

        f_norm  = norm01(flucts)
        d_norm  = norm01(per_residue_delta)

        # Pearson correlation as overlap metric
        corr = float(np.corrcoef(flucts, per_residue_delta)[0, 1])

        _mark_top_residues(ax, resids, top_resids, 0, 1)

        x = np.arange(len(resids))
        ax.plot(x, f_norm, color=PALETTE[label], lw=1.8,
                label=f"{model_type} mode {mode_num} (norm.)")
        ax.fill_between(x, f_norm, alpha=0.12, color=PALETTE[label])

        ax.plot(x, d_norm, color="#333333", lw=1.4, ls="--",
                label="|ΔD| from contact maps (norm.)")
        ax.fill_between(x, d_norm, alpha=0.08, color="#333333")

        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim(-0.05, 1.15)
        _style_ax(ax,
                  ylabel="Normalised value",
                  title=(f"{model_type} – {label}  |  Mode {mode_num} vs |ΔD|  "
                         f"  [Pearson r = {corr:+.3f}]"))
        ax.legend(fontsize=9, loc="upper right")

        ax.set_xticks(range(0, len(resids), 20))
        ax.set_xticklabels(resids[::20], rotation=45, ha="right", fontsize=7)

    ax.set_xlabel("Residue number", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Saved → {out_path}")


def plot_dynamic_domains(cc_matrix, resids, top_resids, n_domains, label, out_path):
    """
    Identify dynamic domains via hierarchical clustering of the GNM
    cross-correlation matrix, then plot the domain assignment alongside the
    cross-correlation heatmap and a mobility profile coloured by domain.

    Residues that cluster together move concertedly (high positive correlation).
    """
    # Convert correlation to distance (1 - |cc| so anticorrelated residues
    # are far apart in domain space, not grouped together)
    dist_matrix = 1.0 - np.abs(cc_matrix)
    np.fill_diagonal(dist_matrix, 0.0)
    dist_condensed = squareform(dist_matrix, checks=False)
    dist_condensed = np.clip(dist_condensed, 0, None)  # numerical safety

    linkage_matrix = linkage(dist_condensed, method="ward")
    domain_labels  = fcluster(linkage_matrix, n_domains, criterion="maxclust")

    n = len(resids)
    top_set = set(top_resids)
    top_idx = [i for i, r in enumerate(resids) if r in top_set]

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1],
                             hspace=0.05, wspace=0.05)

    # ── Cross-correlation heatmap (top-left) ────────────────────────────────
    ax_cc = fig.add_subplot(gs[0, 0])
    norm  = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im    = ax_cc.imshow(cc_matrix, cmap="RdBu_r", origin="upper",
                         norm=norm, aspect="auto")
    for idx in top_idx:
        ax_cc.axhline(idx, color=PALETTE["mark"], lw=0.6, ls="--", alpha=0.8)
        ax_cc.axvline(idx, color=PALETTE["mark"], lw=0.6, ls="--", alpha=0.8)

    step = 20
    ticks = np.arange(0, n, step)
    ax_cc.set_xticks(ticks)
    ax_cc.set_xticklabels(resids[ticks], rotation=45, ha="right", fontsize=7)
    ax_cc.set_yticks(ticks)
    ax_cc.set_yticklabels(resids[ticks], fontsize=7)
    ax_cc.set_title(f"GNM Cross-Correlation & Dynamic Domains  –  {label}",
                    fontsize=11, fontweight="bold")
    ax_cc.set_ylabel("Residue", fontsize=10)

    # ── Domain colour strip (right of heatmap) ──────────────────────────────
    ax_strip = fig.add_subplot(gs[0, 1])
    domain_colours = np.array(
        [matplotlib.colors.to_rgba(PALETTE["domain"][d - 1 % len(PALETTE["domain"])])
         for d in domain_labels]
    )
    ax_strip.imshow(domain_labels.reshape(-1, 1),
                    cmap=matplotlib.colors.ListedColormap(
                        [PALETTE["domain"][d % len(PALETTE["domain"])]
                         for d in range(n_domains)]
                    ),
                    aspect="auto", origin="upper",
                    vmin=1, vmax=n_domains)
    ax_strip.set_xticks([])
    ax_strip.set_yticks(ticks)
    ax_strip.set_yticklabels(resids[ticks], fontsize=7)
    ax_strip.set_title("Domain", fontsize=9)

    # ── Domain assignment profile (bottom-left) ─────────────────────────────
    ax_prof = fig.add_subplot(gs[1, 0])
    for d in range(1, n_domains + 1):
        mask = domain_labels == d
        ax_prof.scatter(
            np.where(mask)[0], domain_labels[mask],
            color=PALETTE["domain"][(d - 1) % len(PALETTE["domain"])],
            s=20, label=f"Domain {d}", zorder=3,
        )
    _mark_top_residues(ax_prof, resids, top_resids, 0.5, n_domains + 0.5)
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

    # ── Colorbar ────────────────────────────────────────────────────────────
    ax_cb = fig.add_subplot(gs[1, 1])
    ax_cb.axis("off")
    cbar = fig.colorbar(im, ax=ax_cb, fraction=0.8, pad=0.05)
    cbar.set_label("Cross-correlation", fontsize=9)

    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Saved → {out_path}")


def plot_summary(gnm_flucts_1, gnm_flucts_2, anm_flucts_1, anm_flucts_2,
                 per_residue_delta, resids, top_resids, out_path):
    """
    Four-panel summary figure comparing GNM and ANM slow-mode profiles for
    both structures alongside the |ΔD| contact-change signal.
    """
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

    def norm01(x):
        xmin, xmax = x.min(), x.max()
        return (x - xmin) / (xmax - xmin + 1e-12)

    x = np.arange(len(resids))
    panels = [
        (axes[0], gnm_flucts_1,       PALETTE[LABEL_1], f"GNM slow mode 1  –  {LABEL_1}",   "Sq. flucts. (Å²)"),
        (axes[1], gnm_flucts_2,       PALETTE[LABEL_2], f"GNM slow mode 1  –  {LABEL_2}",   "Sq. flucts. (Å²)"),
        (axes[2], anm_flucts_1,       PALETTE[LABEL_1], f"ANM slow mode 1  –  {LABEL_1}",   "Sq. flucts. (Å²)"),
        (axes[3], anm_flucts_2,       PALETTE[LABEL_2], f"ANM slow mode 1  –  {LABEL_2}",   "Sq. flucts. (Å²)"),
    ]

    for ax, flucts, color, title, ylabel in panels:
        _mark_top_residues(ax, resids, top_resids, flucts.min(), flucts.max())
        ax.plot(x, flucts, color=color, lw=1.8)
        ax.fill_between(x, flucts, alpha=0.15, color=color)
        # overlay |ΔD| as thin dashed line (normalised to same scale)
        delta_scaled = norm01(per_residue_delta) * flucts.max()
        ax.plot(x, delta_scaled, color="#333333", lw=1.0, ls=":",
                alpha=0.7, label="|ΔD| (scaled)")
        _style_ax(ax, ylabel=ylabel, title=title)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xticks(range(0, len(resids), 20))
        ax.set_xticklabels(resids[::20], rotation=45, ha="right", fontsize=7)

    axes[-1].set_xlabel("Residue number", fontsize=10)
    fig.suptitle(
        "Concerted Motions: GNM & ANM Slow-Mode Profiles vs. Contact Map Change\n"
        "(pink shading = top-20 |ΔD| residues  |  dotted = |ΔD| signal)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Adenylate Kinase — ANM / GNM Concerted Motion Analysis")
    print("=" * 65)

    # ── Load structures ──────────────────────────────────────────────────────
    print(f"\n[1/6] Loading structures ...")
    ca1 = load_ca(PDB_1, CHAIN)
    ca2 = load_ca(PDB_2, CHAIN)
    resids_1 = ca1.getResnums()
    resids_2 = ca2.getResnums()
    print(f"      {LABEL_1}: {len(resids_1)} Cα atoms  |  {LABEL_2}: {len(resids_2)} Cα atoms")

    # ── GNM ─────────────────────────────────────────────────────────────────
    print(f"\n[2/6] Running GNM (cutoff = {GNM_CUTOFF} Å, {N_SLOW_MODES} slow modes) ...")
    gnm1 = run_gnm(ca1, GNM_CUTOFF, N_SLOW_MODES)
    gnm2 = run_gnm(ca2, GNM_CUTOFF, N_SLOW_MODES)

    # ProDy GNM mode indexing: mode 0 is trivial; slow modes start at index 1
    gnm_mode_idx  = list(range(1, N_SLOW_MODES + 1))
    gnm_flucts_1  = get_sq_fluctuations(gnm1, gnm_mode_idx)  # shape (N_modes, N_res)
    gnm_flucts_2  = get_sq_fluctuations(gnm2, gnm_mode_idx)
    gnm_cc_1      = get_cross_correlation(gnm1, gnm_mode_idx)
    gnm_cc_2      = get_cross_correlation(gnm2, gnm_mode_idx)

    # ── ANM ─────────────────────────────────────────────────────────────────
    print(f"\n[3/6] Running ANM (cutoff = {ANM_CUTOFF} Å, {N_SLOW_MODES} slow modes) ...")
    anm1 = run_anm(ca1, ANM_CUTOFF, N_SLOW_MODES)
    anm2 = run_anm(ca2, ANM_CUTOFF, N_SLOW_MODES)

    # ANM mode indexing: modes 0-5 are rigid-body; slow internal modes start at 6
    anm_mode_idx  = list(range(6, 6 + N_SLOW_MODES))
    anm_flucts_1  = get_sq_fluctuations(anm1, anm_mode_idx)
    anm_flucts_2  = get_sq_fluctuations(anm2, anm_mode_idx)
    anm_cc_1      = get_cross_correlation(anm1, anm_mode_idx)
    anm_cc_2      = get_cross_correlation(anm2, anm_mode_idx)

    # ── Load per-residue |ΔD| from previous analysis (or compute placeholder) 
    print("\n[4/6] Loading contact-map |ΔD| signal ...")
    try:
        npz = np.load("contact_map_results.npz", allow_pickle=True)
        per_residue_delta_raw = npz["per_residue_delta"] \
            if "per_residue_delta" in npz else npz["delta"].mean(axis=1)
        labels_npz = npz["labels"].tolist()
        # Align to structure 1 residue numbering
        npz_resids = np.array([int("".join(filter(str.isdigit, lbl.split(":")[1])))
                                for lbl in labels_npz])
        # Map onto resids_1
        per_residue_delta = np.zeros(len(resids_1))
        for i, r in enumerate(resids_1):
            match = np.where(npz_resids == r)[0]
            if len(match):
                per_residue_delta[i] = per_residue_delta_raw[match[0]]
        print(f"      Loaded |ΔD| for {len(resids_1)} residues from contact_map_results.npz")
    except FileNotFoundError:
        print("      contact_map_results.npz not found — using uniform |ΔD| placeholder.")
        print("      Run contact_map_analysis.py first for full overlap plots.")
        per_residue_delta = np.zeros(len(resids_1))
        for i, r in enumerate(resids_1):
            if r in TOP_CHANGING_RESIDS:
                rank = TOP_CHANGING_RESIDS.index(r)
                per_residue_delta[i] = len(TOP_CHANGING_RESIDS) - rank

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n[5/6] Generating plots ...")

    for label, resids, gf, af, gcc, acc in [
        (LABEL_1, resids_1, gnm_flucts_1, anm_flucts_1, gnm_cc_1, anm_cc_1),
        (LABEL_2, resids_2, gnm_flucts_2, anm_flucts_2, gnm_cc_2, anm_cc_2),
    ]:
        plot_slow_mode_profile(
            gf, resids, TOP_CHANGING_RESIDS, label, "GNM",
            f"gnm_slowmode_{label}.png")

        plot_slow_mode_profile(
            af, resids, TOP_CHANGING_RESIDS, label, "ANM",
            f"anm_slowmode_{label}.png")

        plot_cross_correlation(
            gcc, resids, TOP_CHANGING_RESIDS, label, "GNM",
            f"gnm_cross_correlation_{label}.png")

        plot_cross_correlation(
            acc, resids, TOP_CHANGING_RESIDS, label, "ANM",
            f"anm_cross_correlation_{label}.png")

        plot_overlap_with_contact_change(
            gf, per_residue_delta, resids, TOP_CHANGING_RESIDS, label, "GNM",
            f"gnm_contact_overlap_{label}.png")

        plot_overlap_with_contact_change(
            af, per_residue_delta, resids, TOP_CHANGING_RESIDS, label, "ANM",
            f"anm_contact_overlap_{label}.png")

        plot_dynamic_domains(
            gcc, resids, TOP_CHANGING_RESIDS, N_DOMAINS, label,
            f"dynamic_domains_{label}.png")

    plot_summary(
        gnm_flucts_1[0], gnm_flucts_2[0],
        anm_flucts_1[0], anm_flucts_2[0],
        per_residue_delta, resids_1, TOP_CHANGING_RESIDS,
        "concerted_summary.png",
    )

    # ── Save NMD files for VMD / ProDy NMWiz ─────────────────────────────────
    print("\n[6/6] Saving NMD files for VMD visualisation ...")
    try:
        from prody import writeNMD
        writeNMD(f"{LABEL_1}_anm_modes.nmd", anm1[:N_SLOW_MODES], ca1)
        writeNMD(f"{LABEL_2}_anm_modes.nmd", anm2[:N_SLOW_MODES], ca2)
        print(f"  Saved → {LABEL_1}_anm_modes.nmd")
        print(f"  Saved → {LABEL_2}_anm_modes.nmd")
    except Exception as e:
        print(f"  [WARN] Could not write NMD files: {e}")

    # ── Save numerical results ────────────────────────────────────────────────
    np.savez_compressed(
        OUT_NPZ,
        gnm_flucts_1=gnm_flucts_1, gnm_flucts_2=gnm_flucts_2,
        gnm_cc_1=gnm_cc_1,         gnm_cc_2=gnm_cc_2,
        anm_flucts_1=anm_flucts_1, anm_flucts_2=anm_flucts_2,
        anm_cc_1=anm_cc_1,         anm_cc_2=anm_cc_2,
        per_residue_delta=per_residue_delta,
        resids_1=resids_1,          resids_2=resids_2,
        top_changing_resids=np.array(TOP_CHANGING_RESIDS),
    )
    print(f"  Saved → {OUT_NPZ}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  PEARSON CORRELATIONS  (slow mode sq. flucts. vs |ΔD| signal)\n")
    print(f"  {'Model':<8}  {'Structure':<8}  {'Mode':>6}  {'Pearson r':>10}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*10}")
    for model_name, flucts_list, label in [
        ("GNM", gnm_flucts_1, LABEL_1),
        ("GNM", gnm_flucts_2, LABEL_2),
        ("ANM", anm_flucts_1, LABEL_1),
        ("ANM", anm_flucts_2, LABEL_2),
    ]:
        for k, flucts in enumerate(flucts_list):
            r = float(np.corrcoef(flucts, per_residue_delta[:len(flucts)])[0, 1])
            print(f"  {model_name:<8}  {label:<8}  {k+1:>6}  {r:>+10.4f}")

    print(f"\n  Interpretation:")
    print(f"  • High |r| → the slow mode captures the conformational change")
    print(f"    seen in the contact map difference.")
    print(f"  • Check gnm_cross_correlation_*.png to see which residue pairs")
    print(f"    move in concert (red = correlated, blue = anti-correlated).")
    print(f"  • Check dynamic_domains_*.png for domain assignments.")
    print(f"  • Load *_anm_modes.nmd into VMD (NMWiz plugin) to animate")
    print(f"    the slow modes as 3-D arrows on the protein structure.")
    print("=" * 65)


if __name__ == "__main__":
    main()
