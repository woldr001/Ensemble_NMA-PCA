# Ensemble_NMA-PCA

Analyze an ensemble of protein conformers with three complementary views of motion:

1. **Contact-map variability** from direct Cα distance changes across conformers
2. **Elastic network models** using **GNM** and **ANM** in ProDy
3. **Principal component analysis (PCA)** of the observed structural ensemble

This workflow is useful when you already have a small ensemble of representative structures for the **same protein**—for example from clustering of MD trajectories, aSAM, NMR models, or a set of predicted conformers—and you want to answer questions such as:

- Which residue pairs change distance the most across the ensemble?
- Which residues are consistently the most flexible?
- What are the dominant collective motions in the ensemble?
- Do simple elastic network models predict the same motion patterns observed in the structural ensemble?

The repository currently contains three main scripts:

- `contact_map_analysis.py` — **run first**
- `anm_gnm_analysis.py` — **run second**
- `ensemble_pca.py` — **run last**

---

## Workflow overview

### Step 1: Contact-map analysis (`contact_map_analysis.py`)
This script builds a stack of Cα–Cα distance matrices from all conformers, then computes:

- the **ensemble mean distance matrix**
- the **distance standard deviation matrix**
- a **per-residue flexibility score** based on average pairwise distance variability

This is the most direct description of what actually changes across the ensemble.

### Step 2: GNM / ANM analysis (`anm_gnm_analysis.py`)
This script runs **Gaussian Network Model (GNM)** and **Anisotropic Network Model (ANM)** separately on each conformer using **ProDy**, then computes ensemble-level averages of:

- square fluctuation profiles for the slow modes
- residue–residue cross-correlation matrices
- domain assignments based on hierarchical clustering of the ensemble-mean GNM cross-correlation matrix

It also compares GNM/ANM mobility profiles to the flexibility signal from Step 1.

### Step 3: Ensemble PCA (`ensemble_pca.py`)
This script superposes all conformers to an iteratively refined ensemble mean using the **Kabsch algorithm**, then runs PCA on the aligned Cα coordinates. It reports:

- RMSD of each conformer to the ensemble mean
- variance explained by each principal component
- PC1/PC2 conformer separation
- per-residue displacement magnitudes for the principal motions
- optional comparison of PCA motions with GNM/ANM modes

---

## Repository contents

| File | Purpose |
|---|---|
| `contact_map_analysis.py` | Computes ensemble distance maps and residue flexibility from direct structural differences |
| `anm_gnm_analysis.py` | Computes ensemble GNM/ANM fluctuation and correlation patterns with ProDy |
| `ensemble_pca.py` | Computes PCA of the aligned structural ensemble |
| `contact_map_results.npz` | Optional downstream input created by Step 1 |
| `anm_gnm_results.npz` | Optional downstream input created by Step 2 |

---

## Software requirements

### Python
These scripts use modern type hints such as `str | None`, so use **Python 3.10+**.

### Python packages
Install the required packages with:

```bash
pip install numpy scipy matplotlib prody
```

Package usage by script:

- `contact_map_analysis.py`: `numpy`, `scipy`, `matplotlib`
- `anm_gnm_analysis.py`: `numpy`, `scipy`, `matplotlib`, `prody`
- `ensemble_pca.py`: `numpy`, `matplotlib`

---

## Expected input data

The workflow expects a directory containing **multiple PDB files for the same protein**, typically a small conformer ensemble such as:

```text
cluster_repr_0.pdb
cluster_repr_1.pdb
cluster_repr_2.pdb
...
cluster_repr_9.pdb
```

By default, the scripts assume:

- **10 conformers**
- filenames of the form `cluster_repr_{i}.pdb`
- one protein chain of interest per structure
- standard PDB `ATOM` records with **Cα atoms present**

### Important input assumptions

For best results, all conformers should:

- represent the **same protein / same chain**
- use **consistent residue numbering**
- contain the **same residues whenever possible**
- be in roughly similar structural register

### How residue matching is handled

The scripts do **not** all handle missing residues in the same way:

- `contact_map_analysis.py` and `ensemble_pca.py` explicitly compute the **common residue set** shared across all conformers and use only those residues.
- `anm_gnm_analysis.py` uses the **reference structure length** (the first PDB) and warns/skips conformers whose Cα count differs from the reference.

Because of that, the cleanest workflow is to use an ensemble in which the chosen chain has the **same number of Cα atoms in every conformer**.

---

## Directory layout and output behavior

A simple working layout is:

```text
Ensemble_NMA-PCA/
├── contact_map_analysis.py
├── anm_gnm_analysis.py
├── ensemble_pca.py
└── example_run/
    ├── cluster_repr_0.pdb
    ├── cluster_repr_1.pdb
    ├── ...
    └── cluster_repr_9.pdb
```

### Very important: where outputs are written
All three scripts write output files to the **current working directory** where you launch Python, **not automatically to `PDB_DIR`**.

That means you should either:

- run the scripts from the directory where you want the output files to land, or
- modify the output paths in the `USER SETTINGS` block

---

## How to use the workflow

These scripts currently take their settings from the **`USER SETTINGS` block inside each Python file**. They do not currently use command-line arguments.

Before running, open each script and edit the settings to match your dataset.

---

# 1. `contact_map_analysis.py`

## What this script does

This script:

1. loads all conformer PDBs
2. extracts Cα coordinates
3. identifies the **common residue set** across all conformers
4. builds a distance matrix for each conformer
5. computes the ensemble mean and standard deviation of those matrices
6. computes a per-residue flexibility score:

> average standard deviation of each residue’s distance to all other residues

This flexibility score behaves somewhat like a **theoretical B-factor-like signal**, but it is derived from **ensemble distance variability**, not from crystallographic temperature factors.

## User settings

```python
PDB_DIR      = "/path/to/your/pdbs"
N_CONFORMERS = 10
PDB_PATTERN  = "cluster_repr_{i}.pdb"
PROTEIN_NAME = "Your protein"

CONTACT_THRESHOLD_ANG = 8.0
TOP_N_RESIDUES        = 20
CHAIN_ID              = None
PLOT_INDIVIDUAL_MAPS  = True
```

### Meaning of the settings

| Setting | Meaning |
|---|---|
| `PDB_DIR` | Directory containing the input PDB files |
| `N_CONFORMERS` | Number of conformers to load |
| `PDB_PATTERN` | Filename pattern; `{i}` is replaced by conformer index |
| `PROTEIN_NAME` | Label used in plot titles |
| `CONTACT_THRESHOLD_ANG` | Distance contour used for visual overlay on contact-map plots |
| `TOP_N_RESIDUES` | Number of top flexible residues highlighted in plots / CSV summaries |
| `CHAIN_ID` | Chain to analyze. `None` means use the first chain encountered in each file |
| `PLOT_INDIVIDUAL_MAPS` | Whether to save the 2×5 panel of individual conformer maps |

### Example run

```bash
python contact_map_analysis.py
```

## Output files

| File | Meaning | How to interpret it |
|---|---|---|
| `contact_map_mean.png` | Ensemble-mean Cα distance matrix | Average spatial separation of residue pairs across conformers |
| `contact_map_std.png` | Standard deviation of each residue-pair distance | Large values indicate residue pairs whose separation changes strongly across the ensemble |
| `flexibility_profile.png` | Bar chart of the **top** flexible residues | Larger bars = residues involved in more variable pairwise distance relationships |
| `contact_map_individual.png` | Grid of all individual conformer distance maps | Useful for visually checking whether the ensemble contains obvious sub-states or outliers |
| `contact_map_results.npz` | Saved numerical arrays | Used downstream by the other scripts |
| `most_flexible_residues.csv` | Ranked residue list | Full ranking of residues by ensemble flexibility score |

## Contents of `contact_map_results.npz`

- `dist_matrices` — array of shape `(N_conformers, N_res, N_res)`
- `mean_dist` — ensemble-mean distance matrix
- `std_dist` — ensemble distance standard deviation matrix
- `per_residue_flexibility` — one flexibility value per residue
- `labels` — residue labels such as `A:ALA14`
- `conformer_labels` — conformer names such as `conformer_0`
- `top_flexible_resids` — residue numbers of the top flexible positions

## Interpretation notes

- The diagonal of every distance matrix is zero.
- `CONTACT_THRESHOLD_ANG` affects the **plotted contour overlay**, not the underlying distance calculations.
- `flexibility_profile.png` shows the **top-ranked residues only**; the full ranking is in `most_flexible_residues.csv` and the full numerical vector is in `contact_map_results.npz`.

## Troubleshooting for this step

### Error: `File not found`
Check:

- `PDB_DIR`
- `N_CONFORMERS`
- `PDB_PATTERN`
- whether indexing starts at `0`

### Error: `No Cα atoms found`
Common causes:

- wrong chain selected
- your structure lacks Cα atoms for the chosen chain
- file is not a standard protein PDB

Try setting:

```python
CHAIN_ID = None
```

or explicitly choose the chain you want.

### Too few common residues
That means some residues are missing from one or more conformers. The script intersects residue labels across all structures, so any residue absent from even one conformer is removed from the final analysis.

---

# 2. `anm_gnm_analysis.py`

## What this script does

This script runs **GNM** and **ANM** on each conformer using **ProDy**.

For each conformer it computes:

- GNM slow-mode square fluctuations
- ANM slow-mode square fluctuations
- GNM cross-correlation matrix
- ANM cross-correlation matrix
- an ANM `.nmd` file for visualization in **VMD / NMWiz**

Then it computes ensemble-level averages and compares the NMA-derived fluctuation profiles to the contact-map flexibility signal from Step 1.

## User settings

```python
PDB_DIR      = "/path/to/your/pdbs"
N_CONFORMERS = 10
PDB_PATTERN  = "cluster_repr_{i}.pdb"
PROTEIN_NAME = "Your protein"
CHAIN        = "A"

GNM_CUTOFF   = 7.3
ANM_CUTOFF   = 15.0
N_SLOW_MODES = 3
N_DOMAINS    = 3
TOP_N        = 20
```

### Meaning of the settings

| Setting | Meaning |
|---|---|
| `CHAIN` | Chain analyzed by ProDy. **Note:** this script defaults to `'A'`, unlike the other scripts |
| `GNM_CUTOFF` | Contact cutoff for the GNM Kirchhoff matrix |
| `ANM_CUTOFF` | Contact cutoff for the ANM Hessian matrix |
| `N_SLOW_MODES` | Number of low-frequency modes to analyze |
| `N_DOMAINS` | Number of dynamic domains for GNM clustering |
| `TOP_N` | Number of top flexible residues to highlight |

### Example run

```bash
python anm_gnm_analysis.py
```

## Output files

| File | Meaning | How to interpret it |
|---|---|---|
| `gnm_ensemble_fluctuations.png` | GNM slow-mode fluctuation profiles across conformers | Thick line = ensemble mean, band = ensemble std, faint lines = individual conformers |
| `anm_ensemble_fluctuations.png` | Same, for ANM slow modes | Useful for assessing whether the mobility pattern is robust across conformers |
| `gnm_ensemble_crosscorr.png` | Ensemble-mean GNM cross-correlation matrix | Positive values = correlated motion, negative values = anticorrelated motion |
| `anm_ensemble_crosscorr.png` | Ensemble-mean ANM cross-correlation matrix | Same concept, but based on the ANM model |
| `dynamic_domains_ensemble.png` | Domain assignments from clustering the GNM correlation matrix | Residues grouped into dynamically coherent blocks |
| `gnm_anm_vs_flexibility.png` | GNM/ANM slow-mode profiles vs contact-map flexibility | Pearson `r` indicates agreement between predicted mobility and observed ensemble variability |
| `concerted_summary.png` | Four-panel summary figure | Quick overview of the dominant GNM/ANM signals and contact-map flexibility |
| `anm_gnm_results.npz` | Saved numerical results | Used by `ensemble_pca.py` for optional comparison |
| `conformer_<k>_anm_modes.nmd` | Mode visualization file | Load into VMD with NMWiz to animate conformer-specific ANM motions |

## Contents of `anm_gnm_results.npz`

- `gnm_flucts_all`
- `gnm_flucts_mean`
- `gnm_flucts_std`
- `gnm_cc_mean`
- `anm_flucts_all`
- `anm_flucts_mean`
- `anm_flucts_std`
- `anm_cc_mean`
- `per_residue_flexibility`
- `resids`
- `top_flexible_resids`

## Interpretation notes

### GNM / ANM fluctuation plots
These are **model-derived mobilities**, not direct coordinate RMSFs from simulation trajectories. Use them to compare **predicted soft regions** and **dominant collective motions**.

### Cross-correlation matrices
- **positive values**: residues tend to move together
- **negative values**: residues tend to move in opposite directions
- **near zero**: weak coupling

### Pearson `r` in overlap plots
The overlap plot compares each GNM/ANM mode with the contact-map flexibility signal.

- high positive `r` → the mode highlights the same flexible regions seen in the structural ensemble
- near zero `r` → weak agreement
- negative `r` → the mode emphasizes a different or opposite pattern

### Dynamic domains
These are inferred by clustering the **ensemble-mean GNM cross-correlation matrix**. They are best interpreted as **dynamically coherent regions**, not necessarily rigid biochemical domains.

## Important caveats for this step

### 1. Chain handling is different from the other scripts
This script defaults to:

```python
CHAIN = "A"
```

while the other two scripts default to using the **first detected chain** when `CHAIN_ID = None`.

If you accidentally analyze different chains in different scripts, the comparisons will be misleading. Make sure the chain setting is consistent across all three scripts.

### 2. Conformer length mismatches trigger skipping
If a conformer does not have the same number of Cα atoms as the reference structure, the script prints a warning and skips that conformer:

```text
[WARN] Conformer k has X Cα atoms (expected Y). Skipping.
```

This usually means:

- missing residues in one structure
- wrong chain selected
- inconsistent numbering or preprocessing

Best practice: ensure all conformers have the same analyzed chain and residue completeness before running this step.

### 3. Contact-map flexibility is optional but strongly recommended
If `contact_map_results.npz` is missing, the script uses a placeholder uniform vector so that plotting still works, but the overlap interpretation will not be meaningful.

## Troubleshooting for this step

### Error: `ProDy is not installed`
Install it with:

```bash
pip install prody
```

### Error: `Could not parse 'file.pdb'`
Possible causes:

- malformed PDB file
- wrong file path
- unusual structure content that ProDy cannot parse cleanly

### Warning about skipped conformers
Do not ignore this. If multiple conformers are skipped, the ensemble averages are no longer describing the intended full ensemble. Fix chain selection or residue completeness first.

### NMD file does not animate the expected motion
The fluctuation and overlap analysis explicitly uses the non-trivial ANM slow modes. If the exported `.nmd` animation does not look consistent with the reported slow modes, verify the mode indexing you want to visualize in NMWiz.

---

# 3. `ensemble_pca.py`

## What this script does

This script performs PCA on the ensemble after rigid-body alignment.

It:

1. loads Cα coordinates from all conformers
2. finds the common residue set across conformers
3. superposes structures with the Kabsch algorithm
4. builds the `(N_conformers, 3N)` displacement matrix
5. runs SVD-based PCA
6. reports conformer RMSD and principal motions

This is the best script for answering:

- what is the **dominant structural axis of variation** in the ensemble?
- do conformers separate into clusters along PC1 / PC2?
- which residues contribute most strongly to the observed conformational change?

## User settings

```python
PDB_DIR      = "/path/to/your/pdbs"
N_CONFORMERS = 10
PDB_PATTERN  = "cluster_repr_{i}.pdb"
PROTEIN_NAME = "Your protein"

N_PCS_SHOW   = 5
CHAIN_ID     = None
```

### Example run

```bash
python ensemble_pca.py
```

## Output files

| File | Meaning | How to interpret it |
|---|---|---|
| `pca_scree.png` | Variance explained by each PC and cumulative variance | Tells you how many PCs are needed to capture most of the ensemble variance |
| `pca_scatter.png` | Projection of conformers onto PC1 and PC2 | Separates conformers along dominant structural axes; labels include RMSD to the mean |
| `pca_pc1_profile.png` | Per-residue displacement magnitude for PC1 | Highlights residues most involved in the largest collective motion |
| `pca_pc2_profile.png` | Same for PC2 | Often captures the second major concerted motion |
| `pca_gnm_comparison.png` | PCA residue profiles vs GNM/ANM fluctuations | Pearson `r` measures overlap between actual ensemble motion and ENM-predicted motion |
| `pca_results.npz` | Saved numerical PCA results | Downstream numerical access |

## Contents of `pca_results.npz`

- `pcs` — principal component vectors
- `scores` — conformer coordinates in PC space
- `var_explained` — fractional variance explained by each PC
- `eigenvalues` — PCA eigenvalues
- `mean_struct` — ensemble mean structure
- `aligned_coords` — superposed coordinates
- `pc1_profile` — per-residue displacement profile for PC1
- `pc2_profile` — per-residue displacement profile for PC2
- `resids` — residue numbers
- `rmsd_to_mean` — RMSD of each conformer to the ensemble mean

## Interpretation notes

### Scree plot
A steep drop after PC1 or PC2 means the ensemble is dominated by one or two major collective motions.

### PC1 vs PC2 scatter
Each point is one conformer.

- conformers close together are structurally similar after alignment
- widely separated conformers differ strongly along those PCs
- apparent clusters may correspond to substates

### Per-residue PC profiles
The bar height is the **magnitude of the displacement vector** for that residue in a given PC. Tall bars mark residues that contribute strongly to that mode.

### PCA vs GNM/ANM comparison
This comparison is especially useful when you want to know whether the **observed ensemble motion** resembles the **low-frequency elastic-network motions**.

## Important caveats for this step

### Number of meaningful PCs
With `N` conformers, PCA can return at most **`N - 1` non-trivial PCs** after mean-centering.

For the default 10-conformer workflow, that means up to **9 non-trivial PCs**.

### Common residue set is used
Like the contact-map script, PCA only uses residues shared across all conformers.

### Optional annotations and comparisons
- If `contact_map_results.npz` is present, top flexible residues are shaded in the PC profile plots.
- If `anm_gnm_results.npz` is present, `pca_gnm_comparison.png` is generated.

If those files are missing, PCA still runs normally.

## Troubleshooting for this step

### PCA outputs look noisy or hard to interpret
Possible causes:

- ensemble contains multiple unrelated structural states
- poor residue consistency across structures
- too few conformers
- conformers are nearly identical, so variance is tiny

### PC comparison plot is missing
That means `anm_gnm_results.npz` was not found in the working directory.

### Flexibility shading is missing on PC plots
That means `contact_map_results.npz` was not found in the working directory.

---

## Recommended run order

From the directory where you want all output files saved:

```bash
python contact_map_analysis.py
python anm_gnm_analysis.py
python ensemble_pca.py
```

That gives the richest interpretation because:

- Step 2 can compare GNM/ANM to contact-map flexibility
- Step 3 can annotate PC profiles with top flexible residues and compare PCA to GNM/ANM

---

## Minimal example

Suppose your PDB files are in:

```text
/path/to/IPNS_ensemble/
```

and are named:

```text
cluster_repr_0.pdb
...
cluster_repr_9.pdb
```

Then set in all three scripts:

```python
PDB_DIR = "/path/to/IPNS_ensemble"
N_CONFORMERS = 10
PDB_PATTERN = "cluster_repr_{i}.pdb"
PROTEIN_NAME = "IPNS"
```

Also make sure the chain setting is consistent across scripts, for example:

```python
CHAIN_ID = "A"   # contact_map_analysis.py
CHAIN    = "A"   # anm_gnm_analysis.py
CHAIN_ID = "A"   # ensemble_pca.py
```

Then run:

```bash
cd /path/to/output_directory
python /path/to/repo/contact_map_analysis.py
python /path/to/repo/anm_gnm_analysis.py
python /path/to/repo/ensemble_pca.py
```

---

## Interpreting the workflow as a whole

The three steps answer related but different questions:

| Analysis | What it measures | Best use |
|---|---|---|
| Contact-map variability | Direct structural variability already present in the ensemble | Find flexible residues and changing residue-pair separations |
| GNM/ANM | Predicted low-frequency motions from elastic network models | Identify concerted motions, correlations, and soft dynamic regions |
| PCA | Dominant axes of structural variance in the aligned ensemble | Summarize observed conformational subspace and major collective deformations |

A strong story often looks like this:

- residues with high contact-map flexibility are also highlighted in GNM/ANM slow modes
- PCA shows that the conformers separate mainly along one or two collective motions
- PCA–GNM/ANM correlations suggest that the observed ensemble follows physically reasonable soft modes

---

## Common pitfalls

### 1. Inconsistent chain settings
This is the easiest way to get misleading comparisons. Keep the chain selection consistent across all scripts.

### 2. Missing residues in some conformers
This reduces the common residue set in Steps 1 and 3 and can cause warnings or skipped conformers in Step 2.

### 3. Running from the wrong directory
The scripts read input PDBs from `PDB_DIR`, but write outputs into the **current working directory**.

### 4. Assuming these are trajectory RMSFs
The contact-map flexibility and GNM/ANM fluctuation profiles are related to flexibility, but they are not the same thing as RMSF computed directly from a time-resolved trajectory.

### 5. Over-interpreting a very small ensemble
With only a handful of conformers, PCA and even contact-map variability can be dominated by a few structural outliers. Always inspect the individual conformer maps and the PCA scatter.

---

## Example: running this workflow as an `sbatch` Slurm job

The simplest HPC strategy is to run all three scripts sequentially in one batch job.

```bash
#!/bin/bash --login
#SBATCH --job-name=ensemble_nma_pca
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Example environment setup: adapt to your cluster
module purge
module load Python/3.11.3-GCCcore-12.3.0

# If you use a virtual environment or conda env, activate it here instead
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ensemble_nma_pca

# Directory where you want outputs written
cd /path/to/your/run_directory

# Optional: force a non-interactive matplotlib backend on headless nodes
export MPLBACKEND=Agg

python /path/to/Ensemble_NMA-PCA/contact_map_analysis.py
python /path/to/Ensemble_NMA-PCA/anm_gnm_analysis.py
python /path/to/Ensemble_NMA-PCA/ensemble_pca.py
```

### Notes for Slurm usage

- The scripts are **CPU-only**; no GPU is required.
- `cpus-per-task=4` and `mem=8G` is usually plenty for a 10-conformer Cα-only workflow, though very large proteins may need more memory.
- Because output files are written to the current working directory, `cd` into the directory where you want plots and `.npz` files saved before launching Python.
- Make sure the Python scripts themselves point to the correct `PDB_DIR` in their `USER SETTINGS` block.

### Example submission

```bash
mkdir -p logs
sbatch run_ensemble_nma_pca.sbatch
```

---

## Optional: inspecting saved `.npz` files

You can inspect numerical outputs in Python:

```python
import numpy as np

cm = np.load("contact_map_results.npz", allow_pickle=True)
print(cm.files)
print(cm["per_residue_flexibility"].shape)

nma = np.load("anm_gnm_results.npz", allow_pickle=True)
print(nma.files)
print(nma["gnm_flucts_mean"].shape)

pca = np.load("pca_results.npz", allow_pickle=True)
print(pca.files)
print(pca["scores"].shape)
```

---

## Suggested preprocessing before using this workflow

If you are generating ensembles from MD or structure prediction pipelines, it is helpful to:

- remove non-protein clutter if it interferes with parsing
- standardize residue numbering
- ensure the same chain exists in every file
- ensure missing loops or termini are handled consistently
- cluster first if you are reducing a large ensemble to representative structures

---

## Summary

This workflow is designed to give a new user three complementary views of conformational behavior from a small structural ensemble:

- **what changed directly** (`contact_map_analysis.py`)
- **what an elastic network predicts should move collectively** (`anm_gnm_analysis.py`)
- **what dominant structural axes actually exist in the ensemble** (`ensemble_pca.py`)

When used together, the outputs provide a compact and interpretable picture of residue flexibility, domain-scale motion, and agreement between model-based and data-driven descriptions of protein dynamics.
