---
name: gwr-analysis
description: >
  Run a complete GWR (Geographically Weighted Regression) analysis in Python.
  Covers data loading, variable selection, OLS baseline, Moran's I spatial
  autocorrelation tests, manual GWR fitting, FDR significance masking,
  publication-quality maps and tables. Works on any spatial dataset —
  not just Brussels data.
argument-hint: "[path/to/data or description of dataset]"
---
# GWR Analysis in Python — Complete Methodology Skill

Derived entirely from the Brussels loneliness isolation study
(`monitoring_des_quartiers_isolation_ols_gwr.ipynb`). All patterns, thresholds,
and decisions are grounded in the actual working notebook, not generic GWR
documentation.

---

## 0. Instructions for Claude Code

**Before writing any notebook cell, do these steps in order:**

### 0.1 Inspect data first

Fetch and inspect all endpoints in parallel before scaffolding anything.
For each endpoint check:
- Feature count (do all layers have the same number of sectors?)
- Value ranges (are shares between 0–100? could any be raw counts?)
- Whether `content-disposition` headers are present
  (→ determines whether to use §1.4 pivot or §1.7 multi-endpoint merge)
- Null counts per variable

```
Use WebFetch or requests.head() on each URL.
Log: url | varname | n_features | value_min | value_max | has_content_disposition
```

### 0.2 Choose the loading pattern

| Situation | Pattern to use |
|---|---|
| Single endpoint, long-format, `content-disposition` present | §1.4 + §1.5 (pivot) |
| One URL per variable (most open-data portals) | §1.7 (multi-endpoint merge) |
| Local CSV / shapefile / GeoJSON | §1.6 |

### 0.3 Fixed notebook cell order

Always produce cells in this exact order. Do not add or reorder cells.

```
Cell 1  — Imports                          (§1.1)
Cell 2  — Configuration + DATASETS + VAR_LABELS  (§1.2)
Cell 3  — Data fetch                       (§1.3 or §1.6)
Cell 4  — Build wide-format GeoDataFrame   (§1.4+1.5 or §1.7)
Cell 5  — Data validation + RUN_GWR gate   (§1.8)
Cell 6  — Descriptive maps
Cell 7  — Correlation screen + variable selection  (§3)
Cell 8  — Prepare regression data (dropna, standardise, gdf_gwr, coords)  (§6.0)
Cell 9  — OLS global model
Cell 10 — Moran's I on OLS residuals  → decides whether GWR is warranted  (§4)
Cell 11 — Bandwidth selection              (§5, gated by RUN_GWR)
Cell 12 — GWR fitting loop                 (§6.1, gated by RUN_GWR)
Cell 13 — Moran's I on local coefficients  (§6.2, gated by RUN_GWR)
Cell 14 — FDR + CN masking                 (§7, gated by RUN_GWR)
Cell 15 — Local R² map                     (gated by RUN_GWR)
Cell 16 — Masked coefficient maps          (§7.4, gated by RUN_GWR)
Cell 17 — OLS results table                (§8)
Cell 18 — GWR summary table                (§8, gated by RUN_GWR)
```

---

## 1. Complete Workflow

```
Inspect data → choose loading pattern → wide GeoDataFrame
    → validate + set RUN_GWR → descriptive maps
    → correlation screen → standardise
    → OLS + Moran's I on residuals → [if RUN_GWR]:
        bandwidth → GWR fitting → Moran's I on coefs
        → FDR + CN masking → masked maps → synthesis table
```

### 1.1 Imports

```python
import os, warnings, requests, string
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt, seaborn as sns
from time import time
from shapely.geometry import shape
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import cdist
from scipy.stats import t as t_dist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.multitest import multipletests
from libpysal.weights import Queen
from esda.moran import Moran
from splot.esda import moran_scatterplot

warnings.filterwarnings('ignore')
```

### 1.2 Configuration cell (always second, after imports)

```python
import os

# ── Analysis Parameters ───────────────────────────────────────────────────
DEPENDENT_VAR      = 'your_snake_case_var_name'
GWR_BANDWIDTH      = 5000          # metres in projected CRS (or use Sel_BW — see §5)
CORR_THRESHOLD     = 0.7           # |r| above which variables are flagged
CN_THRESHOLD       = 10            # Benassi & Iglesias-Pascual (2025) recommend 10
SIGNIFICANCE_LEVEL = 0.05
N_WORKERS          = 10
REQUEST_TIMEOUT    = 15
TARGET_CRS         = 'EPSG:31370'  # projected CRS in metres — change for your country

# ── Dataset registry ──────────────────────────────────────────────────────
# Tuple: (url, snake_case_varname, human_readable_label)
# First entry = dependent variable; rest = independent variables.
DATASETS = [
    ('https://...', 'dep_var_name',      'Dependent variable (units)'),
    ('https://...', 'indep_var_name_1',  'Predictor 1 (units)'),
    ('https://...', 'indep_var_name_2',  'Predictor 2 (units)'),
]

# ── Human-readable labels ─────────────────────────────────────────────────
# Used in ALL maps, tables, and axis titles from Cell 6 onward.
# Define here so every downstream cell can do VAR_LABELS.get(var, var).
VAR_LABELS = {d[1]: d[2] for d in DATASETS}

# ── Output Directories ─────────────────────────────────────────────────────
OUT_FIGURES = 'outputs/figures'
OUT_DATA    = 'outputs/data'
OUT_MAPS    = 'outputs/maps'
OUT_MASKED  = 'outputs/figures/masked_maps'

for d in [OUT_FIGURES, OUT_DATA, OUT_MAPS, OUT_MASKED]:
    os.makedirs(d, exist_ok=True)
```

**Why:** Centralising all tunable numbers means one cell to touch when
re-running with different parameters. `DATASETS` is the single source of truth
for URLs, variable names, and labels — `VAR_LABELS` is derived from it
automatically so it is always in sync. Defining `VAR_LABELS` here (not in §8)
means map titles and axis labels work from Cell 6 onward.

### 1.3 Parallel API fetch

Use `fetch_dataset(url, varname)` — varname is always **explicit** from the
`DATASETS` registry, never parsed from response headers. Many APIs (including
most open-data portals) do not set `content-disposition`; relying on it silently
crashes or produces garbage variable names.

```python
def fetch_dataset(url, varname):
    """Fetch one GeoJSON endpoint. varname comes from DATASETS, not headers."""
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return {'varname': varname, 'data': r.json(), 'status': 'ok', 'error': None}
    except Exception as e:
        return {'varname': varname, 'data': None, 'status': 'failed', 'error': str(e)}

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    raw_results = list(executor.map(lambda d: fetch_dataset(d[0], d[1]), DATASETS))

failed = [r for r in raw_results if r['status'] != 'ok']
if failed:
    for f in failed: print(f'FAILED {f["varname"]}: {f["error"]}')

valid_results = [r for r in raw_results if r['status'] == 'ok']
print(f'{len(valid_results)}/{len(DATASETS)} endpoints fetched successfully')
```

### 1.4 Build GeoDataFrame (single long-format endpoint → pivot)

Use this pattern **only** when a single endpoint returns all variables in
long format (one row per sector × variable, with `metric_name` and `value`
columns). It requires the API to set a `content-disposition` header with a
descriptive filename.

```python
def extract_metadata(filename):
    """Extract year, area type, and metric name from an API filename.
    REQUIRES content-disposition header — not available on all APIs."""
    import re
    parts = filename.replace('.geojson', '').split('_')
    year = next((int(p) for p in parts if re.match(r'^\d{4}$', p)), None)
    area = parts[-1]
    metric_name = ' '.join(p for p in parts
                           if not re.match(r'^\d{4}$', p) and p != area)
    return year, area, metric_name

all_data = []
for result in valid_results:
    year, area, metric_name = extract_metadata(result['filename'])
    for feature in result['data']['features']:
        props = feature['properties']
        all_data.append({
            'geometry': shape(feature['geometry']),
            'id': feature.get('id'),
            'name': props.get('name'),
            'value': props.get('value'),
            'year': year,
            'metric_name': metric_name,
            'area_type': area,
        })

# CRITICAL: use a projected CRS (metres) for GWR bandwidth to be meaningful
gdf = gpd.GeoDataFrame(all_data, crs='EPSG:31370')
```

### 1.5 Pivot to analysis-ready wide format (single long-format endpoint)

```python
analysis_df = gdf.pivot_table(
    index=['id', 'name'],
    columns='metric_name',
    values='value',
    aggfunc='first'
).reset_index()

# Normalise column names — API filenames use hyphens
analysis_df.columns = [col.replace('-', '_') for col in analysis_df.columns]
```

### 1.7 Multi-endpoint merge (one URL per variable — the common case)

Use this pattern when each variable comes from a **separate API endpoint**
(e.g. perspective.brussels, most national open-data portals). Variable names
come from `DATASETS` — no header parsing needed.

```python
gdfs = {}
for result in valid_results:
    varname = result['varname']
    rows = [
        {
            'id'      : str(feat.get('id', feat['properties'].get('id', ''))),
            'name'    : feat['properties'].get('name', ''),
            'geometry': shape(feat['geometry']),
            varname   : feat['properties'].get('value'),
        }
        for feat in result['data']['features']
    ]
    gdfs[varname] = gpd.GeoDataFrame(rows, crs='EPSG:31370')
    print(f'{varname}: {len(gdfs[varname])} features, '
          f'{gdfs[varname][varname].isna().sum()} nulls')

# Inner join — only keep sectors present in ALL layers
# Log before/after: inner join silently drops sectors missing from any layer
gdf_merged = gdfs[DEPENDENT_VAR][['id', 'name', 'geometry', DEPENDENT_VAR]]
indep_var_names = [d[1] for d in DATASETS if d[1] != DEPENDENT_VAR]
for varname in indep_var_names:
    gdf_merged = gdf_merged.merge(
        gdfs[varname][['id', varname]], on='id', how='inner'
    )

print(f'\nAfter inner join: {len(gdf_merged)} sectors '
      f'(started with {max(len(g) for g in gdfs.values())})')
gdf_merged = gdf_merged.set_geometry('geometry').set_crs('EPSG:31370')
```

**Key differences from the pivot pattern:**
- Feature counts often differ per layer — always log the before/after count
- No `content-disposition` header → no `extract_metadata()` → names from `DATASETS`
- Result is already wide-format; no pivot step needed

### 1.6 Alternative: Loading Data from File

Use this when your data is already in a CSV, shapefile, or GeoJSON — not fetched from an API.
The downstream code (§3–§9) is identical regardless of data source.

```python
# ── Option A: CSV with lat/lon columns ─────────────────────────────────────
df  = pd.read_csv('data.csv')
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
    crs='EPSG:4326',          # WGS84 if coordinates are decimal degrees
).to_crs('EPSG:XXXXX')        # re-project to a local CRS in metres

# ── Option B: Shapefile or GeoJSON (attributes already joined) ──────────────
gdf = gpd.read_file('sectors.shp')   # or .geojson
if gdf.crs.to_epsg() != XXXXX:       # your target EPSG
    gdf = gdf.to_crs('EPSG:XXXXX')

# ── Option C: Separate spatial and attribute files ──────────────────────────
gdf_spatial   = gpd.read_file('sectors.geojson')
df_attributes = pd.read_csv('attributes.csv')   # must share a key column
gdf = gdf_spatial.merge(df_attributes, on='sector_id', how='left')
gdf = gdf.to_crs('EPSG:XXXXX')

# ── After any loading method — verify before proceeding ────────────────────
print(gdf.crs)                                         # confirm projected CRS
print(f'{len(gdf)} observations, {gdf.shape[1]} columns')
print(gdf[[dep_var] + indep_vars].isna().sum())        # NaN counts per column
```

**Key requirement:** The GeoDataFrame must be in a **projected CRS (metres)**
before GWR. Pass the EPSG code of your country's national grid (e.g. 27700 for
British National Grid, 2154 for French Lambert-93, 31370 for Belgian Lambert 72).
Never run GWR in EPSG:4326 (degrees) — the bandwidth will be meaningless.

The result in all three cases is a wide-format GeoDataFrame with one row per
geographic unit (sector / municipality / ward) and one column per variable,
matching the format produced by §1.5.

### 1.8 Data validation and RUN_GWR gate

Always run this cell immediately after loading data — before any analysis.
`RUN_GWR` is a boolean used to skip all GWR-specific cells when sample size
is too small (see gate pattern below).

```python
all_vars = [DEPENDENT_VAR] + [d[1] for d in DATASETS if d[1] != DEPENDENT_VAR]

# 1. Summary statistics
print(gdf_merged[all_vars].describe().round(2).to_string())

# 2. Value-range anomaly check — flag variables that look like counts not shares
print('\nValue-range check:')
for var in all_vars:
    v = gdf_merged[var].dropna()
    if v.max() > 200:
        print(f'  WARNING: {VAR_LABELS.get(var, var)} max={v.max():.0f} '
              f'— may be a raw count rather than a % share')

# 3. Null counts after merge
print('\nNulls per variable:')
print(gdf_merged[all_vars].isna().sum().to_string())

# 4. Sample size gate
n_complete = gdf_merged[all_vars].dropna().shape[0]
print(f'\nComplete observations: {n_complete}')

if n_complete < 100:
    print('WARNING: n < 100. GWR is not reliable. Proceeding with OLS only.')
    RUN_GWR = False
elif n_complete < 300:
    print('CAUTION: n < 300. GWR results are exploratory only.')
    RUN_GWR = True
else:
    print(f'n = {n_complete} >= 300. GWR is appropriate.')
    RUN_GWR = True

print(f'\nRUN_GWR = {RUN_GWR}')
```

**Gate pattern — add this at the top of every GWR-specific cell:**

```python
if not RUN_GWR:
    print('Skipped — RUN_GWR=False (n < 100). See OLS results above.')
else:
    pass  # cell body here
```

GWR-specific cells: bandwidth selection, GWR fitting loop, Moran's I on
coefficients, FDR + CN masking, local R² map, masked coefficient maps,
GWR summary table.

---

## 2. Known Bugs and Fixes

### 2.1 mgwr + Python 3.12 / NumPy 2.x incompatibility

**Symptom:** `mgwr.gwr.GWR(...).fit()` crashes with a NumPy internals error or
raises `AttributeError` on Python 3.12+ / NumPy 2.x.

**Root cause:** `mgwr`'s GWR fitting code uses deprecated NumPy matrix
operations removed in NumPy 2.0.

**Fix used in this notebook:** Use `mgwr.sel_bw.Sel_BW` *only* for bandwidth
selection (it is stable), then implement GWR fitting manually with
`sklearn.linear_model.LinearRegression` and a Gaussian kernel:

```python
# ✅ Safe: use mgwr only for Sel_BW
from mgwr.sel_bw import Sel_BW

selector = Sel_BW(coords, y, X, fixed=False, kernel='bisquare', spherical=False)
k_optimal = int(selector.search(criterion='AICc', search_method='golden_section'))

# Convert k → metres
distances = cdist(coords, coords)
bandwidth = np.sort(distances, axis=1)[:, k_optimal].mean()

# ✅ Safe: manual GWR loop (pure NumPy + sklearn)
for i in range(n_obs):
    weights = np.exp(-(distances[i, :] / bandwidth) ** 2)   # Gaussian kernel
    model_local = LinearRegression()
    model_local.fit(X, y, sample_weight=weights)
    ...
```

**Alternative:** Pin `numpy<2.0` and use a Python 3.11 environment:
```
numpy==1.26.4
mgwr==2.1.2
```

### 2.2 `Sel_BW` with `fixed=False` returns neighbours, not metres

When `fixed=False` (adaptive bandwidth), `Sel_BW.search()` returns **k** (an
integer count of nearest neighbours), not a distance in metres.

```python
# ❌ Wrong: treating k as metres
bandwidth = k_optimal

# ✅ Correct: convert k → mean distance to k-th nearest neighbour
distances = cdist(coords, coords)
bandwidth = np.sort(distances, axis=1)[:, k_optimal].mean()
```

### 2.3 y must be 2D for mgwr, 1D for sklearn

```python
y_for_sel_bw = gdf_gwr[dep_var].values.reshape(-1, 1)   # for Sel_BW
y_for_sklearn = gdf_gwr[dep_var].values                  # for LinearRegression
```

### 2.4 `libpysal.weights.Queen` re-creation in a loop

Computing `Queen.from_dataframe(gdf_gwr[valid_mask])` inside a loop for
each coefficient is extremely slow. Compute the weights matrix **once**
from the full GeoDataFrame, then subset when needed.

---

## 3. Variable Selection Methodology

### Step 1 — Theoretical selection
Only include variables with a plausible causal pathway to the dependent
variable. Document the exclusion rationale in a `<details>` markdown block.

### Step 2 — Temporal alignment
Prefer variables from the same reference year. Flag mismatches in the API
table (e.g. private garden access 2001 vs. all others 2021).

### Step 3 — Pearson correlation screen

```python
CORR_THRESHOLD = 0.7   # Dormann et al. 2013, Methods in Ecology and Evolution

high_corr_pairs = []
for i in range(len(indep_vars)):
    for j in range(i + 1, len(indep_vars)):
        r = corr_matrix.loc[indep_vars[i], indep_vars[j]]
        if abs(r) > CORR_THRESHOLD:
            high_corr_pairs.append(...)

# Count how many pairs each variable appears in
var_counts = {}
for pair in high_corr_pairs:
    var_counts[pair['Variable 1']] = var_counts.get(pair['Variable 1'], 0) + 1
    var_counts[pair['Variable 2']] = var_counts.get(pair['Variable 2'], 0) + 1
# → remove the variable appearing in the most pairs first
```

### Step 4 — Document exclusions explicitly

```python
# Variables excluded due to high multicollinearity (|r| > CORR_THRESHOLD):
# - 'part_des_surfaces_impermeables': r > 0.7 with [other_var]
# - 'age_moyen': r > 0.7 with share aged 65+
# - 'concentrations_no2': r > 0.7 with [other_var]
excluded_vars = ['part_des_surfaces_impermeables', 'age_moyen',
                 'concentrations_moyennes_annuelles_en_dioxyde_dazote_no2']

indep_vars = [col for col in regression_df.columns
              if col not in ['id', 'name', dep_var] + excluded_vars]
```

### Step 5 — Check sample size before proceeding

| Situation | Recommendation |
|---|---|
| n < 100 | Do not run GWR. Use OLS only. |
| 100 ≤ n < 300 | GWR results are unreliable. Treat as exploratory only. |
| n ≥ 300, p ≤ 10 | GWR is appropriate. |
| n ≥ 1 000, p > 10 | GWR becomes computationally intensive (see §10). |

**Effective local sample size:** Each local model is estimated from all n
observations weighted by distance. For a fixed 5,000 m bandwidth in a compact
urban area (Brussels: ~700 sectors, ~3 km radius), the effective local n is
roughly 150–250. Verify that this is at least 10× the number of predictors.

```python
# After Sel_BW, monitor whether the adaptive bandwidth is suspiciously large
k_ratio = k_optimal / len(coords)
if k_ratio > 0.30:
    print(f'WARNING: bandwidth uses {k_ratio:.0%} of all obs — '
          f'local models are nearly global. GWR may not add value over OLS.')
```

**p-to-n rule:** For p predictors, you need at least 10p complete observations
for OLS. For GWR, apply the same rule to the *effective local n* (≈ half the
obs within bandwidth), not total n. With p = 7 and bandwidth covering 200
sectors, effective local n ≈ 200 >> 70 → acceptable.

### Step 6 — Standardise before regression

```python
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_original)
y_scaled = scaler_y.fit_transform(y_original.reshape(-1, 1)).ravel()

# Persist for downstream GWR cells
standardization_info = {
    'scaler_X': scaler_X, 'scaler_y': scaler_y,
    'indep_vars': indep_vars, 'dep_var': dep_var,
    'n_obs': len(regression_data_scaled),
}
```

**Interpretation of standardised coefficients:**
> beta = 0.40 means: a +1 SD increase in X is associated with a +0.40 SD
> increase in Y, holding all other predictors constant.

---

## 4. Deciding Between OLS and GWR

Run OLS first. GWR is only warranted if OLS leaves spatial structure in
the residuals.

### Decision flowchart

```
Run OLS
  │
  ├─ Moran's I on residuals significant (p < 0.05)?
  │     NO  → OLS is sufficient. Stop here.
  │     YES → spatial autocorrelation in residuals → proceed to GWR
  │
  ├─ n ≥ 300?
  │     NO  → sample too small for reliable GWR. Report OLS with spatial lag.
  │     YES → continue
  │
  └─ Run GWR. Check effective coverage per variable:
        0% for all variables → bandwidth too large or relationships are global.
                               Report OLS; GWR adds noise.
        0% for one variable  → that variable has no spatially varying effect;
                               drop from GWR or report it as globally non-significant.
        > 0% for ≥ 1 variable → report GWR alongside OLS.
```

### What "0% effective coverage" means

A variable has 0% effective coverage when, after FDR correction and CN masking,
**no sector** has a locally significant and numerically stable coefficient.
This can mean:

1. **Globally irrelevant:** the variable does not predict Y anywhere.
   → Drop from GWR; consider dropping from OLS too.
2. **Globally uniform:** the effect is the same everywhere, but not locally
   distinguishable from noise at sector level.
   → Keep in OLS (where global power is higher), exclude from GWR narrative.
3. **Bandwidth too large:** all local models look like the global OLS model.
   → Try a smaller bandwidth and re-check.
4. **Sample too small:** effective local n is too low to detect anything.
   → Report as inconclusive.

### When Moran's I on OLS residuals is ambiguous

If Moran's I p-value is between 0.01 and 0.10, run both OLS and GWR and
compare AICc (lower is better). If GWR AICc > OLS AICc + 3, prefer OLS.
Manual AICc computation for the custom sklearn GWR is complex; use the
Moran's I threshold as the primary decision rule.

---

## 5. Bandwidth Selection

### Option A — Automated (mgwr Sel_BW, recommended)

```python
from mgwr.sel_bw import Sel_BW

# coords: (n, 2) array of centroid coordinates in projected CRS
# y: (n, 1) array, standardised dependent variable
# X: (n, p) array, standardised independent variables

selector = Sel_BW(
    coords,
    y.reshape(-1, 1),
    X,
    fixed=False,        # Adaptive: returns k neighbours, not a fixed radius
    kernel='bisquare',  # Standard for GWR
    spherical=False,    # MUST be False for projected coordinates (metres)
)

k_optimal = int(selector.search(
    criterion='AICc',           # AICc balances fit vs. over-fitting
    search_method='golden_section',
    verbose=True,
))

# Convert k (neighbours) → mean bandwidth in metres
distances = cdist(coords, coords)  # pre-compute and reuse
bandwidth = np.sort(distances, axis=1)[:, k_optimal].mean()

print(f'k = {k_optimal} / {len(coords)} ({k_optimal/len(coords)*100:.1f}%)')
print(f'Mean bandwidth: {bandwidth:.0f} m')
```

### Option B — Fixed bandwidth (simpler, reproducible)

```python
# Use when Sel_BW is unavailable or when replicating a specific study
GWR_BANDWIDTH = 5000   # metres — set in config cell

bandwidth = GWR_BANDWIDTH
distances = cdist(coords, coords)
```

**When to prefer fixed:**
- Reproducing a published result with a stated bandwidth
- When the region has a natural scale (e.g. 5 km ≈ Brussels walking catchment)
- When Sel_BW fails due to library incompatibilities

**Key difference:** adaptive bandwidth (`fixed=False`) varies the spatial reach
by observation density — useful for regions with uneven sector sizes. Fixed
bandwidth applies the same kernel radius everywhere.

---

## 6. GWR Fitting Loop (Manual sklearn Implementation)

### 6.0 Setup: Build gdf_gwr, extract coords, X, and y

This is the critical bridge between data loading (§1) and the fitting loop.
Every downstream cell references `gdf_gwr`, `coords`, `X`, `y`, `indep_vars`.

```python
# 1. Drop rows with missing values in regression variables
regression_data = (
    gdf[['id', 'name', 'geometry', dep_var] + indep_vars]
    .dropna()
    .reset_index(drop=True)
)
print(f'{len(regression_data)} complete obs '
      f'(dropped {len(gdf) - len(regression_data)} with NaN)')

# 2. Standardise (do this AFTER pivot, AFTER dropna)
X_raw = regression_data[indep_vars].values
y_raw = regression_data[dep_var].values
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X_raw)
y = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

# 3. GWR GeoDataFrame — keeps geometry aligned with X/y row indices
gdf_gwr = regression_data.copy()

# 4. Centroid coordinates in projected CRS (metres)
#    CRITICAL: geometry must be in a projected CRS, not EPSG:4326
gdf_proj = gdf_gwr.to_crs('EPSG:31370')   # substitute your national CRS
coords = np.column_stack([
    gdf_proj.geometry.centroid.x,
    gdf_proj.geometry.centroid.y,
])

# 5. Pre-compute full distance matrix (O(n²) — compute ONCE, reuse everywhere)
distances = cdist(coords, coords)   # shape (n_obs, n_obs), values in metres

n_obs = len(coords)
print(f'coords: {coords.shape} | distances: {distances.shape}')
print(f'Dist range: {distances[distances > 0].min():.0f}m – {distances.max():.0f}m')
```

**Row alignment is mandatory.** After `dropna().reset_index(drop=True)`, row
`i` of `X`, `y`, `coords`, and `gdf_gwr` must all refer to the same
observation. Never sort or filter these arrays independently after this point.

### 6.1 The Fitting Loop

```python
# Pre-compute distance matrix ONCE before the loop
coords = np.column_stack([gdf_gwr.geometry.centroid.x,
                          gdf_gwr.geometry.centroid.y])
distances = cdist(coords, coords)   # shape (n, n)

n_obs = len(coords)
n_vars = X.shape[1]

local_coefs        = np.zeros((n_obs, n_vars + 1))  # col 0 = intercept
local_r2           = np.zeros(n_obs)
local_resid        = np.zeros(n_obs)
local_cond_numbers = np.zeros(n_obs)

start_time = time()

for i in range(n_obs):
    # Gaussian kernel — continuous, decays smoothly with distance
    weights = np.exp(-(distances[i, :] / bandwidth) ** 2)

    try:
        model_local = LinearRegression()
        model_local.fit(X, y, sample_weight=weights)

        local_coefs[i, 0] = model_local.intercept_
        local_coefs[i, 1:] = model_local.coef_

        y_pred = model_local.predict(X)
        local_resid[i] = y[i] - y_pred[i]

        # Weighted R²
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        local_r2[i] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Condition number via SVD of weighted X matrix
        W_sqrt = np.diag(np.sqrt(weights))
        X_weighted = W_sqrt @ X
        U, s, Vt = np.linalg.svd(X_weighted, full_matrices=False)
        local_cond_numbers[i] = s.max() / s.min() if s.min() > 1e-10 else np.inf

    except Exception:
        local_coefs[i, :]  = np.nan
        local_r2[i]        = np.nan
        local_resid[i]     = np.nan
        local_cond_numbers[i] = np.inf

    if (i + 1) % max(1, n_obs // 10) == 0:
        print(f'  {(i+1)/n_obs*100:.0f}% complete...')

print(f'Fitted in {time() - start_time:.1f}s')

# Store in GeoDataFrame
gdf_gwr['local_r2']         = local_r2
gdf_gwr['residuals']        = local_resid
gdf_gwr['intercept']        = local_coefs[:, 0]
gdf_gwr['condition_number'] = local_cond_numbers
for i, var in enumerate(indep_vars):
    gdf_gwr[f'coef_{var}']  = local_coefs[:, i + 1]
```

**Naming convention:** coefficients stored as `coef_{snake_case_var_name}` —
used consistently in all downstream cells.

### 6.2 Moran's I on Local GWR Coefficients (Spatial Nonstationarity Test)

This is the key diagnostic that **justifies GWR over OLS**. A significant
Moran's I on a coefficient surface means the effect of that variable is not
randomly distributed across space — it varies in a spatially structured way.

```python
from libpysal.weights import Queen
from esda.moran import Moran

# Compute spatial weights once from the full GWR GeoDataFrame
w = Queen.from_dataframe(gdf_gwr, silence_warnings=True)
w.transform = 'r'   # row-standardise

moran_coef_results = {}

for var in indep_vars:
    col    = f'coef_{var}'
    values = gdf_gwr[col].values
    valid  = ~np.isnan(values)

    if valid.sum() < 30:
        print(f'{var}: too few valid obs ({valid.sum()}) — skipping Moran')
        continue

    # Subset weights matrix to non-NaN observations
    w_sub = Queen.from_dataframe(
        gdf_gwr[valid].reset_index(drop=True), silence_warnings=True
    )
    w_sub.transform = 'r'

    mi = Moran(values[valid], w_sub, permutations=999)
    moran_coef_results[var] = {
        'Moran I': round(mi.I, 3),
        'p-value': round(mi.p_sim, 4),
    }
    sig = '✓ spatial structure' if mi.p_sim < 0.05 else '✗ no structure'
    print(f'{var}: I = {mi.I:.3f}, p = {mi.p_sim:.4f}  {sig}')
```

**Interpretation table:**

| Moran's I on coefficient | p-value | Meaning |
|---|---|---|
| > 0, p < 0.05 | Significant | Effect varies **smoothly** across space → GWR is justified |
| ≈ 0, p ≥ 0.05 | Not significant | Effect is spatially random → this variable behaves globally |
| < 0, p < 0.05 | Significant | Negative spatial autocorrelation (checkerboard) — rare, investigate |

**Report these values in your GWR statistics table (Table 3).** A variable
with Moran's I p ≥ 0.05 on its coefficient surface should be flagged as
"globally uniform" in the synthesis table — its local coefficients are not
spatially structured even if they vary numerically.

**Workflow position:** Run after the GWR fitting loop, before FDR correction.
The Moran's I result informs which variables deserve detailed interpretation.

---

## 7. FDR Correction and CN Masking

### 6.1 Why FDR over Bonferroni

> Spatial data is inherently correlated — adjacent sectors share similar
> values. Bonferroni assumes independence, which is violated; it
> over-penalises and masks real effects. Benjamini-Hochberg (BH) controls
> the *false discovery rate*, offering better power for correlated spatial tests.
> Reference: Benjamini & Hochberg (1995), JRSS-B.

### 6.2 CN threshold

Use **CN_THRESHOLD = 10** (Benassi & Iglesias-Pascual, 2025).
The original Belsley et al. (1980) threshold of 15 is more permissive.
CN > threshold → local model is numerically unstable → mask that location
regardless of significance.

### 6.3 Full implementation

```python
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests

n_obs = len(gdf_gwr)
dof   = n_obs - X.shape[1] - 1

mask_stats   = []
all_sig_maps = {}

for var in indep_vars:
    col       = f'coef_{var}'
    coef_vals = gdf_gwr[col].values

    # Pseudo-SE: std of local coefficients / sqrt(n)
    # (proper SE needs the hat-matrix diagonal; not computed here)
    coef_std  = np.nanstd(coef_vals)
    se_approx = coef_std / np.sqrt(n_obs) if coef_std > 0 else 1.0

    t_stats    = coef_vals / se_approx
    p_vals_raw = 2 * t_dist.sf(np.abs(t_stats), df=dof)

    # Benjamini-Hochberg FDR correction
    valid_mask  = ~np.isnan(p_vals_raw)
    p_vals_fdr  = np.full(n_obs, np.nan)
    if valid_mask.sum() > 0:
        reject, p_corrected, _, _ = multipletests(
            p_vals_raw[valid_mask], alpha=SIGNIFICANCE_LEVEL, method='fdr_bh'
        )
        p_vals_fdr[valid_mask] = p_corrected

    sig_mask      = p_vals_fdr < SIGNIFICANCE_LEVEL
    stable_mask   = gdf_gwr['condition_number'].values <= CN_THRESHOLD
    # "Effective" = significant AND numerically stable
    effective_mask = sig_mask & stable_mask

    gdf_gwr[f'sig_{var}']        = sig_mask.astype(int)
    gdf_gwr[f'stable_sig_{var}'] = effective_mask.astype(int)
    all_sig_maps[var]             = effective_mask

    mask_stats.append({
        'Variable' : var,
        'Pct_sig'  : 100 * sig_mask.sum()      / n_obs,
        'Pct_valid': 100 * effective_mask.sum() / n_obs,
        'N_sig'    : int(sig_mask.sum()),
        'N_valid'  : int(effective_mask.sum()),
    })
```

### 6.4 Masked coefficient maps

```python
for var in indep_vars:
    col            = f'coef_{var}'
    effective_mask = all_sig_maps[var]

    fig, (ax_all, ax_masked) = plt.subplots(1, 2, figsize=(20, 8))

    # Left panel: all sectors
    gdf_gwr.plot(column=col, ax=ax_all, cmap='RdBu_r',
                 edgecolor='black', linewidth=0.2, legend=True)
    ax_all.set_title(f'All sectors\n{var_label}')
    ax_all.axis('off')

    # Right panel: significant & stable only (gray background)
    gdf_gwr.plot(ax=ax_masked, color='lightgray',
                 edgecolor='black', linewidth=0.2, alpha=0.4)
    if effective_mask.sum() > 0:
        gdf_gwr[effective_mask].plot(column=col, ax=ax_masked, cmap='RdBu_r',
                                     edgecolor='black', linewidth=0.2, legend=True)
    ax_masked.set_title(
        f'Significant & stable (CN ≤ {CN_THRESHOLD})\n'
        f'{var_label} — {effective_mask.sum()} sectors '
        f'({100*effective_mask.sum()/n_obs:.1f}%)'
    )
    ax_masked.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUT_MASKED}/masked_{var}.png', dpi=150, bbox_inches='tight')
    plt.close()
```

---

## 8. Publication-Quality Table Template

Used for OLS results (Table 2), GWR statistics (Table 3), and synthesis
(Table 5). Key parameters that produce the academic look:

```python
# ── Generic publication table ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
ax  = fig.add_subplot(111)
ax.axis('off')

table = ax.table(
    cellText   = results_df.values,
    colLabels  = results_df.columns,
    cellLoc    = 'left',
    loc        = 'center',
    colWidths  = [0.25, 0.10, 0.10, 0.08, 0.12, 0.12, 0.12, 0.06],  # adjust to columns
    bbox       = [0, 0.3, 1, 0.65],   # leave room for stats block below
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.8)

# All cells: white background, serif font, no borders by default
for key, cell in table.get_celld().items():
    cell.set_linewidth(0)
    cell.set_edgecolor('black')
    cell.set_facecolor('white')
    cell.set_text_props(family='serif')

# Header row: bold, top+bottom border only
for j in range(len(results_df.columns)):
    cell = table[(0, j)]
    cell.set_text_props(weight='bold', family='serif', size=11)
    cell.visible_edges = 'TB'
    cell.set_linewidth(2.0)

# Data rows: no border, right-align numbers
for i in range(1, len(results_df) + 1):
    for j in range(len(results_df.columns)):
        cell = table[(i, j)]
        cell.set_text_props(family='serif', size=11,
                            ha='left' if j == 0 else 'right')
        cell.visible_edges = ''

# Last row: bottom border (closing line)
for j in range(len(results_df.columns)):
    cell = table[(len(results_df), j)]
    cell.visible_edges = 'B'
    cell.set_linewidth(1.0)

# Model statistics block (below table)
fig.text(0.5, 0.18, stats_text, ha='center', fontsize=10, family='serif',
         bbox=dict(boxstyle='round', facecolor='white',
                   edgecolor='black', linewidth=1.0))

# Title and dependent variable note
fig.text(0.5, 0.97, 'Table N: Title',
         ha='center', fontsize=14, fontweight='bold', family='serif')
fig.text(0.5, 0.08, f'Dependent variable: ...',
         ha='center', fontsize=10, style='italic', family='serif')

plt.tight_layout()
plt.savefig(f'{OUT_FIGURES}/table_name.png', dpi=300,
            bbox_inches='tight', facecolor='white')
```

### Significance stars convention

```python
def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''
```

### Variable label dictionary pattern

```python
# Map internal snake_case names → readable English labels
VAR_LABELS = {
    'Intercept': 'Intercept',
    'distance_moyenne_dacces_aux_4_biens_de_base': 'Avg distance to basic goods',
    'part_de_leurope_des_14_hors_belgique':        'Share EU-born (non-Belgian)',
    # ... one entry per variable
}
# Usage
label = VAR_LABELS.get(var_name, var_name.replace('_', ' ').title())
```

For GWR tables where the variable name comes from
`coef_name.replace('_', ' ').title()`:
```python
# The key must be the title-cased version, not the snake_case name
VAR_LABELS_GWR = {
    'Distance Moyenne Dacces Aux 4 Biens De Base': 'Avg distance to basic goods',
    '--- Condition Number ---': 'Condition Number',
    # ...
}
```

### Coefficient maps: shared symmetric colour scale

```python
# Use the same scale across all coefficient maps for comparability
global_abs_max = max(
    max(abs(gdf_gwr[f'coef_{v}'].min()), abs(gdf_gwr[f'coef_{v}'].max()))
    for v in indep_vars
)
# Then for each map:
gdf_gwr.plot(column=f'coef_{var}', ax=ax, cmap='RdBu_r',
             vmin=-global_abs_max, vmax=global_abs_max, ...)
```

---

## 9. Narrative Interpretation for Non-Technical Audiences

### OLS results
> "The global model explains **X%** of the variation in isolation rates across
> Brussels sectors. Holding all other factors constant, a one-standard-deviation
> increase in [variable] is associated with a **+beta SD** change in the
> isolation rate. This effect is [significant / not significant] at the 5% level."

### Moran's I on residuals
> "If the OLS residuals show significant spatial clustering (Moran's I > 0,
> p < 0.05), the relationships are not uniform across space — a spatially
> varying model (GWR) is more appropriate."
>
> Rule of thumb: if Moran's I drops by **> 50%** from the raw variable to
> the OLS residuals, OLS is already capturing much of the spatial structure.

### GWR bandwidth
> "The model gives more weight to nearby sectors than distant ones when
> estimating the local relationship. The bandwidth (here, **X metres**) is
> the distance at which a sector receives roughly 37% of the weight of the
> focal sector."

### Local R²
> "The colour shows how well the model fits in each neighbourhood. Darker
> green areas are better explained by the chosen variables; grey areas
> may be driven by factors not captured in the model."

### Masked coefficient maps
> "Grey sectors either showed a statistically unreliable effect (after
> correcting for multiple comparisons) or had an unstable local model
> (condition number > threshold). Only the coloured sectors have
> coefficients we can interpret with confidence."

### Effective coverage
> "For [variable], the relationship was significant and stable in **X%**
> of sectors. This means the effect is not uniform across Brussels —
> [interpret direction from the map]."

### Direction of coefficients
> Positive (red): higher [variable] → higher isolation rate in that area.
> Negative (blue): higher [variable] → lower isolation rate in that area.
> The *size* of the coefficient shows how strongly the relationship holds locally.

---

## 10. Common Pitfalls to Avoid

### Data
- **Mixing CRS:** always use a projected CRS (metres) for GWR. Never pass
  lat/lon coordinates when bandwidth is in metres.
  Set `spherical=False` in `Sel_BW` for projected coordinates.
- **Not normalising column names:** API filenames use hyphens; pivot tables
  inherit them. Always do `col.replace('-', '_')` after pivot.
- **Missing values silently dropped:** `dropna()` reduces n_obs. Log the
  count before and after: `print(f'{len(regression_data)} complete obs')`

### Standardisation
- **Standardising before the pivot, not after:** always pivot first, then
  standardise. Standardising the long-format GeoDataFrame produces wrong values.
- **Not preserving scalers:** store `scaler_X` and `scaler_y` in
  `standardization_info` so you can back-transform coefficients later.

### Bandwidth
- **Confusing adaptive k with fixed radius:** `Sel_BW(..., fixed=False)`
  returns *k*, not metres. Convert explicitly via the distance matrix.
- **Recomputing distances inside the loop:** compute `cdist(coords, coords)`
  once before the loop — it is O(n²) and computing it n times is catastrophic.

### GWR fitting
- **Using `mgwr.gwr.GWR().fit()` on Python 3.12+:** crashes with NumPy 2.x.
  Use the manual sklearn loop instead (§2.1).
- **Not computing condition numbers:** without CN, you cannot distinguish
  reliable from unstable local estimates. Always compute and store them.
- **Interpreting masked-out sectors:** never report coefficients for sectors
  where `stable_sig_{var} == 0`. They are noise.

### Significance testing
- **Using raw p-values from n_obs tests:** multiply-testing inflates false
  positives massively for n ≈ 600. Always apply FDR (BH) correction.
- **Applying Bonferroni to spatial data:** violates independence assumption.
  Use BH (§6).
- **Setting CN threshold too high:** the original Belsley et al. threshold
  (15) is too permissive for fine-grained spatial data. Use 10 (Benassi &
  Iglesias-Pascual, 2025).

### Visualisation
- **Not using a shared colour scale across coefficient maps:** makes
  visual comparison impossible. Compute `global_abs_max` once and apply
  it to all panels.
- **Showing all-sectors maps without masking:** always produce a
  paired (all / masked) figure so the reader can see what is being
  suppressed.

### Tables
- **French variable names in published outputs:** always maintain a
  `VAR_LABELS` dictionary mapping internal API names to readable English
  labels. The internal names are data infrastructure, not final output.
- **Saving tables without `facecolor='white'`:** transparent backgrounds
  produce grey PDFs. Always pass `facecolor='white'` to `savefig`.

### Performance (large n)

The manual sklearn GWR loop is O(n²) in memory (distance matrix) and O(n × n × p)
in compute. This is fine up to n ≈ 2,000 sectors; it becomes slow or infeasible
for larger datasets.

| n | Distance matrix size | Approx. runtime | Recommendation |
|---|---|---|---|
| < 500 | < 2 MB | seconds | Direct approach |
| 500–2,000 | 2–32 MB | 1–5 min | Direct approach |
| 2,000–10,000 | 32 MB–800 MB | 5–60 min | Sparse distance cutoff |
| > 10,000 | > 800 MB | Hours / OOM | Grid subsampling or mgwr on Py 3.11 |

**Mitigations:**

```python
# ── Sparse cutoff: only compute distances within a threshold ────────────────
from sklearn.neighbors import BallTree
import numpy as np

tree    = BallTree(np.radians(coords) if spherical else coords)
CUTOFF  = 3 * bandwidth     # metres; beyond this weight ≈ 0
indices = tree.query_radius(coords, r=CUTOFF)   # list of neighbour arrays

# In the loop: only compute weights for neighbours, zero elsewhere
for i in range(n_obs):
    nbrs    = indices[i]
    w_local = np.zeros(n_obs)
    d_local = np.linalg.norm(coords[nbrs] - coords[i], axis=1)
    w_local[nbrs] = np.exp(-(d_local / bandwidth) ** 2)
    model_local.fit(X, y, sample_weight=w_local)
```

```python
# ── Grid subsampling: fit GWR on a subset, interpolate to all units ─────────
# 1. Create a regular grid of focal points covering the study area
# 2. Fit GWR only at grid points (n_grid << n_obs)
# 3. Interpolate coefficients to all observations (e.g. IDW or kriging)
# This is the standard approach in commercial GWR software for large n.
```

- **Use mgwr on Python 3.11:** `mgwr` is C-accelerated and handles n > 10,000
  efficiently. Pin `numpy<2.0` and use a Python 3.11 environment (see §2.1).
- **Parallelise the loop:** the local models are independent — use
  `joblib.Parallel(n_jobs=-1)` to distribute across CPU cores.

---

## References

- Fotheringham, Brunsdon & Charlton (2002). *Geographically Weighted
  Regression.* Wiley.
- Benjamini & Hochberg (1995). Controlling the false discovery rate.
  *JRSS-B*, 57(1), 289–300.
- Dormann et al. (2013). Collinearity: a review of methods. *Ecography*,
  36, 27–46.
- Belsley, Kuh & Welsch (1980). *Regression Diagnostics.* Wiley.
- Benassi & Iglesias-Pascual (2025). [GWR CN threshold recommendation.]
