# Brussels Loneliness Regression Analysis

Geospatial regression analysis of isolation rates across Brussels statistical sectors using OLS and Geographically Weighted Regression (GWR).

---

## Overview

**Research question:** What socioeconomic and urban factors explain spatial variation in isolation rates across Brussels statistical sectors?

**Unit of analysis:** Statistical sectors (*secteurs statistiques*) — approximately 2,000 administrative sub-units covering the Brussels-Capital Region.

**Time period:** Primarily 2021 data. Private garden access from 2001; average distance to basic goods from 2022.

**Methodology:**
1. **OLS (Ordinary Least Squares):** Estimates global (region-wide) linear relationships between the isolation rate and socioeconomic/urban predictors.
2. **GWR (Geographically Weighted Regression):** Allows regression coefficients to vary spatially, revealing local heterogeneity in relationships. Implemented using a Gaussian kernel with a fixed bandwidth of 5,000 m in EPSG:31370 (Belgian Lambert 72).

All coefficients are standardised (mean = 0, std = 1), making them directly comparable across variables.

---

## Data Sources

Data is fetched from the [Brussels Perspective geodata portal](https://geodata.perspective.brussels/) via REST API:

```
https://geodata.perspective.brussels/api/geodata/mdq/geojson/{ID}/fr
```

| Role | API ID | Description | Year | Unit |
|------|--------|-------------|------|------|
| **Dependent** | 25137 | Share of isolated persons aged 30+ in total private households | 2021 | % |
| Independent | 63403 | Median net taxable income per inhabitant | 2021 | € |
| Independent | 59638 | Average distance to nearest basic good or service | 2022 | m |
| Independent | 24586 | Share of dwellings with private garden access | 2001 | % |
| Independent | 25435 | Share of EU-born population (non-Belgian) | 2021 | % |
| Independent | 62141 | Population growth rate | 2021 | % |
| Independent | 24714 | Share of population aged 30–44 | 2021 | % |
| Independent | 25278 | Share of population aged 65+ | 2021 | % |

Variable names follow the French naming convention from the Brussels Perspective API. English display labels are used throughout the notebook.

---

## Repository Structure

```
.
├── brussels_loneliness_regression_v1_cleaned.ipynb  # Main analysis notebook
├── requirements.txt                                 # Python dependencies
├── README.md                                        # This file
├── .gitignore
└── outputs/
    ├── figures/
    │   ├── metrics_grid.png
    │   ├── summary_statistics_table.png
    │   ├── correlation_matrix.png
    │   ├── ols_results_table.png
    │   ├── moran_tests_comparison.png
    │   ├── regression_diagnostics.png
    │   ├── gwr_local_r2.png
    │   ├── gwr_local_coefficients.png
    │   ├── gwr_statistics_table.png
    │   ├── gwr_significance_summary_table.png
    │   ├── gwr_synthesis_table.png
    │   ├── high_cn_areas_map.png
    │   └── masked_maps/           # Per-variable GWR significance maps
    │       └── masked_*.png
    ├── data/
    │   ├── variable_summary_statistics.csv
    │   ├── high_correlations.csv
    │   ├── gwr_coefficients_statistics.csv
    │   ├── gwr_significance_summary.csv
    │   ├── gwr_synthesis_table.csv
    │   └── high_cn_areas.csv
    └── maps/
        ├── gwr_results.geojson    # GWR local coefficients for all sectors
        └── *.png                  # Individual metric choropleth maps
```

---

## Setup Instructions

**Requirements:** Python 3.10+

```bash
# Clone or download the repository
git clone <repository-url>
cd "Loneliness in Brussels"

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

1. Clone the repository and install dependencies (see above).
2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `brussels_loneliness_regression_v1_cleaned.ipynb`.
4. Run **Kernel → Restart & Run All**.

The notebook fetches data live from the Brussels Perspective API; an internet connection is required. All outputs are saved to the `outputs/` directory.

---

## Outputs

All generated files are saved in `outputs/`:

| File | Description |
|------|-------------|
| `figures/metrics_grid.png` | Grid of choropleth maps for all input variables |
| `figures/summary_statistics_table.png` | Table 1: Descriptive statistics |
| `figures/correlation_matrix.png` | Pearson correlation heatmap |
| `figures/ols_results_table.png` | Table 2: OLS regression results |
| `figures/moran_tests_comparison.png` | Moran's I on raw variable and OLS residuals |
| `figures/regression_diagnostics.png` | OLS diagnostic plots |
| `figures/gwr_local_r2.png` | Local R² choropleth |
| `figures/gwr_local_coefficients.png` | Grid of local GWR coefficient maps |
| `figures/gwr_statistics_table.png` | Table 3: GWR coefficient statistics + Moran's I |
| `figures/high_cn_areas_map.png` | Map of high condition number areas |
| `figures/gwr_significance_summary_table.png` | Table 4: FDR-corrected local significance coverage |
| `figures/gwr_synthesis_table.png` | Table 5: Full GWR synthesis |
| `figures/masked_maps/masked_*.png` | Per-variable: significant & stable sectors only |
| `data/variable_summary_statistics.csv` | Descriptive statistics for all variables |
| `data/high_correlations.csv` | Variable pairs with \|r\| > 0.7 |
| `data/gwr_coefficients_statistics.csv` | Coefficient statistics + Moran's I per variable |
| `data/gwr_significance_summary.csv` | Coverage statistics per variable |
| `data/gwr_synthesis_table.csv` | Full Table 5 data |
| `data/high_cn_areas.csv` | Sectors with condition number above threshold |
| `maps/gwr_results.geojson` | GeoJSON: all GWR local coefficients |

---

## Methodology Notes

### Geographically Weighted Regression (GWR)

GWR extends OLS by allowing each observation to have its own set of regression coefficients, estimated from nearby observations weighted by distance. The Gaussian kernel assigns weight:

```
w_i = exp(-(d_ij / bandwidth)²)
```

where `d_ij` is the distance between focal point `i` and neighbouring point `j`, and `bandwidth` = 5,000 m (fixed, Belgian Lambert 72 / EPSG:31370).

GWR is implemented manually via scikit-learn weighted OLS (rather than the `mgwr` library) for transparency and control over standard error estimation. Methodology follows Fotheringham, Brunsdon & Charlton (2002).

### Local Significance Testing

Local t-statistics are approximated as `coef / local_se` where `local_se` is derived from the cross-sectional standard deviation of local coefficients. False Discovery Rate (FDR) correction (Benjamini-Hochberg) is applied in preference to Bonferroni because:

- Spatial data is inherently correlated, violating Bonferroni's independence assumption.
- BH controls the false discovery rate rather than the family-wise error rate, offering better statistical power for exploratory spatial analysis.

Sectors are considered effectively significant when both: (1) the FDR-corrected p-value < 0.05, and (2) the local condition number ≤ 15.

### Multicollinearity Screening

Variables with Pearson |r| > 0.7 with any other independent variable are flagged and excluded from the regression model (Dormann et al. 2013). Three variables were excluded: impermeable surface share, average age, and annual mean NO₂ concentration.

### Coordinate Reference System

All geospatial operations use **EPSG:31370** (Belgian Lambert 72), the official Belgian national CRS. Coordinates are in metres, enabling direct bandwidth specification for GWR. Maps are re-projected to EPSG:4326 (WGS84) only for display.

---

## References

- Fotheringham, A.S., Brunsdon, C. & Charlton, M. (2002). *Geographically Weighted Regression: The Analysis of Spatially Varying Relationships*. Wiley.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57(1), 289–300.
- Dormann, C.F. et al. (2013). Collinearity: a review of methods to deal with it. *Ecography*, 36, 27–46.
- Belsley, D.A., Kuh, E. & Welsch, R.E. (1980). *Regression Diagnostics*. Wiley.

---

## License

This analysis is released under the [MIT License](https://opensource.org/licenses/MIT).

The underlying data is provided by [Brussels Perspective](https://perspective.brussels/) under their open data terms. Please credit the portal when reusing or citing this work.

---

## Contact

For questions or feedback, please open an issue in this repository.
