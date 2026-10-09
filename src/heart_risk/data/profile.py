"""
src/heart_risk/data/profile.py
─────────────────────────────────────────────────────────────────────────────
Exploratory Data Analysis (EDA) module.

WHY this module exists:
  To provide an evidence-backed picture of the data's quality and structure.
  This ensures the cleaning phase (Phase 3) is designed from facts, not guesses.

RULES:
  - Read-only: We load `patients_raw`, analyze it in-memory, and never write back.
  - Descriptive: No modeling or train/test splits.
  - Leakage discipline: Feature vs Target analysis is descriptive only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend so figures save to file without opening windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import Engine

from heart_risk.config import Settings
from heart_risk.db import repository

logger = logging.getLogger(__name__)


# ── In-Memory Parsing & Utils ────────────────────────────────────────────────

def get_missing_tokens(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Find distinct tokens that could mean 'missing' (e.g., '', '?', 'NA', whitespace).
    """
    missing_dict = {}
    for col in cols:
        val_counts = df[col].value_counts(dropna=False)
        # Look for suspicious missing-like string values
        suspicious_tokens = {}
        for val, count in val_counts.items():
            if pd.isna(val):
                suspicious_tokens["<NaN>"] = count
            elif isinstance(val, str):
                v_strip = val.strip()
                if v_strip in ["", "?", "NA", "NaN", "null", "None"] or val.isspace():
                    suspicious_tokens[repr(val)] = count
        missing_dict[col] = suspicious_tokens
    return missing_dict

def parse_in_memory(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Create an in-memory copy of the dataframe with parsed dtypes for analysis.
    WHY: We need numeric and categorical types to plot distributions, but we
    must NOT modify the actual database.
    """
    df_parsed = df.copy()
    
    # Standardize missing tokens to np.nan for profiling purposes
    for col in cols:
        # replace blank, "?", etc. with NaN
        df_parsed[col] = df_parsed[col].replace(r'^\s*$', np.nan, regex=True)
        df_parsed[col] = df_parsed[col].replace(['?', 'NA', 'NaN'], np.nan)
        
    numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'num']
    for col in numeric_cols:
        if col in df_parsed.columns:
            df_parsed[col] = pd.to_numeric(df_parsed[col], errors='coerce')
            
    return df_parsed


# ── Investigations ───────────────────────────────────────────────────────────

class EDAProfiler:
    def __init__(self, df_raw: pd.DataFrame, settings: Settings):
        self.settings = settings
        self.df_raw = df_raw
        self.cols = settings.expected_raw_columns
        self.df = parse_in_memory(df_raw, self.cols)
        
        self.reports_dir = settings.reports_dir
        self.fig_dir = self.reports_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        
        self.md_lines = [
            "# Exploratory Data Analysis (EDA) Report",
            "**Note:** This analysis is read-only. No data was modified.",
            ""
        ]

    def run_all(self):
        self.inv_1_structure()
        self.inv_2_missingness()
        self.inv_3_impossible_values()
        self.inv_4_categories()
        self.inv_5_target()
        self.inv_6_numeric_dists()
        self.inv_7_feature_vs_disease()
        self.inv_8_hospital_effect()
        
        self._write_report()

    def _add_md(self, text: str):
        self.md_lines.append(text)
        
    def _save_fig(self, fig: plt.Figure, name: str) -> str:
        path = self.fig_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        # Return relative path for markdown
        return f"figures/{name}.png"

    def inv_1_structure(self):
        self._add_md("## 1. Structure and Integrity")
        self._add_md(f"- **Row count:** {len(self.df_raw)}")
        self._add_md(f"- **Columns:** {len(self.cols)} expected features + metadata.")
        
        # ID Uniqueness
        if 'id' in self.df_raw.columns:
            id_counts = self.df_raw['id'].value_counts()
            dupe_ids = id_counts[id_counts > 1]
            if not dupe_ids.empty:
                self._add_md(f"- **ID Uniqueness:** FAIL. {len(dupe_ids)} duplicate IDs found.")
            else:
                self._add_md("- **ID Uniqueness:** PASS. `id` column is completely unique.")
                
        # Duplicate rows (excluding metadata and id)
        check_cols = [c for c in self.cols if c != 'id']
        dupe_rows = self.df_raw.duplicated(subset=check_cols, keep=False).sum()
        self._add_md(f"- **Duplicate Rows (ignoring ID):** {dupe_rows} rows are duplicates.")
        
        # Rows per dataset
        if 'dataset' in self.df_raw.columns:
            self._add_md("- **Rows per hospital (`dataset`):**")
            counts = self.df_raw['dataset'].value_counts()
            for k, v in counts.items():
                self._add_md(f"  - {k}: {v}")
        self._add_md("")

    def inv_2_missingness(self):
        self._add_md("## 2. Missingness")
        
        missing_tokens = get_missing_tokens(self.df_raw, self.cols)
        self._add_md("### Missing Tokens Inventory")
        for col, tokens in missing_tokens.items():
            if tokens:
                tok_str = ", ".join([f"`{k}`: {v}" for k, v in tokens.items()])
                self._add_md(f"- **{col}**: {tok_str}")
                
        # Missing percent overall and per hospital
        self._add_md("### Missingness per Hospital (%)")
        missing_mask = self.df[self.cols].isna()
        
        # Heatmap of missingness
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(missing_mask, cbar=False, cmap='viridis', yticklabels=False, ax=ax)
        ax.set_title("Missing Data Patterns (Yellow = Missing)")
        fig_path = self._save_fig(fig, "missingness_heatmap")
        self._add_md(f"![Missingness Heatmap]({fig_path})\n*Caption: Co-occurrence of missing values. Yellow lines indicate missing cells.*")
        
        self.df['dataset'] = self.df_raw['dataset'] # ensure dataset is present
        missing_by_hosp = missing_mask.groupby(self.df['dataset']).mean() * 100
        
        # Format as markdown table manually to avoid tabulate dependency
        missing_by_hosp_rounded = missing_by_hosp.round(1)
        self._add_md("| Dataset | " + " | ".join(missing_by_hosp_rounded.columns) + " |")
        self._add_md("|---|" + "|".join(["---" for _ in missing_by_hosp_rounded.columns]) + "|")
        for idx, row in missing_by_hosp_rounded.iterrows():
            self._add_md(f"| {idx} | " + " | ".join(row.astype(str)) + " |")
        self._add_md("\n*Note: High missingness in `ca`, `thal`, and `slope` is strongly clustered by hospital.*")
        self._add_md("")

    def inv_3_impossible_values(self):
        self._add_md("## 3. Impossible and Suspicious Values")
        num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
        
        self._add_md("| Feature | Min | Max | Median | Suspicious Count | Notes |")
        self._add_md("|---------|-----|-----|--------|------------------|-------|")
        for col in num_cols:
            if col not in self.df.columns: continue
            s = self.df[col].dropna()
            if len(s) == 0: continue
            
            c_min, c_max, c_med = s.min(), s.max(), s.median()
            
            susp_count = 0
            notes = []
            if col == 'chol':
                susp_count = (s == 0).sum()
                notes.append(f"{susp_count} zero values")
            elif col == 'trestbps':
                susp_count = (s <= 0).sum()
                notes.append(f"{susp_count} zero/neg values")
            elif col == 'oldpeak':
                susp_count = (s < 0).sum()
                notes.append(f"{susp_count} negative values")
                
            note_str = ", ".join(notes) if notes else "Looks ok"
            self._add_md(f"| {col} | {c_min} | {c_max} | {c_med} | {susp_count} | {note_str} |")
            
        self._add_md("\n**Observation on Zeros:**")
        if 'chol' in self.df.columns:
            zeros_by_hosp = self.df[self.df['chol'] == 0].groupby('dataset').size()
            self._add_md(f"- `chol` zeros by hospital: {zeros_by_hosp.to_dict()}")
            self._add_md("- *Interpretation: Zeros in cholesterol are biologically impossible and likely represent missing data encoding, isolated to specific hospitals.*")
        self._add_md("")

    def inv_4_categories(self):
        self._add_md("## 4. Category and Boolean Inventories")
        cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']
        
        for col in cat_cols:
            if col not in self.df_raw.columns: continue
            val_counts = self.df_raw[col].value_counts(dropna=False)
            self._add_md(f"- **{col}**: {val_counts.to_dict()}")
            
        self._add_md("\n**Notes on `ca`:**")
        self._add_md("- `ca` represents number of major vessels (0-3). Some sources treat it as numeric, but it represents discrete counts. The presence of blanks and its categorical nature in medical diagnosis suggests treating it as categorical or ordinal.")
        self._add_md("")

    def inv_5_target(self):
        self._add_md("## 5. Target Analysis")
        if 'num' not in self.df.columns: return
        
        self.df['disease_binary'] = (self.df['num'] > 0).astype(float)
        # Note: NaN in num propagates to NaN in binary if there were any, but num should be 0-4
        
        # Distribution of num
        counts = self.df['num'].value_counts().sort_index()
        self._add_md(f"- **`num` (0-4) Distribution:** {counts.to_dict()}")
        self._add_md("- *Levels 3 and 4 are rare. Formulating this as a binary task (0 vs >0) is likely more robust for modeling.*")
        
        # Balance by hospital
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=self.df, x='dataset', y='disease_binary', ax=ax)
        ax.set_title("Disease Prevalence by Hospital (Binary Target)")
        ax.set_ylabel("Proportion with Disease (num > 0)")
        fig_path = self._save_fig(fig, "target_balance_hospital")
        
        self._add_md(f"\n![Target Balance]({fig_path})\n*Caption: Proportion of sick patients per hospital. Notice the variance across sites.*")
        self._add_md("")

    def inv_6_numeric_dists(self):
        self._add_md("## 6. Numeric Distributions")
        num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            if col in self.df.columns and not self.df[col].dropna().empty:
                sns.histplot(self.df[col].dropna(), kde=True, ax=axes[i])
                axes[i].set_title(f"Distribution of {col}")
        
        fig_path = self._save_fig(fig, "numeric_histograms")
        self._add_md(f"![Numeric Histograms]({fig_path})\n*Caption: Histograms of continuous features.*")
        
        # Outliers (IQR)
        self._add_md("\n**Outlier counts (IQR rule):**")
        for col in num_cols:
            s = self.df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outliers = ((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum()
            self._add_md(f"- `{col}`: {outliers} outliers")
        self._add_md("")
        
        # Boxplots by hospital
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            if col in self.df.columns and not self.df[col].dropna().empty:
                sns.boxplot(data=self.df, x='dataset', y=col, ax=axes[i])
                axes[i].set_title(f"{col} by Hospital")
                axes[i].tick_params(axis='x', rotation=45)
        
        fig_path = self._save_fig(fig, "numeric_boxplots_hospital")
        self._add_md(f"![Boxplots by Hospital]({fig_path})\n*Caption: Feature distributions separated by hospital source. `chol` zeros clearly skew specific hospitals.*")
        self._add_md("")

    def inv_7_feature_vs_disease(self):
        self._add_md("## 7. Feature vs Disease (Descriptive Only)")
        self._add_md("> **LEAKAGE WARNING:** These plots explore target relationships on the FULL dataset. They are descriptive only and MUST NOT be used for feature selection or encoding choices, as that would leak test-set information into the model design.\n")
        
        # Numeric vs disease
        num_cols = ['age', 'thalch']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, col in enumerate(num_cols):
            if col in self.df.columns and not self.df[col].dropna().empty:
                sns.boxplot(data=self.df, x='disease_binary', y=col, ax=axes[i])
                axes[i].set_title(f"{col} vs Disease")
        
        fig_path = self._save_fig(fig, "feature_vs_disease_numeric")
        self._add_md(f"![Numeric vs Disease]({fig_path})\n*Caption: Older age and lower max heart rate (`thalch`) correlate observationally with disease presence.*")
        
        # Correlation matrix
        corr_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
        corr = self.df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Numeric Correlation Matrix")
        fig_path = self._save_fig(fig, "correlation_heatmap")
        self._add_md(f"\n![Correlation Heatmap]({fig_path})\n*Caption: Linear correlation among continuous features.*")
        self._add_md("")

    def inv_8_hospital_effect(self):
        self._add_md("## 8. Hospital / Source Effect Summary")
        self._add_md("The 4 hospitals (Cleveland, Hungary, Switzerland, VA Long Beach) are heavily heterogeneous:")
        self._add_md("1. **Missingness:** `ca`, `thal`, and `slope` are almost entirely missing in Switzerland and VA Long Beach.")
        self._add_md("2. **Data conventions:** The fake `chol=0` values are isolated to Switzerland and VA Long Beach.")
        self._add_md("3. **Disease Rate:** Prevalence varies significantly across sites.")
        self._add_md("\n**Conclusion:** Pooling these sites blindly is risky. A standard random train/test split might learn hospital-specific quirks (like `chol=0`) instead of true physiological risk. A 'leave-one-hospital-out' cross-validation strategy would be robust to test generalization.")
        self._add_md("")

    def _write_report(self):
        report_path = self.reports_dir / "eda_report.md"
        report_path.write_text("\n".join(self.md_lines), encoding="utf-8")
        logger.info(f"EDA report written to: {report_path}")


def run_profiling(engine: Engine, settings: Settings) -> None:
    """
    Load raw data and execute the EDA profiling suite.
    """
    logger.info("Starting EDA profiling...")
    df_raw = repository.read_raw(engine)
    if df_raw.empty:
        raise ValueError("patients_raw table is empty! Run `make ingest` first.")
    
    profiler = EDAProfiler(df_raw, settings)
    profiler.run_all()
    logger.info("EDA profiling complete.")
