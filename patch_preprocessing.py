"""
Refactor the preprocessing section of Privat_test.ipynb:
  - Add English markdown explanation cells before each code cell
  - Clean up code (remove Ukrainian comments, duplicate lines)
  - Split large cells into logical units with explanations
  - Keep all global variables / function signatures intact
"""

import json, sys, copy

NOTEBOOK = "Privat_test.ipynb"

with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ── Utility: create a markdown cell ─────────────────────────────────────────
def md_cell(lines, cell_id):
    """Return a new markdown cell dict."""
    if isinstance(lines, str):
        lines = [lines]
    # Ensure every line ends with \n except possibly the last
    formatted = []
    for i, l in enumerate(lines):
        if not l.endswith("\n") and i < len(lines) - 1:
            formatted.append(l + "\n")
        else:
            formatted.append(l)
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": formatted,
    }

def code_cell(lines, cell_id, exec_count=None):
    """Return a new code cell dict."""
    if isinstance(lines, str):
        lines = [lines]
    formatted = []
    for i, l in enumerate(lines):
        if not l.endswith("\n") and i < len(lines) - 1:
            formatted.append(l + "\n")
        else:
            formatted.append(l)
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": formatted,
    }


# ── Find the cells we need to replace by id ─────────────────────────────────
# Build an index: id -> position
id_to_idx = {c.get("id"): i for i, c in enumerate(cells)}

# Cells to refactor (by cell id):
#   "7d1e192e" - markdown about splitting (line 782) - needs English rewrite
#   "d638551d" - markdown about cleaning (line 790) - needs English rewrite
#   "36da0efc" - code: drop empty/dup cols (line 798) - add md + clean
#   "5ea08b69" - code: common preprocessing (line 838) - add md + clean dupes
#   "c7016378" - code: CV splits (line 890) - add md
#   "59a9951b" - code: all preprocessing functions (line 932) - split + add md
#   "33c8bfcc" - code: LR feature selection (line 1030) - add md + clean

# ── Build the replacement cell list ─────────────────────────────────────────
# We will rebuild the section from the first markdown ("7d1e192e") through
# the LR feature selection cell ("33c8bfcc"), then splice it back in.

first_replace_id = "7d1e192e"
last_replace_id = "33c8bfcc"

first_idx = id_to_idx[first_replace_id]
last_idx = id_to_idx[last_replace_id]

print(f"Replacing cells from index {first_idx} (id={first_replace_id}) "
      f"to {last_idx} (id={last_replace_id})")

new_section = []

# ─── 1. Markdown: Why StratifiedGroupKFold + data cleaning intro ────────────
new_section.append(md_cell([
    "## Data Cleaning & Cross-Validation Strategy\n",
    "\n",
    "Before training any model, we need to perform two essential preparation steps:\n",
    "\n",
    "1. **Data cleaning** — remove columns that carry zero useful information (100% missing values) and eliminate exact duplicate columns to reduce dimensionality and noise.\n",
    "2. **Cross-validation design** — since a single entity (`id`) can appear in multiple rows, a naive random split would leak information: the model could memorise patterns from validation entities it has already seen during training. To prevent this **group-level data leakage**, we use `StratifiedGroupKFold`, which:\n",
    "   - Groups all rows belonging to the same `id` together, ensuring no entity appears in both the train and validation sets.\n",
    "   - Stratifies by the target column to keep the positive-class rate roughly equal across folds — critical with a ~38.5:1 class imbalance (≈27 000 negatives vs ≈700 positives)."
], cell_id="md_cleaning_intro"))

# ─── 2. Code: Drop empty & duplicate columns ────────────────────────────────
new_section.append(md_cell([
    "### Step 1 — Remove Empty and Duplicate Columns\n",
    "\n",
    "Columns that are 100% missing (`NaN`) provide absolutely no signal to any model and only increase memory usage. Likewise, exact duplicate columns (identical values in every row) are redundant — keeping them would waste computation and could confuse regularised models.\n",
    "\n",
    "After removal we rebuild the authoritative column lists `cat_cols`, `num_cols`, and `FINAL_FEATURES`."
], cell_id="md_drop_cols"))

new_section.append(code_cell([
    "cols_before = len(df.columns)\n",
    "df = df.drop(columns=empty_cols, errors=\"ignore\")\n",
    "print(f\"Dropped {cols_before - len(df.columns)} columns with 100% missing values.\")\n",
    "\n",
    "cat_cols = sorted([c for c in df.columns if c.startswith(\"cat_\")])\n",
    "num_cols = sorted([c for c in df.columns if c.startswith(\"num_\")])\n",
    "\n",
    "feature_cols = cat_cols + num_cols\n",
    "dup_groups = df[feature_cols].T.duplicated(keep=\"first\")\n",
    "dup_cols = dup_groups[dup_groups].index.tolist()\n",
    "df = df.drop(columns=dup_cols, errors=\"ignore\")\n",
    "print(f\"Dropped {len(dup_cols)} exact duplicate columns.\")\n",
    "\n",
    "cat_cols = sorted([c for c in df.columns if c.startswith(\"cat_\")])\n",
    "num_cols = sorted([c for c in df.columns if c.startswith(\"num_\")])\n",
    "\n",
    "FINAL_FEATURES = cat_cols + num_cols\n",
    "\n",
    "print(f\"\\n=== Final feature set: {len(cat_cols)} categorical \"\n",
    "      f\"+ {len(num_cols)} numerical = {len(FINAL_FEATURES)} total ===\")"
], cell_id="36da0efc"))

# ─── 3. Code: Common preprocessing — extract X, y, groups ───────────────────
new_section.append(md_cell([
    "### Step 2 — Extract Features, Target and Groups\n",
    "\n",
    "Here we prepare the global matrices that every downstream model will use:\n",
    "\n",
    "| Variable | Description |\n",
    "|----------|-------------|\n",
    "| `X_all` | Feature matrix (all final features) |\n",
    "| `y_all` | Binary target vector (`gb`) |\n",
    "| `groups_all` | Entity ids — used by `StratifiedGroupKFold` to prevent leakage |\n",
    "| `final_num` / `final_cat` | Convenience lists of numerical / categorical feature names |\n",
    "| `final_mnar` | Numerical columns that contain at least one missing value (MNAR — Missing Not At Random). These will later receive binary *is_missing* indicator features inside the Logistic Regression pipeline. |\n",
    "\n",
    "**Important:** Categorical columns are cast to `int` with missing values filled by `-1`. This is a safe global operation because it does not depend on any train/val split — it is a deterministic type cast, not a learned transform."
], cell_id="md_extract_xy"))

new_section.append(code_cell([
    "for col in cat_cols:\n",
    "    df[col] = df[col].fillna(-1).astype(int)\n",
    "\n",
    "X_all = df[FINAL_FEATURES].copy()\n",
    "y_all = df[TARGET_COL].values\n",
    "groups_all = df[ID_COL].values\n",
    "\n",
    "final_num = [c for c in FINAL_FEATURES if c.startswith(\"num_\")]\n",
    "final_cat = [c for c in FINAL_FEATURES if c.startswith(\"cat_\")]\n",
    "final_mnar = [c for c in final_num if X_all[c].isnull().any()]\n",
    "\n",
    "print(f\"MNAR columns (with missings): {len(final_mnar)}\")\n",
    "print(f\"X_all shape: {X_all.shape}\")\n",
    "print(f\"Final numerical:   {len(final_num)}\")\n",
    "print(f\"Final categorical: {len(final_cat)}\")\n",
    "print(f\"Target balance: \"\n",
    "      f\"{pd.Series(y_all).value_counts(normalize=True).round(4).to_dict()}\")"
], cell_id="5ea08b69"))

# ─── 4. Code: Build StratifiedGroupKFold splits ─────────────────────────────
new_section.append(md_cell([
    "### Step 3 — Build Cross-Validation Splits with `StratifiedGroupKFold`\n",
    "\n",
    "With extreme class imbalance (~2.2% positives), a simple random split could easily produce folds where the minority class is under- or over-represented. `StratifiedGroupKFold` solves two problems simultaneously:\n",
    "\n",
    "1. **Stratification** — each fold preserves roughly the same positive-class rate as the full dataset.\n",
    "2. **Group constraint** — all rows sharing the same `id` are kept together in either train **or** validation, never split across both. This eliminates entity-level data leakage.\n",
    "\n",
    "The resulting `splits` list (a list of `(train_indices, val_indices)` tuples) is reused by every model pipeline below."
], cell_id="md_cv_splits"))

new_section.append(code_cell([
    "entity_target = df.groupby(ID_COL)[TARGET_COL].first().reset_index()\n",
    "entity_y = df[ID_COL].map(\n",
    "    entity_target.set_index(ID_COL)[TARGET_COL]\n",
    ").values\n",
    "\n",
    "sgkf = StratifiedGroupKFold(\n",
    "    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE\n",
    ")\n",
    "splits = list(sgkf.split(X_all, entity_y, groups=groups_all))\n",
    "\n",
    "print(f\"CV splits (StratifiedGroupKFold, {N_SPLITS} folds):\")\n",
    "for i, (tr, va) in enumerate(splits):\n",
    "    n_groups_tr = len(set(groups_all[tr]))\n",
    "    n_groups_va = len(set(groups_all[va]))\n",
    "    overlap = set(groups_all[tr]) & set(groups_all[va])\n",
    "    print(\n",
    "        f\"  Fold {i}: \"\n",
    "        f\"train={len(tr):,} rows ({n_groups_tr} entities), \"\n",
    "        f\"val={len(va):,} rows ({n_groups_va} entities), \"\n",
    "        f\"val positive rate={y_all[va].mean():.2%}, \"\n",
    "        f\"id overlap={len(overlap)}\"\n",
    "    )"
], cell_id="c7016378"))

# ─── 5. Shared helpers ──────────────────────────────────────────────────────
new_section.append(md_cell([
    "## Preprocessing Pipelines\n",
    "\n",
    "We define two distinct preprocessing pipelines — one for **Logistic Regression** and one for **Tree-Based Models** (CatBoost / LightGBM). The two model families have fundamentally different requirements:\n",
    "\n",
    "| Requirement | Logistic Regression | CatBoost / LightGBM |\n",
    "|---|---|---|\n",
    "| Missing values | **Must** be imputed (LR cannot handle `NaN`) | Handled natively — no imputation needed |\n",
    "| Feature scaling | **Required** — coefficients are sensitive to magnitude | Not needed — trees split on thresholds, scale-invariant |\n",
    "| Categorical encoding | Integer label encoding (after scaling) | Integer label encoding (CatBoost treats them specially) |\n",
    "| Missing-value indicators | Useful (MNAR pattern may be predictive) | Not needed (the tree can learn the split on missing vs. present) |\n",
    "\n",
    "Both pipelines share two common helpers — `sanitize_names` and `encode_categoricals` — described first.\n",
    "\n",
    "> **Data-leakage prevention rule:** every *learned* transform (median imputation, scaler fitting, label encoding vocabulary) is fitted **only on the training fold** and then applied to the validation fold. This mirrors a real production scenario where validation data is unseen."
], cell_id="md_pipelines_intro"))

# ─── 5a. sanitize_names ─────────────────────────────────────────────────────
new_section.append(md_cell([
    "### Helper — `sanitize_names`\n",
    "\n",
    "Some gradient-boosting libraries (LightGBM, CatBoost) raise errors when column names contain special characters such as brackets, spaces, or unicode symbols. This helper replaces every non-word character with an underscore."
], cell_id="md_sanitize_names"))

new_section.append(code_cell([
    "def sanitize_names(cols):\n",
    '    """Replace non-word characters in column names with underscores."""\n',
    '    return [re.sub(r"[^\\w]", "_", c) for c in cols]'
], cell_id="code_sanitize_names"))

# ─── 5b. encode_categoricals ────────────────────────────────────────────────
new_section.append(md_cell([
    "### Helper — `encode_categoricals` (leakage-safe)\n",
    "\n",
    "Label encoding maps every unique category string to a deterministic integer. To prevent data leakage:\n",
    "\n",
    "1. The set of unique values (the *vocabulary*) is extracted **from the training fold only**.\n",
    "2. The same mapping is applied to the validation fold. Any category that appears only in validation but not in training is mapped to `-1` (unknown).\n",
    "\n",
    "This ensures the model never gets information about the validation distribution during training."
], cell_id="md_encode_cat"))

new_section.append(code_cell([
    "def encode_categoricals(X_train, X_val, cat_cols):\n",
    '    """Label-encode categoricals. Vocabulary is fitted on train only."""\n',
    "    X_tr, X_va = X_train.copy(), X_val.copy()\n",
    "    for col in cat_cols:\n",
    "        if col not in X_tr.columns:\n",
    "            continue\n",
    '        uniques = X_tr[col].astype(str).fillna("__NA__").unique()\n',
    "        mapping = {v: i for i, v in enumerate(sorted(uniques))}\n",
    '        X_tr[col] = X_tr[col].astype(str).fillna("__NA__").map(mapping).fillna(-1).astype(int)\n',
    '        X_va[col] = X_va[col].astype(str).fillna("__NA__").map(mapping).fillna(-1).astype(int)\n',
    "    return X_tr, X_va"
], cell_id="code_encode_cat"))

# ─── 5c. prepare_for_logreg ─────────────────────────────────────────────────
new_section.append(md_cell([
    "### Pipeline A — `prepare_for_logreg`\n",
    "\n",
    "Logistic Regression is a **linear model** — it computes a weighted sum of features and passes the result through a sigmoid. This imposes strict requirements on the input data:\n",
    "\n",
    "1. **MNAR indicators** — for every numerical column that has missing values, we add a binary flag (`col__miss = 1` if the value is missing). In our dataset 319 out of 380 numerical features contain missings, so this step alone creates hundreds of extra features that capture the *pattern* of missingness — which can be highly predictive in banking / credit-scoring domains.\n",
    "2. **Label encoding** — categorical features are converted to integers (fitted on train, applied to val — see `encode_categoricals`).\n",
    "3. **Median imputation** — after creating the indicators, remaining `NaN` values are filled with the **training-fold median**. Using the training median (not the global median) prevents information leaking from validation.\n",
    "4. **RobustScaler** — unlike `StandardScaler`, `RobustScaler` uses the median and interquartile range instead of mean/std. This makes it far more resilient to outliers — very important given that our features have a median skewness of ~10.\n",
    "\n",
    "All four steps are fitted **on the training fold only** and then applied to both train and validation."
], cell_id="md_prepare_logreg"))

new_section.append(code_cell([
    "def prepare_for_logreg(X_train, X_val, mnar_cols, num_cols, cat_cols):\n",
    '    """\n',
    "    Full preprocessing pipeline for Logistic Regression:\n",
    "      1. Add binary is_missing indicators for MNAR columns\n",
    "      2. Label encode categoricals (fit on train)\n",
    "      3. Median imputation (fit on train)\n",
    "      4. RobustScaler (fit on train)\n",
    '    """\n',
    "    X_tr = X_train.copy()\n",
    "    X_va = X_val.copy()\n",
    "\n",
    "    for col in mnar_cols:\n",
    "        if col in X_tr.columns:\n",
    '            X_tr[f"{col}__miss"] = X_tr[col].isnull().astype(np.int8)\n',
    '            X_va[f"{col}__miss"] = X_va[col].isnull().astype(np.int8)\n',
    "\n",
    "    X_tr, X_va = encode_categoricals(X_tr, X_va, cat_cols)\n",
    "\n",
    "    medians = X_tr.median()\n",
    "    medians = medians.fillna(0)\n",
    "    X_tr = X_tr.fillna(medians)\n",
    "    X_va = X_va.fillna(medians)\n",
    "\n",
    "    all_cols = X_tr.columns.tolist()\n",
    "    scaler = RobustScaler()\n",
    "    X_tr_s = pd.DataFrame(\n",
    "        scaler.fit_transform(X_tr), columns=all_cols, index=X_tr.index\n",
    "    )\n",
    "    X_va_s = pd.DataFrame(\n",
    "        scaler.transform(X_va), columns=all_cols, index=X_va.index\n",
    "    )\n",
    "\n",
    "    return X_tr_s, X_va_s"
], cell_id="code_prepare_logreg"))

# ─── 5d. prepare_for_trees ──────────────────────────────────────────────────
new_section.append(md_cell([
    "### Pipeline B — `prepare_for_trees`\n",
    "\n",
    "Tree-based models (CatBoost, LightGBM, XGBoost) work by recursively splitting the feature space along individual features. This gives them several natural advantages:\n",
    "\n",
    "- **No imputation needed** — modern gradient-boosting libraries learn an optimal direction for missing values at every split, so `NaN`s can be left as-is.\n",
    "- **No scaling needed** — the split threshold is chosen by scanning sorted feature values, so the absolute magnitude of a feature is irrelevant.\n",
    "- **No MNAR indicators needed** — the tree can learn to split on \"missing vs. present\" internally.\n",
    "\n",
    "The only preprocessing step is **label encoding** of categorical columns (same leakage-safe procedure as for Logistic Regression)."
], cell_id="md_prepare_trees"))

new_section.append(code_cell([
    "def prepare_for_trees(X_train, X_val, cat_cols):\n",
    '    """\n',
    "    Minimal preprocessing for tree-based models:\n",
    "      - Label encode categoricals (fit on train)\n",
    "      - NaN values are kept as-is (CatBoost/LightGBM handle them natively)\n",
    "      - No scaling needed (trees are scale-invariant)\n",
    '    """\n',
    "    return encode_categoricals(X_train, X_val, cat_cols)"
], cell_id="code_prepare_trees"))

# ─── 5e. find_optimal_f1 ────────────────────────────────────────────────────
new_section.append(md_cell([
    "### Helper — `find_optimal_f1`\n",
    "\n",
    "After obtaining predicted probabilities, we need to pick a classification threshold. The default `0.5` is almost never optimal for highly imbalanced datasets. This helper performs a grid search over 200 threshold candidates between `0.001` and `0.999` and returns the one that maximises the F1-score."
], cell_id="md_find_f1"))

new_section.append(code_cell([
    "def find_optimal_f1(y_true, y_prob, n=200):\n",
    '    """Search for the probability threshold that maximises F1-score."""\n',
    "    thresholds = np.linspace(0.001, 0.999, n)\n",
    "    best_f1, best_t = 0.0, 0.5\n",
    "    for t in thresholds:\n",
    "        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)\n",
    "        if f1 > best_f1:\n",
    "            best_f1, best_t = f1, t\n",
    "    return best_t, best_f1\n",
    "\n",
    'print("Preprocessing functions defined.")'
], cell_id="code_find_f1"))

# ─── 6. LR feature selection cell ───────────────────────────────────────────
new_section.append(md_cell([
    "### Step 4 — Select Statistically Significant Features for Logistic Regression\n",
    "\n",
    "With 506 features and only ~700 positive samples, Logistic Regression is highly susceptible to overfitting. To mitigate this, we perform univariate feature selection *before* training:\n",
    "\n",
    "- **Numerical features** — keep only those whose distribution differs significantly between the positive and negative classes (Mann-Whitney U test, p < 0.05).\n",
    "- **Categorical features** — keep only those associated with the target (Chi-squared test, p < 0.05).\n",
    "\n",
    "This reduces the feature space substantially and removes noise features that could hurt a linear model. The filtered feature lists (`sig_num`, `sig_cat`, `sig_mnar`) and the corresponding subset `X_all_lr` are used exclusively by the Logistic Regression pipeline; the tree-based models continue to use the full `X_all` and `FINAL_FEATURES`."
], cell_id="md_lr_feat_sel"))

new_section.append(code_cell([
    "sig_num = significant[\"column\"].tolist()\n",
    "sig_cat = chi2_sig[\"column\"].tolist()\n",
    "sig_mnar = [\n",
    "    c for c in sig_num\n",
    "    if c in X_all.columns and X_all[c].isnull().any()\n",
    "]\n",
    "\n",
    'print("Features for LR:")\n',
    'print(f"  Numerical  (sig): {len(sig_num)}  (was {len(final_num)})")\n',
    'print(f"  Categorical(sig): {len(sig_cat)}  (was {len(final_cat)})")\n',
    'print(f"  MNAR indicators : {len(sig_mnar)}")\n',
    "\n",
    "sig_features = sig_num + sig_cat\n",
    "X_all_lr = X_all[sig_features].copy()"
], cell_id="33c8bfcc"))


# ── Splice the new section into the notebook ─────────────────────────────────
nb["cells"] = cells[:first_idx] + new_section + cells[last_idx + 1:]

# ── Write back ───────────────────────────────────────────────────────────────
with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook patched successfully. "
      f"Replaced {last_idx - first_idx + 1} cells with {len(new_section)} cells.")
