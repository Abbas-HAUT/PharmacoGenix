"""
pharmacogenix_pipeline.py
==============================================================================
End-to-end reproduction of the PharmacoGenix study pipeline described in the
manuscript methods, built directly on a preprocessed compound dataset that
already contains:

    molecule_chembl_id, canonical_smiles, standard_value, Activity Class,
    MW, LogP, NumHDonors, NumHAcceptors, standard_value_norm, pIC50_values

Pipeline stages (each maps to a section of the methods text):
  1. Load & validate the dataset.
  2. Outlier removal (IQR + Z-score) on the numeric descriptor/target columns.
  3. Mann-Whitney U hypothesis testing (active vs. inactive) on MW, LogP,
     NumHDonors, NumHAcceptors, standard_value_norm, pIC50_values, with
     Seaborn/Matplotlib visualizations of each comparison.
  4. Molecular fingerprint generation from canonical_smiles (RDKit Morgan/
     ECFP fingerprints -- the "fingerprint descriptors" used as ML input;
     see the note in generate_fingerprints() re: PaDEL, the tool literally
     named in the manuscript).
  5. Variance-threshold feature selection (threshold = 0.01, as specified).
  6. Random Forest Regressor: 80/20 split, randomized hyperparameter search
     over n_estimators / max_depth / min_samples_split, 5-fold cross-
     validation, and R2 / RMSE evaluation on the held-out test set.
  7. Persist the trained model + fingerprint config for the companion
     Streamlit app (pharmacogenix_app.py).

Run:
    python pharmacogenix_pipeline.py --data compounds.csv --outdir results
==============================================================================
"""
import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

try:
    import pingouin as pg
    HAVE_PINGOUIN = True
except ImportError:
    HAVE_PINGOUIN = False

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from rdkit import RDLogger

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import randint

import joblib

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

REQUIRED_COLUMNS = ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'Activity Class',
                    'MW', 'LogP', 'NumHDonors', 'NumHAcceptors', 'standard_value_norm', 'pIC50_values']
UTEST_FEATURES = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors', 'standard_value_norm', 'pIC50_values']


def parse_args():
    p = argparse.ArgumentParser(description="PharmacoGenix RF-regressor pipeline")
    p.add_argument('--data', type=str, required=True, help='Path to the compound CSV dataset.')
    p.add_argument('--outdir', type=str, default='results', help='Output directory for models/plots/tables.')
    p.add_argument('--radius', type=int, default=2, help='Morgan fingerprint radius (ECFP4 = radius 2).')
    p.add_argument('--n_bits', type=int, default=1024, help='Morgan fingerprint bit-vector length.')
    p.add_argument('--n_iter_search', type=int, default=25, help='RandomizedSearchCV iterations.')
    p.add_argument('--cv_folds', type=int, default=5, help='Cross-validation folds.')
    p.add_argument('--skip_tuning', action='store_true', help='Skip hyperparameter search (fast smoke test).')
    p.add_argument('--random_state', type=int, default=42)
    return p.parse_args()


# ==========================================================================
# 1. LOAD & VALIDATE
# ==========================================================================
def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {missing}\n"
                          f"Columns found: {list(df.columns)}")
    print(f"Loaded {len(df)} compounds from {path}")
    print(df[REQUIRED_COLUMNS].describe(include='all').transpose().to_string())
    return df


def normalize_class_label(v):
    """'Activity Class' values are mapped to {'active','inactive','intermediate'}
    by substring match, since exact label spelling/casing can vary by source."""
    v = str(v).strip().lower()
    if 'inactive' in v:
        return 'inactive'
    if 'active' in v:
        return 'active'
    return 'intermediate'


# ==========================================================================
# 2. OUTLIER REMOVAL (IQR + Z-score, as specified in the methods)
# ==========================================================================
def remove_outliers(df, cols, z_thresh=3.0, iqr_k=1.5):
    """
    Keeps a row only if it passes BOTH the IQR whisker test and the Z-score
    test for every column in `cols`. This directly implements the manuscript's
    "IQR method and Z-score analysis" outlier removal step.
    """
    mask = pd.Series(True, index=df.index)
    for col in cols:
        x = df[col].astype(float)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        iqr_ok = x.between(lo, hi)
        z = (x - x.mean()) / (x.std(ddof=0) + 1e-12)
        z_ok = z.abs() <= z_thresh
        mask &= (iqr_ok & z_ok)
    n_removed = (~mask).sum()
    print(f"Outlier removal (IQR + Z-score) on {cols}: removed {n_removed} of {len(df)} rows.")
    return df[mask].reset_index(drop=True)


# ==========================================================================
# 3. MANN-WHITNEY U TESTING (active vs. inactive)
# ==========================================================================
def run_mannwhitney_tests(df, features, class_col='Activity Class', alpha=0.05):
    df = df.copy()
    df['_class'] = df[class_col].apply(normalize_class_label)
    active = df[df['_class'] == 'active']
    inactive = df[df['_class'] == 'inactive']
    print(f"\nMann-Whitney U test group sizes -- active: {len(active)}, inactive: {len(inactive)}")

    rows = []
    for feat in features:
        x, y = active[feat].dropna(), inactive[feat].dropna()
        if HAVE_PINGOUIN:
            res = pg.mwu(x, y, alternative='two-sided')
            u_stat, p_val = res['U-val'].iloc[0], res['p-val'].iloc[0]
            rbc = res['RBC'].iloc[0] if 'RBC' in res.columns else np.nan
        else:
            u_stat, p_val = stats.mannwhitneyu(x, y, alternative='two-sided')
            rbc = np.nan
        rows.append({'Feature': feat, 'U_statistic': u_stat, 'p_value': p_val,
                     'rank_biserial_corr': rbc, 'significant_at_0.05': p_val < alpha})
    return pd.DataFrame(rows), df


def plot_mannwhitney_features(df_with_class, features, results_df, out_dir):
    n = len(features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    plot_df = df_with_class[df_with_class['_class'].isin(['active', 'inactive'])]
    for i, feat in enumerate(features):
        ax = axes[i]
        sns.boxplot(data=plot_df, x='_class', y=feat, order=['inactive', 'active'],
                    palette={'inactive': '#8C8C8C', 'active': '#0072B2'}, ax=ax, width=0.5)
        sns.stripplot(data=plot_df, x='_class', y=feat, order=['inactive', 'active'],
                      color='black', alpha=0.25, size=2.5, ax=ax)
        p_val = results_df.loc[results_df['Feature'] == feat, 'p_value'].iloc[0]
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.set_title(f'{feat}\nMann-Whitney U, p = {p_val:.2e} ({sig})', fontsize=10)
        ax.set_xlabel('')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'mannwhitney_feature_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ==========================================================================
# 4. FINGERPRINT DESCRIPTOR GENERATION
# ==========================================================================
def generate_fingerprints(smiles_series, radius=2, n_bits=1024):
    """
    RDKit Morgan (ECFP-equivalent) fingerprints are used here as the binary
    "fingerprint descriptors" described in the manuscript.

    Note on PaDEL: the manuscript names the "Paddle" library, almost
    certainly PaDEL-Descriptor -- a Java-based tool wrapped in Python via
    `padelpy`. It computes a similar (and in the original study, PubChem-
    style) fingerprint set, but requires a local Java Runtime Environment
    and the PaDEL jar/XML descriptor files, which makes it far less portable
    than a pure-Python dependency. RDKit's Morgan fingerprints are the
    standard, dependency-light substitute and are used here so this script
    runs with `pip install rdkit` alone. If you specifically need PaDEL
    fingerprints (e.g. to match the original study bit-for-bit), install
    `padelpy` and Java, then replace this function with a call to
    `padelpy.from_smiles(smiles, fingerprints=True)`.
    """
    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_series):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(np.array(fp, dtype=np.int8))
        valid_idx.append(i)
    X = np.array(fps, dtype=np.int8)
    return X, valid_idx


# ==========================================================================
# 5-6. ML DATASET PREP, FEATURE SELECTION, RF TRAINING
# ==========================================================================
def prepare_ml_dataset(df, args):
    """
    Restricts to clearly active/inactive compounds (intermediate-potency
    compounds are excluded, as specified), computes fingerprints from
    canonical_smiles, and aligns them with the pIC50 regression target.
    """
    df = df.copy()
    df['_class'] = df['Activity Class'].apply(normalize_class_label)
    ml_df = df[df['_class'].isin(['active', 'inactive'])].reset_index(drop=True)
    print(f"\nCompounds retained for ML (active + inactive only): {len(ml_df)} of {len(df)}")

    X, valid_idx = generate_fingerprints(ml_df['canonical_smiles'], radius=args.radius, n_bits=args.n_bits)
    ml_df = ml_df.iloc[valid_idx].reset_index(drop=True)
    y = ml_df['pIC50_values'].values.astype(float)

    n_dropped = len(ml_df) - len(y)  # kept for symmetry/clarity; should be 0 here
    print(f"Fingerprints generated for {X.shape[0]} compounds ({X.shape[1]}-bit Morgan/ECFP fingerprints).")
    return X, y, ml_df


def select_features(X_train, X_test, threshold=0.01):
    """Variance thresholding, as specified (drops near-constant fingerprint bits)."""
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    selector = VarianceThreshold(threshold=threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel = selector.transform(X_test)
    print(f"Variance thresholding (>= {threshold}): {X_train.shape[1]} -> {X_train_sel.shape[1]} features")
    return X_train_sel, X_test_sel, selector, imputer


def train_rf_model(X, y, args):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.random_state)

    X_train_sel, X_test_sel, selector, imputer = select_features(X_train, X_test, threshold=0.01)

    base_model = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)

    if args.skip_tuning:
        print("\n--skip_tuning set: fitting a default RandomForestRegressor (no search).")
        best_model = base_model.fit(X_train_sel, y_train)
    else:
        param_dist = {
            'n_estimators': randint(50, 500),
            'max_depth': [None, 5, 10, 15, 20, 30],
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5),
        }
        search = RandomizedSearchCV(
            base_model, param_distributions=param_dist, n_iter=args.n_iter_search,
            cv=args.cv_folds, scoring='r2', random_state=args.random_state, n_jobs=-1, verbose=1)
        print(f"\nRunning RandomizedSearchCV ({args.n_iter_search} iterations, {args.cv_folds}-fold CV)...")
        search.fit(X_train_sel, y_train)
        best_model = search.best_estimator_
        print(f"Best hyperparameters: {search.best_params_}")

    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    cv_r2 = cross_val_score(best_model, X_train_sel, y_train, cv=cv, scoring='r2', n_jobs=-1)
    cv_rmse = -cross_val_score(best_model, X_train_sel, y_train, cv=cv,
                                scoring='neg_root_mean_squared_error', n_jobs=-1)

    y_pred = best_model.predict(X_test_sel)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nCross-validation ({args.cv_folds}-fold, training set): "
          f"R2 = {cv_r2.mean():.4f} +/- {cv_r2.std():.4f} | "
          f"RMSE = {cv_rmse.mean():.4f} +/- {cv_rmse.std():.4f}")
    print(f"Held-out test set: R2 = {test_r2:.4f} | RMSE = {test_rmse:.4f}")

    return {
        'model': best_model, 'selector': selector, 'imputer': imputer,
        'y_test': y_test, 'y_pred': y_pred,
        'cv_r2_mean': cv_r2.mean(), 'cv_r2_std': cv_r2.std(),
        'cv_rmse_mean': cv_rmse.mean(), 'cv_rmse_std': cv_rmse.std(),
        'test_r2': test_r2, 'test_rmse': test_rmse,
    }


def plot_prediction_performance(result, out_dir):
    y_test, y_pred = result['y_test'], result['y_pred']
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolor='black', linewidth=0.3, s=35, color='#0072B2')
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], '--', color='#333333', linewidth=1.4, label='y = x')
    ax.set_xlabel('Observed pIC50')
    ax.set_ylabel('Predicted pIC50')
    ax.set_title(f"RF regressor: predicted vs. observed pIC50\n"
                 f"Test R\u00b2 = {result['test_r2']:.3f}, RMSE = {result['test_rmse']:.3f}")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'rf_pIC50_predicted_vs_observed.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ==========================================================================
# 7. SAVE ARTIFACTS FOR THE STREAMLIT APP
# ==========================================================================
def save_artifacts(result, args, out_dir):
    model_path = os.path.join(out_dir, 'rf_pIC50_model.pkl')
    selector_path = os.path.join(out_dir, 'variance_selector.pkl')
    imputer_path = os.path.join(out_dir, 'imputer.pkl')
    config_path = os.path.join(out_dir, 'fingerprint_config.json')

    joblib.dump(result['model'], model_path)
    joblib.dump(result['selector'], selector_path)
    joblib.dump(result['imputer'], imputer_path)
    with open(config_path, 'w') as f:
        json.dump({'radius': args.radius, 'n_bits': args.n_bits}, f, indent=2)

    print(f"\nSaved model to:      {model_path}")
    print(f"Saved selector to:   {selector_path}")
    print(f"Saved imputer to:    {imputer_path}")
    print(f"Saved fp config to:  {config_path}")
    print("These four files are exactly what pharmacogenix_app.py expects to find.")


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.data)

    df_clean = remove_outliers(df, ['MW', 'LogP', 'standard_value_norm', 'pIC50_values'])

    print("\nRunning Mann-Whitney U tests (active vs. inactive)...")
    mwu_results, df_with_class = run_mannwhitney_tests(df_clean, UTEST_FEATURES)
    print(mwu_results.round(4).to_string(index=False))
    mwu_results.to_csv(os.path.join(args.outdir, 'mannwhitney_results.csv'), index=False)
    plot_mannwhitney_features(df_with_class, UTEST_FEATURES, mwu_results, args.outdir)

    print("\nPreparing fingerprint-based ML dataset...")
    X, y, ml_df = prepare_ml_dataset(df_clean, args)

    print("\nTraining Random Forest Regressor...")
    result = train_rf_model(X, y, args)
    plot_prediction_performance(result, args.outdir)

    save_artifacts(result, args, args.outdir)

    summary = pd.DataFrame([{
        'n_compounds_total': len(df), 'n_compounds_after_outlier_removal': len(df_clean),
        'n_compounds_ml': len(ml_df), 'fingerprint_bits': args.n_bits, 'fingerprint_radius': args.radius,
        'cv_r2_mean': result['cv_r2_mean'], 'cv_r2_std': result['cv_r2_std'],
        'cv_rmse_mean': result['cv_rmse_mean'], 'cv_rmse_std': result['cv_rmse_std'],
        'test_r2': result['test_r2'], 'test_rmse': result['test_rmse'],
    }])
    summary.to_csv(os.path.join(args.outdir, 'run_summary.csv'), index=False)
    print(f"\nSaved run summary to: {os.path.join(args.outdir, 'run_summary.csv')}")
    print("\nDone.")


if __name__ == '__main__':
    main()
