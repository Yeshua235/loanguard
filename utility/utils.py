import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.linalg import svdvals
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from pathlib import Path
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, precision_recall_curve


# Report

def recall_at_precision(model, data: pd.DataFrame, target: pd.DataFrame | np.ndarray, precision: float) ->  float:
    prec, recall, _ = precision_recall_curve(target, model.predict_proba(data)[:, 1])
    idx = np.argmin(np.abs(prec - precision))
    return recall[idx]

def optimal_threshold(model, data: pd.DataFrame, target: pd.DataFrame | np.ndarray) ->  float:
    precision, recall, threshold = precision_recall_curve(target, model.predict_proba(data)[:, 1])
    best_idx = np.argmax(recall + precision)
    return threshold[best_idx]

# Model Selection Utilities

def evaluate_model(model, data: pd.DataFrame, target: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
    """
    Cross-validated metrics for a classifier (expects model to be an estimator or Pipeline).
    Important: to avoid leakage, pass a Pipeline that includes preprocessing.
    """
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=37)
    scores = cross_validate(
        estimator=model,
        X=data, y=target,
        cv=cv, scoring={"f1": "f1", "pr_auc": "average_precision"},
        n_jobs=-1, error_score="raise",
        return_train_score=False,
    )

    return {
        "f1-score": scores["test_f1"],
        "pr_auc": scores["test_pr_auc"],
    }

model_selection_fig_dir = Path(__file__).resolve().parent.parent/"reports"/"model_selection"
model_selection_fig_dir.mkdir(parents=True, exist_ok=True)

def plot_roc_curve(model, data: pd.DataFrame, target: pd.DataFrame | np.ndarray, curve_label: str):
    """
    Fits `model` and returns a PrecisionRecallDisplay.
    Uses response_method='auto' so it works for estimators with predict_proba OR decision_function.
    """
    fitted_model = model.fit(data, target)

    return RocCurveDisplay.from_estimator(
        estimator=fitted_model,
        X=data, y=target,
        pos_label=1, response_method='auto',
        name=curve_label, plot_chance_level=True,
        chance_level_kw={'label':'Chance Level'}
    )


def plot_pr_curve(model, data: pd.DataFrame, target: pd.DataFrame | np.ndarray, curve_label: str):
    """
    Fits `model` and returns a PrecisionRecallDisplay.
    Uses response_method='auto' so it works for estimators with predict_proba OR decision_function.
    """
    fitted_model = model.fit(data, target)

    return PrecisionRecallDisplay.from_estimator(
        estimator=fitted_model,
        X=data, y=target,
        pos_label=1, response_method='auto',
        name=curve_label, plot_chance_level=True,
        chance_level_kw={'label':'Chance Level'}
    )

def cross_val_scores_plot(scores: list[pd.DataFrame], names: list[str], choice_column: str):
    """
    Build a boxplot comparing CV score distributions across models.
    `scores` is a list of DataFrames (each with columns like 'f1-score', 'pr_auc').
    Returns matplotlib Axes.
    """
    if len(scores) != len(names):
        raise ValueError("scores and names must have the same length")

    series_list = []
    for df, name in zip(scores, names):
        if choice_column not in df.columns:
            raise KeyError(f"{choice_column=} not found in DataFrame columns: {list(df.columns)}")
        series_list.append(df[choice_column].rename(name))

    plot_frame = pd.concat(series_list, axis=1)
    ax = plot_frame.boxplot(grid=True)
    ax.set_ylabel(choice_column)
    return ax


# Feature Engineering Utilities

class RepCompare:
    """
    Compare high-D processed data vs. PCA-reduced data (same samples).
    Metrics: RV coefficient, pairwise distance correlation, and reconstruction R2.
    Also provides visualizations optimized via subsampling.
    """
    def __init__(self, random_state: int = 37, dtype: str = 'float32'):
        self.random_state = random_state
        self.dtype = dtype
        self.output_dir = Path(__file__).resolve().parent.parent/"reports"/"feature_engineering"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------
    def _as_array(self, X):
        return np.asarray(X, dtype=self.dtype)

    def _center(self, X):
        return X - np.mean(X, axis=0, keepdims=True)

    def _sample_idx(self, n: int, size: int | None):
        if size is None or size >= n:
            return np.arange(n)
        rng = np.random.default_rng(self.random_state)
        return rng.choice(n, size=size, replace=False)

    def _get_pca(self, pca_or_pipeline):
        # Accept PCA instance or a pipeline containing PCA
        if hasattr(pca_or_pipeline, 'inverse_transform') and hasattr(pca_or_pipeline, 'explained_variance_ratio_'):
            return pca_or_pipeline
        if hasattr(pca_or_pipeline, 'named_steps'):
            for step in pca_or_pipeline.named_steps.values():
                if isinstance(step, PCA):
                    return step
        if hasattr(pca_or_pipeline, 'steps'):
            for _, step in pca_or_pipeline.steps:
                if isinstance(step, PCA):
                    return step
        raise ValueError("PCA transformer not found. Provide a fitted PCA or a pipeline that contains PCA.")

    # ---------- metrics ----------
    def rv_coefficient(self, X, Z):
        """
        RV coefficient between two matrices (same rows, any columns).
        """
        Xc = self._center(self._as_array(X))
        Zc = self._center(self._as_array(Z))
        Sxz = Xc.T @ Zc
        Sxx = Xc.T @ Xc
        Szz = Zc.T @ Zc
        num = np.trace(Sxz @ Sxz.T)
        den = np.sqrt(np.trace(Sxx @ Sxx) * np.trace(Szz @ Szz) + 1e-12)
        return float(num / den)

    def distance_matrix_corr(self, X, Z, metric: str = 'euclidean', sample_size: int = 5000, method: str = 'spearman'):
        """
        Correlate sample-to-sample distances in processed vs. PCA space.
        Uses subsampling for speed/memory.
        """
        X = self._as_array(X)
        Z = self._as_array(Z)
        n = min(X.shape[0], Z.shape[0])
        idx = self._sample_idx(n, sample_size)
        DX = pairwise_distances(X[idx], metric=metric)
        DZ = pairwise_distances(Z[idx], metric=metric)
        iu = np.triu_indices(len(idx), k=1)
        x = DX[iu]
        y = DZ[iu]
        if method == 'spearman':
            r, _ = spearmanr(x, y)
        else:
            r = np.corrcoef(x, y)[0, 1]
        return float(r)

    def reconstruction_r2(self, X, pca_or_pipeline):
        """
        R2 of reconstructing processed X from its PCA projection.
        """
        X = self._as_array(X)
        pca = self._get_pca(pca_or_pipeline)
        Z = pca.transform(X)
        Xhat = pca.inverse_transform(Z)
        R = X - Xhat
        num = np.sum(R * R)
        den = np.sum(X * X) + 1e-12
        return float(1.0 - num / den)

    def evaluate(self, X_processed, Z_pca, pca_or_pipeline, sample_size: int = 5000):
        """
        Compute all metrics and return a dict.
        """
        return {
            'RV': self.rv_coefficient(X_processed, Z_pca),
            'DistanceCorr': self.distance_matrix_corr(X_processed, Z_pca, sample_size=sample_size),
            'ReconstructionR2': self.reconstruction_r2(X_processed, pca_or_pipeline),
        }

    # ---------- visualizations ----------
    def plot_distance_preservation(self, X, Z, sample_size: int = 3000, metric: str = 'euclidean', method: str = 'spearman', title_prefix: str = ""):
        """
        Hexbin of pairwise distances: processed vs. PCA space, with y=x line.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        X = self._as_array(X)
        Z = self._as_array(Z)
        n = min(X.shape[0], Z.shape[0])
        idx = self._sample_idx(n, sample_size)
        DX = pairwise_distances(X[idx], metric=metric)
        DZ = pairwise_distances(Z[idx], metric=metric)
        iu = np.triu_indices(len(idx), k=1)
        x = DX[iu]
        y = DZ[iu]

        # compute correlation directly from the distance vectors
        if method == 'spearman':
            r, _ = spearmanr(x, y)
        else:
            r = np.corrcoef(x, y)[0, 1]

        fig, ax = plt.subplots(figsize=(8, 6))
        hb = ax.hexbin(x, y, gridsize=60, cmap='viridis', mincnt=1)
        ax.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', linewidth=1, label='y = x')
        ax.set_xlabel('Processed distances')
        ax.set_ylabel('PCA distances')
        ax.set_title(f"{title_prefix}Distance preservation (corr={r:.3f})")
        ax.legend(loc='lower right')
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('count')
        fig.tight_layout()
        plt.savefig(f"{self.output_dir}/distance-preservation.png")
        plt.show()

    def plot_reconstruction_residuals(self, X, pca_or_pipeline, sample_size: int = 20000, title_prefix: str = ""):
        """
        Histogram of per-sample reconstruction MSE, annotated with R2.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        X = self._as_array(X)
        idx = self._sample_idx(X.shape[0], sample_size)
        pca = self._get_pca(pca_or_pipeline)
        Z = pca.transform(X[idx])
        Xhat = pca.inverse_transform(Z)
        mse_per_sample = np.mean((X[idx] - Xhat) ** 2, axis=1)
        r2 = self.reconstruction_r2(X, pca)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(mse_per_sample, bins=50, color='steelblue', alpha=0.9)
        ax.set_xlabel('Per-sample reconstruction MSE')
        ax.set_ylabel('Count')
        ax.set_title(f"{title_prefix}Reconstruction error (R2={r2:.4f})")
        fig.tight_layout()
        plt.savefig(f"{self.output_dir}/reconstruction-error-r2.png")
        plt.show()

    def plot_explained_variance(self, pca_or_pipeline, title_prefix: str = ""):
        """
        Cumulative explained variance for PCA components.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        pca = self._get_pca(pca_or_pipeline)
        evr = np.asarray(pca.explained_variance_ratio_, dtype=self.dtype)
        cum = np.cumsum(evr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(np.arange(1, len(evr) + 1), cum, marker='o')
        ax.set_xlabel('Components')
        ax.set_ylabel('Cumulative explained variance')
        ax.set_title(f"{title_prefix}PCA cumulative explained variance")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(f"{self.output_dir}/pca-cumulative-explained-variance.png")
        plt.show()


class DRMetrics:
    def __init__(
        self,
        dims_percent=None,
        n_neighbors=12,
        random_state=37,
        isomap_fit_sample=20000,   # subsample for Isomap fit
        eval_sample=50000,         # subsample for evaluation metrics
        ridge_alpha=1e-6,
        pca_solver='randomized',   # fast solver for tall matrices
        dtype='float32'            # reduce memory footprint
    ) -> None:
        """
        Fast comparison of PCA and Isomap reconstructions on preprocessed data.
        """
        self.METRICS = ['Frobenius', 'Spectral', 'MSE', 'R2']
        self.TARGET_DIMS_PERCENT = dims_percent or [0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.isomap_fit_sample = isomap_fit_sample
        self.eval_sample = eval_sample
        self.ridge_alpha = ridge_alpha
        self.pca_solver = pca_solver
        self.dtype = dtype

    # Metrics
    def frobenius_norm(self, R): return norm(R, 'fro')
    def spectral_norm(self, R): return svdvals(R)[0]
    def mse(self, R): return np.mean(R**2)
    def reconstruction_R2(self, R, X): return 1.0 - (self.frobenius_norm(R)**2) / (norm(X, 'fro')**2 + 1e-12)

    # Helpers
    def _dims_from_percent(self, n_features: int):
        dims = {max(1, min(n_features - 1, round(p * n_features))) for p in self.TARGET_DIMS_PERCENT}
        return sorted(dims)

    def _maybe_subsample(self, X, size):
        if size is None or size >= X.shape[0]: return X
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(X.shape[0], size=size, replace=False)
        return X[idx]

    # PCA (fit once at max k, reuse components)
    def _pca_recon_all_k(self, X_eval, dims):
        max_k = max(dims)
        pca = PCA(n_components=max_k, svd_solver=self.pca_solver, random_state=self.random_state)
        Z_full = pca.fit_transform(X_eval)                   # [n_eval x max_k]
        comps = pca.components_                              # [max_k x p]
        mu = pca.mean_                                       # [p]
        recons = {}
        for k in dims:
            Xhat_k = Z_full[:, :k] @ comps[:k, :] + mu
            recons[k] = Xhat_k
        return recons

    # Isomap + linear decoder (fit per k on a smaller fit set)
    def _isomap_recon(self, X_eval, k):
        X_fit = self._maybe_subsample(X_eval, self.isomap_fit_sample)
        iso = Isomap(n_neighbors=self.n_neighbors, n_components=k)
        iso.fit(X_fit)
        Z_fit = iso.transform(X_fit)                         # [n_fit x k]
        G = Z_fit.T @ Z_fit                                  # [k x k]
        B = Z_fit.T @ X_fit                                  # [k x p]
        A = np.linalg.solve(G + self.ridge_alpha * np.eye(k), B)  # [k x p]
        Z_eval = iso.transform(X_eval)                       # [n_eval x k]
        return Z_eval @ A                                    # [n_eval x p]

    # Evaluation (fast via subsampling)
    def evaluate_methods(self, X: pd.DataFrame | np.ndarray, dims: list[int] | None = None):
        """
        Evaluate PCA and Isomap across dimensions on a subsampled evaluation set.
        X must be preprocessed (e.g., RobustScaler).
        """
        X = np.asarray(X, dtype=self.dtype)
        X_eval = self._maybe_subsample(X, self.eval_sample)
        n_features = X_eval.shape[1]
        dims = dims or self._dims_from_percent(n_features)
        dims = [d for d in dims if 1 <= d < n_features]

        rows_scores, rows_errors = [], []

        # PCA once, all k
        try:
            pca_recons = self._pca_recon_all_k(X_eval, dims)
            for k in dims:
                Xhat = pca_recons[k]
                R = X_eval - Xhat
                rows_scores.append({
                    'Method': 'PCA', 'Dim': k,
                    'Frobenius': self.frobenius_norm(R),
                    'Spectral': self.spectral_norm(R),
                    'MSE': self.mse(R),
                    'R2': self.reconstruction_R2(R, X_eval)
                })
        except Exception as e:
            rows_errors.append({'Method': 'PCA', 'Dim': None, 'Error': str(e)})

        # Isomap per k (on smaller fit set)
        for k in dims:
            try:
                Xhat = self._isomap_recon(X_eval, k)
                R = X_eval - Xhat
                rows_scores.append({
                    'Method': 'Isomap', 'Dim': k,
                    'Frobenius': self.frobenius_norm(R),
                    'Spectral': self.spectral_norm(R),
                    'MSE': self.mse(R),
                    'R2': self.reconstruction_R2(R, X_eval)
                })
            except Exception as e:
                rows_errors.append({'Method': 'Isomap', 'Dim': k, 'Error': str(e)})

        df_scores = pd.DataFrame(rows_scores)
        df_errors = pd.DataFrame(rows_errors)
        return df_scores, df_errors

    # Selection helpers
    def select_best(self, df_scores: pd.DataFrame, metric: str = 'R2', tol: float = 0.01, prefer_lower_dim: bool = True):
        if df_scores.empty or metric not in df_scores.columns: return None
        maximize = (metric == 'R2')
        best = df_scores[metric].max() if maximize else df_scores[metric].min()
        mask = df_scores[metric] >= (1 - tol) * best if maximize else df_scores[metric] <= (1 + tol) * best
        candidates = df_scores[mask].copy()
        sort_cols = ['Dim'] if prefer_lower_dim else [metric]
        return candidates.sort_values(sort_cols, ascending=True).iloc[0].to_dict()

    def min_dim_for_target(self, df_scores: pd.DataFrame, metric: str = 'R2', target: float = 0.95):
        if df_scores.empty or metric not in df_scores.columns: return pd.DataFrame()
        rows = []
        for meth, grp in df_scores.groupby('Method'):
            hit = grp.loc[grp[metric] >= target].sort_values('Dim').head(1)
            if not hit.empty:
                r = hit.iloc[0]
                rows.append({'Method': meth, 'Dim': int(r['Dim']), metric: float(r[metric])})
        return pd.DataFrame(rows)

    # Plotting
    def plot_metric_trends_df(self, df_scores, title_prefix=""):
        if df_scores.empty:
            print("No successful results to plot.")
            return
        plt.style.use('seaborn-v0_8-whitegrid')
        output_dir = Path(__file__).resolve().parent.parent/"reports"/"feature_engineering"
        output_dir.mkdir(parents=True, exist_ok=True)

        for m in self.METRICS:
            fig, ax = plt.subplots(figsize=(10, 6))
            for meth, grp in df_scores.groupby('Method'):
                grp_sorted = grp.sort_values('Dim')
                ax.plot(grp_sorted['Dim'], grp_sorted[m], marker='o', linewidth=2, markersize=6, label=meth)
                idx = grp_sorted[m].idxmax() if m == 'R2' else grp_sorted[m].idxmin()
                if pd.notna(idx):
                    r = df_scores.loc[idx]
                    ax.scatter(r['Dim'], r[m], s=60, edgecolor='k', facecolor='none')
                    ax.annotate(f"best {meth}: k={int(r['Dim'])}", (r['Dim'], r[m]),
                                textcoords="offset points", xytext=(6, -6), fontsize=9)
            ax.set_xlabel('Reduced dimension (k)')
            ax.set_ylabel('Reconstruction R2' if m == 'R2' else m)
            ax.set_title(f"{title_prefix}{m} vs. dimensionality")
            ax.grid(True, alpha=0.3)
            if m in ('Frobenius', 'Spectral', 'MSE'):
                ymin, ymax = ax.get_ylim()
                if ymin > 0 and ymax / max(ymin, 1e-12) > 10:
                    ax.set_yscale('log')
            ax.legend(ncol=2, fontsize=9, frameon=True)
            fig.tight_layout()
            plt.savefig(f"{output_dir}/{m}-vs-dimensionality.png")
            plt.show()


# EDA Utilities

def data_distribution_frame(data: pd.DataFrame) -> pd.DataFrame:
    '''
    function for inspecting the data distribution of the target classes

    :param data: The target matrix
    :type data: pd.DataFrame
    :return: A dataframe showing the value counts for each class, and its percent of the total target column
    :rtype: DataFrame
    '''
    result_frame =  pd.concat(
                            (pd.Series(data.value_counts(), name='value counts'),
                            pd.Series([f'{round(val/data.shape[0]*100)}%' for val in data.value_counts()], name='percent of total')),
    axis=1
                    )
    return result_frame


def columns_with_null(data: pd.DataFrame, option:str='') -> pd.DataFrame:
    '''
    Function for displaying columns containing null values in a dataframe in a more informative manner.

    :param data: The Data Matrix
    :type data: pd.DataFrame
    :param option: column dtype to return. Numeric columns [option="numeric"], Non-numeric columns [option="non-numeric] or both (default).
    :return: A DataFrame showing columns with null values, their datatype, the amount of nulls present in count and in percentage of the total
    :rtype: DataFrame
    '''
    cols = data.columns

    float_bad_cols = {}
    object_bad_cols = {}

    for col in cols:
        nulls = int(data[col].isnull().sum())
        if nulls:
            if is_numeric_dtype(data[col]):
                float_bad_cols[col] = nulls
            else:
                object_bad_cols[col] = nulls

    num_bad_cols = [(col, data[col].dtype, round(float_bad_cols[col]/data.shape[0] * 100), float_bad_cols[col]) for col  in float_bad_cols.keys()]
    str_bad_cols = [(col, data[col].dtype, round(object_bad_cols[col]/data.shape[0] * 100), object_bad_cols[col]) for col in object_bad_cols.keys()]

    num_bad_frame = pd.DataFrame(num_bad_cols, columns=['column name', 'column dtype', 'percent of total', 'number of nulls'])
    str_bad_frame = pd.DataFrame(str_bad_cols, columns=['column name', 'column dtype', 'percent of total', 'number of nulls'])

    if option == 'numeric':
        return num_bad_frame
    elif option == 'non-numeric':
        return str_bad_frame
    else:
        null_col_frames = pd.concat((num_bad_frame, str_bad_frame), axis=0)
        return null_col_frames
