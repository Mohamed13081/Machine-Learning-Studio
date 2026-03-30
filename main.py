from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io, json, traceback
from typing import Optional, List

# ── Sklearn imports ──────────────────────────────────────────
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
# Supervised
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.svm import SVC
# Unsupervised
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

app = FastAPI(title="ML Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory dataset store ──────────────────────────────────
dataset_store = {}

# ── Models ───────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    dataset_id: str
    method: str          # supervised | unsupervised
    algorithms: List[str]
    target_col: Optional[str] = None
    test_size: float = 0.2
    knn_k: int = 5
    n_clusters: int = 3
    n_components: int = 2
    n_estimators: int = 100
    eps: float = 0.5
    min_samples: int = 5

# ── Upload endpoint ───────────────────────────────────────────
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception:
        raise HTTPException(400, "Invalid CSV file")

    import uuid
    did = str(uuid.uuid4())[:8]
    dataset_store[did] = df

    # Stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = df.select_dtypes(exclude=[np.number]).columns.tolist()
    preview      = df.head(10).fillna("").astype(str).to_dict(orient="records")

    return {
        "dataset_id": did,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": df.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "preview": preview,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict(),
        "describe": df.describe().fillna(0).to_dict()
    }

# ── Analyze endpoint ──────────────────────────────────────────
@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if req.dataset_id not in dataset_store:
        raise HTTPException(404, "Dataset not found — upload CSV first")

    df = dataset_store[req.dataset_id].copy()

    try:
        if req.method == "supervised":
            return run_supervised(df, req)
        else:
            return run_unsupervised(df, req)
    except Exception:
        raise HTTPException(500, traceback.format_exc())


# ════════════════════════════════════════════════════════════
# SUPERVISED
# ════════════════════════════════════════════════════════════
def run_supervised(df: pd.DataFrame, req: AnalyzeRequest):
    if not req.target_col or req.target_col not in df.columns:
        raise HTTPException(400, "target_col is required for supervised learning")

    target = req.target_col
    feature_cols = [c for c in df.columns if c != target]

    # Encode / clean
    df = df.dropna()
    X = pd.get_dummies(df[feature_cols])
    y_raw = df[target]

    # Decide task: classification or regression
    is_regression = y_raw.dtype in [np.float64, np.float32] and y_raw.nunique() > 20
    if is_regression:
        y = y_raw.values.astype(float)
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=req.test_size, random_state=42
    )

    results = {}
    for algo_id in req.algorithms:
        results[algo_id] = run_single_supervised(
            algo_id, X_train, X_test, y_train, y_test,
            X_scaled, y, X.columns.tolist(), req, is_regression
        )

    return {
        "method": "supervised",
        "task": "regression" if is_regression else "classification",
        "features": X.columns.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "results": results
    }


def run_single_supervised(algo_id, X_train, X_test, y_train, y_test,
                           X_all, y_all, feature_names, req, is_regression):
    model = get_supervised_model(algo_id, req, is_regression)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    out = {"algo": algo_id}

    if is_regression:
        out["metrics"] = {
            "r2":   round(r2_score(y_test, y_pred), 4),
            "mse":  round(mean_squared_error(y_test, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "mae":  round(mean_absolute_error(y_test, y_pred), 4),
        }
        out["scatter"] = {
            "actual":    y_test[:60].tolist(),
            "predicted": y_pred[:60].tolist()
        }
        residuals = (y_test - y_pred).tolist()
        out["residuals"] = residuals[:200]
    else:
        avg = "binary" if len(np.unique(y_all)) == 2 else "weighted"
        out["metrics"] = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
        }
        cm = confusion_matrix(y_test, y_pred)
        out["confusion_matrix"] = cm.tolist()

        # ROC (binary only)
        if len(np.unique(y_all)) == 2 and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            out["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(auc(fpr, tpr), 4)}

    # Feature importance
    fi = get_feature_importance(model, feature_names)
    if fi:
        out["feature_importance"] = fi

    # Learning curve
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            get_supervised_model(algo_id, req, is_regression),
            X_all, y_all, cv=3,
            train_sizes=np.linspace(0.1, 1.0, 8),
            scoring="r2" if is_regression else "accuracy",
            n_jobs=-1
        )
        out["learning_curve"] = {
            "train_sizes": train_sizes.tolist(),
            "train_scores": train_scores.mean(axis=1).tolist(),
            "val_scores":   val_scores.mean(axis=1).tolist(),
        }
    except Exception:
        pass

    return out


def get_supervised_model(algo_id, req, is_regression):
    MAP = {
        "logistic": LogisticRegression(max_iter=1000),
        "knn":      KNeighborsClassifier(n_neighbors=req.knn_k),
        "dtree":    DecisionTreeClassifier(random_state=42),
        "rf":       RandomForestRegressor(n_estimators=req.n_estimators, random_state=42)
                    if is_regression else
                    RandomForestClassifier(n_estimators=req.n_estimators, random_state=42),
        "svm":      SVC(probability=True),
        "linreg":   LinearRegression(),
        "ridge":    Ridge(),
        "lasso":    Lasso(),
    }
    if algo_id not in MAP:
        raise HTTPException(400, f"Unknown algorithm: {algo_id}")
    return MAP[algo_id]


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        vals = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    pairs = sorted(zip(feature_names, vals.tolist()), key=lambda x: -x[1])
    total = sum(v for _, v in pairs) or 1
    return [{"feature": f, "importance": round(v / total, 4)} for f, v in pairs[:15]]


# ════════════════════════════════════════════════════════════
# UNSUPERVISED
# ════════════════════════════════════════════════════════════
def run_unsupervised(df: pd.DataFrame, req: AnalyzeRequest):
    df = df.dropna()
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        raise HTTPException(400, "Need at least 2 numeric columns for unsupervised learning")

    scaler = StandardScaler()
    X = scaler.fit_transform(num_df)
    feature_names = num_df.columns.tolist()

    results = {}
    for algo_id in req.algorithms:
        results[algo_id] = run_single_unsupervised(algo_id, X, feature_names, req)

    return {
        "method": "unsupervised",
        "features": feature_names,
        "n_samples": len(X),
        "results": results
    }


def run_single_unsupervised(algo_id, X, feature_names, req):
    out = {"algo": algo_id}

    if algo_id == "pca":
        n = min(req.n_components, X.shape[1])
        pca = PCA(n_components=n)
        X_proj = pca.fit_transform(X)
        out["projection"] = X_proj[:, :2].tolist()
        out["explained_variance"] = pca.explained_variance_ratio_.tolist()
        out["cumulative_variance"] = np.cumsum(pca.explained_variance_ratio_).tolist()
        # loadings
        loadings = pca.components_[:2]
        out["loadings"] = [{"feature": f, "pc1": round(float(loadings[0, i]), 4), "pc2": round(float(loadings[1, i]), 4)}
                           for i, f in enumerate(feature_names)]
        out["metrics"] = {
            "n_components": n,
            "total_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
            "pc1_variance": round(float(pca.explained_variance_ratio_[0]), 4),
            "pc2_variance": round(float(pca.explained_variance_ratio_[1] if n > 1 else 0), 4),
        }

    elif algo_id == "tsne":
        n = min(2, X.shape[1])
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_proj = tsne.fit_transform(X)
        out["projection"] = X_proj.tolist()
        out["metrics"] = {"kl_divergence": round(float(tsne.kl_divergence_), 4), "n_iter": tsne.n_iter_}

    elif algo_id == "kmeans":
        # Elbow
        inertias = []
        ks = list(range(1, min(11, len(X))))
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(round(float(km.inertia_), 2))

        km_final = KMeans(n_clusters=req.n_clusters, random_state=42, n_init=10)
        labels = km_final.fit_predict(X)
        silh = silhouette_score(X, labels) if len(set(labels)) > 1 else 0

        # 2D projection for visualization
        pca2 = PCA(n_components=2)
        X2 = pca2.fit_transform(X)
        out["projection"] = [{"x": float(p[0]), "y": float(p[1]), "cluster": int(l)} for p, l in zip(X2, labels)]
        out["elbow"] = {"ks": ks, "inertias": inertias}
        cluster_sizes = {str(k): int((labels == k).sum()) for k in range(req.n_clusters)}
        out["cluster_sizes"] = cluster_sizes
        out["metrics"] = {
            "n_clusters": req.n_clusters,
            "silhouette": round(float(silh), 4),
            "inertia": round(float(km_final.inertia_), 2),
            "n_samples": len(X)
        }

    elif algo_id == "dbscan":
        db = DBSCAN(eps=req.eps, min_samples=req.min_samples)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        silh = silhouette_score(X, labels) if n_clusters > 1 and len(set(labels)) > 1 else 0
        pca2 = PCA(n_components=2)
        X2 = pca2.fit_transform(X)
        out["projection"] = [{"x": float(p[0]), "y": float(p[1]), "cluster": int(l)} for p, l in zip(X2, labels)]
        out["metrics"] = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette": round(float(silh), 4),
            "n_samples": len(X)
        }

    elif algo_id == "hierarch":
        agg = AgglomerativeClustering(n_clusters=req.n_clusters)
        labels = agg.fit_predict(X)
        silh = silhouette_score(X, labels) if len(set(labels)) > 1 else 0
        pca2 = PCA(n_components=2)
        X2 = pca2.fit_transform(X)
        out["projection"] = [{"x": float(p[0]), "y": float(p[1]), "cluster": int(l)} for p, l in zip(X2, labels)]
        out["metrics"] = {
            "n_clusters": req.n_clusters,
            "silhouette": round(float(silh), 4),
            "n_samples": len(X)
        }

    elif algo_id == "iso":
        iso = IsolationForest(random_state=42)
        scores = iso.fit_predict(X)
        n_anomalies = int((scores == -1).sum())
        pca2 = PCA(n_components=2)
        X2 = pca2.fit_transform(X)
        out["projection"] = [{"x": float(p[0]), "y": float(p[1]), "anomaly": int(l == -1)} for p, l in zip(X2, scores)]
        out["metrics"] = {
            "n_anomalies": n_anomalies,
            "anomaly_rate": round(n_anomalies / len(X), 4),
            "n_samples": len(X)
        }

    return out


@app.get("/")
def root():
    return {"status": "ML Studio API is running 🚀"}
