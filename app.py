from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io, base64, warnings, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, silhouette_score
)
from sklearn.decomposition import PCA

# ✅ FIXED: use templates folder
app = Flask(__name__)
CORS(app)


# ✅ HOME ROUTE FIXED
@app.route('/')
def index():
    return render_template('index.html')


# Optional: remove favicon error
@app.route('/favicon.ico')
def favicon():
    return '', 204


# ── Utility ──
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Detect Problem ──
def detect_problem(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if cat_cols:
        return 'classification', cat_cols[-1]

    if num_cols:
        target = num_cols[-1]
        if df[target].nunique() <= 10:
            return 'classification', target
        return 'regression', target

    return 'clustering', None


# ── Preprocess ──
def preprocess(df, target):
    df = df.copy()
    steps = []

    # Encode categorical
    for col in df.select_dtypes(exclude=[np.number]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    steps.append("Encoded categorical columns")

    # Fill missing
    df = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(df), columns=df.columns)
    steps.append("Filled missing values")

    # Split
    if target:
        X = df.drop(columns=[target])
        y = df[target]
    else:
        X = df
        y = None

    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    steps.append("Standardized data")

    return X, y, steps


# ── Regression ──
def run_regression(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    fig = plt.figure()
    plt.scatter(y_te, y_pred)
    img = fig_to_b64(fig)

    return {
        "algorithm": "Linear Regression",
        "metrics": {
            "R2": round(r2_score(y_te, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_te, y_pred)), 2)
        },
        "chart": img,
        "predictions": [{"actual": float(a), "predicted": float(p)} for a, p in zip(y_te[:10], y_pred[:10])]
    }


# ── Classification ──
def run_classification(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier()
    }

    scores = {}
    for name, m in models.items():
        scores[name] = cross_val_score(m, X, y, cv=3).mean()

    best_name = max(scores, key=scores.get)
    model = models[best_name]
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    fig = plt.figure()
    plt.imshow(confusion_matrix(y_te, y_pred))
    img = fig_to_b64(fig)

    return {
        "algorithm": best_name,
        "metrics": {
            "Accuracy": round(accuracy_score(y_te, y_pred), 4)
        },
        "cv_scores": scores,
        "chart": img,
        "predictions": [
            {"actual": int(a), "predicted": int(p), "correct": bool(a == p)}
            for a, p in zip(y_te[:10], y_pred[:10])
        ]
    }


# ── Clustering ──
def run_clustering(X):
    model = KMeans(n_clusters=3)
    labels = model.fit_predict(X)

    fig = plt.figure()
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
    img = fig_to_b64(fig)

    return {
        "algorithm": "KMeans",
        "metrics": {"Clusters": 3},
        "chart": img,
        "predictions": [{"row": i, "cluster": int(c)} for i, c in enumerate(labels[:10])]
    }


# ── MAIN API ──
@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file"}), 400

    df = pd.read_csv(file)

    problem, target = detect_problem(df)
    X, y, steps = preprocess(df, target)

    if problem == "regression":
        result = run_regression(X, y)
    elif problem == "classification":
        result = run_classification(X, y)
    else:
        result = run_clustering(X)

    return jsonify({
        "profile": {
            "rows": len(df),
            "cols": len(df.columns),
            "numeric": int(len(df.select_dtypes(include=np.number).columns)),
            "categorical": int(len(df.select_dtypes(exclude=np.number).columns)),
            "missing": int(df.isnull().sum().sum()),
            "preview": df.head(5).astype(str).to_dict(orient="records")
        },
        "problem": problem,
        "target": target,
        "steps": steps,
        **result
    })


# ── RUN ──
if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(port=5050, debug=True)