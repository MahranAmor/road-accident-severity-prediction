from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve static frontend
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
if not os.path.isdir(static_dir):
    static_dir = os.path.join(os.getcwd(), 'static')
app.mount('/static', StaticFiles(directory=static_dir), name='static')

MODEL = None
SELECTED_FEATURES = []
IMPUTER = None
THRESHOLD = 0.67  # default optimized threshold

# path for saved model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'xgb_model.joblib')
if not os.path.isdir(os.path.dirname(MODEL_PATH)):
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    except Exception:
        pass


@app.get('/', response_class=HTMLResponse)
async def index():
    index_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse('<h1>Dashboard static file not found</h1>', status_code=404)


@app.get('/features')
async def features_endpoint():
    return JSONResponse({'features': SELECTED_FEATURES, 'threshold': THRESHOLD})


def _train_model_from_csv():
    global MODEL, SELECTED_FEATURES, IMPUTER, THRESHOLD
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dataset_final_processed.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.getcwd(), 'data', 'dataset_final_processed.csv')

    if not os.path.exists(data_path):
        print('Data file not found at', data_path)
        return False

    df = pd.read_csv(data_path)
    if 'grav' in df.columns:
        df = df.drop(columns=['grav'])
    if 'grave' not in df.columns:
        raise RuntimeError("Dataset must contain 'grave' target column")

    y = df['grave'].copy()
    X = df.drop(columns=['grave'])

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    dropped = []
    for col in non_numeric:
        conv = pd.to_numeric(X[col].astype(str).str.replace(r'[()\\s]', '', regex=True).str.replace(',', '.', regex=False), errors='coerce')
        if conv.notna().mean() >= 0.8:
            X[col] = conv
        else:
            dropped.append(col)
    if dropped:
        X = X.drop(columns=dropped, errors='ignore')

    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        X = X.drop(columns=all_nan, errors='ignore')

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise RuntimeError('No numeric features available for modeling')

    X_num = X[numeric_cols].copy()

    preferred = ['agg', 'col', 'nb_usagers', 'nb_vehicules', 'v1', 'plan', 'situ', 'vma']
    SELECTED_FEATURES = [c for c in preferred if c in X_num.columns]
    if len(SELECTED_FEATURES) < 1:
        k = min(8, X_num.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_num, y)
        mask = selector.get_support()
        SELECTED_FEATURES = list(X_num.columns[mask])

    IMPUTER = SimpleImputer(strategy='median')
    X_sel = X_num.loc[:, SELECTED_FEATURES]
    X_sel_imp = pd.DataFrame(IMPUTER.fit_transform(X_sel), columns=SELECTED_FEATURES)

    neg = int((y == 0).sum()) if hasattr(y, 'sum') else max(1, int((y == 0)))
    pos = int((y == 1).sum()) if hasattr(y, 'sum') else max(1, int((y == 1)))
    ratio = max(1, int(round(neg / pos)))

    params = dict(colsample_bytree=1.0, learning_rate=0.1, max_depth=5, n_estimators=200, subsample=1.0, scale_pos_weight=ratio)
    MODEL = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)
    MODEL.fit(X_sel_imp, y)
    THRESHOLD = 0.67
    print('Model trained on CSV. Features:', SELECTED_FEATURES)

    return True


def _load_saved_model():
    global MODEL, SELECTED_FEATURES, IMPUTER, THRESHOLD
    try:
        from joblib import load
        if os.path.exists(MODEL_PATH):
            payload = load(MODEL_PATH)
            MODEL = payload.get('model')
            SELECTED_FEATURES = payload.get('features', [])
            IMPUTER = payload.get('imputer')
            THRESHOLD = payload.get('threshold', THRESHOLD)
            print('Loaded model from', MODEL_PATH)
            return True
    except Exception as e:
        print('Failed to load saved model:', e)
    return False


@app.on_event('startup')
def train_or_load_model():
    # Prefer loading a saved joblib model for fast startup
    loaded = _load_saved_model()
    if loaded:
        return
    # else train from CSV
    ok = _train_model_from_csv()
    if not ok:
        print('Model not trained: data missing')


@app.post('/predict')
async def predict(request: Request):
    global MODEL, SELECTED_FEATURES, IMPUTER, THRESHOLD
    if MODEL is None:
        return JSONResponse({'error': 'Model not available. Ensure data file exists and app has trained model.'}, status_code=500)

    payload = await request.json()
    try:
        values = [float(payload.get(fe, None)) for fe in SELECTED_FEATURES]
    except Exception:
        return JSONResponse({'error': 'Invalid input types. Provide numeric values for features.'}, status_code=400)

    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values):
        return JSONResponse({'error': f'Missing or NaN feature value. Required features: {SELECTED_FEATURES}'}, status_code=400)

    arr = np.array(values).reshape(1, -1)
    arr_imp = IMPUTER.transform(arr)
    proba = MODEL.predict_proba(arr_imp)[0, 1]
    pred_default = int(proba >= 0.5)
    pred_thresh = int(proba >= THRESHOLD)

    return JSONResponse({'probability': float(proba), 'prediction_default': int(pred_default), 'prediction_threshold': int(pred_thresh), 'threshold': float(THRESHOLD), 'used_features': SELECTED_FEATURES})
