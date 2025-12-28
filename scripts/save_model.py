"""Train the tuned XGBoost on full dataset and save model+imputer+features to models/xgb_model.joblib"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from joblib import dump

root = Path(__file__).resolve().parents[1]
data_path = root / 'data' / 'dataset_final_processed.csv'
model_path = root / 'models' / 'xgb_model.joblib'
model_path.parent.mkdir(parents=True, exist_ok=True)

if not data_path.exists():
    raise SystemExit('data file not found: ' + str(data_path))

print('Loading', data_path)
df = pd.read_csv(data_path)
if 'grav' in df.columns:
    df = df.drop(columns=['grav'])
if 'grave' not in df.columns:
    raise SystemExit("target 'grave' not found in dataset")

y = df['grave'].copy()
X = df.drop(columns=['grave'])

# minimal cleaning
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
for col in non_numeric:
    conv = pd.to_numeric(X[col].astype(str).str.replace(r'[()\s]', '', regex=True).str.replace(',', '.', regex=False), errors='coerce')
    if conv.notna().mean() >= 0.8:
        X[col] = conv
    else:
        X = X.drop(columns=[col])

all_nan = [c for c in X.columns if X[c].isna().all()]
if all_nan:
    X = X.drop(columns=all_nan)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    raise SystemExit('no numeric cols')

X_num = X[numeric_cols]
preferred = ['agg', 'col', 'nb_usagers', 'nb_vehicules', 'v1', 'plan', 'situ', 'vma']
selected = [c for c in preferred if c in X_num.columns]
if len(selected) < 1:
    k = min(8, X_num.shape[1])
    sel = SelectKBest(score_func=f_classif, k=k)
    sel.fit(X_num, y)
    selected = list(X_num.columns[sel.get_support()])

imputer = SimpleImputer(strategy='median')
X_sel = X_num.loc[:, selected]
X_sel_imp = imputer.fit_transform(X_sel)

neg = int((y==0).sum())
pos = int((y==1).sum())
ratio = max(1, int(round(neg/pos)))
params = dict(colsample_bytree=1.0, learning_rate=0.1, max_depth=5, n_estimators=200, subsample=1.0, scale_pos_weight=ratio)
model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)
print('Training model...')
model.fit(X_sel_imp, y)

payload = {
    'model': model,
    'imputer': imputer,
    'features': selected,
    'threshold': 0.67
}

print('Saving to', model_path)
dump(payload, model_path)
print('Done')

