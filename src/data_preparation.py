import pandas as pd
import numpy as np
import os

# Helper: resolve project data directory relative to this file
def _data_dir():
    # file is src/data_preparation.py -> project root is one level up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# =========================
# 1. Charger les données
# =========================
def load_data():
    """Charge tous les fichiers CSV des accidents"""
    data_path = _data_dir()

    carac = pd.read_csv(os.path.join(data_path, "carcteristiques-2022.csv"), sep=";")
    usagers = pd.read_csv(os.path.join(data_path, "usagers-2022 (1).csv"), sep=";")
    vehicules = pd.read_csv(os.path.join(data_path, "vehicules-2022.csv"), sep=";")
    lieux = pd.read_csv(os.path.join(data_path, "lieux-2022.csv"), sep=";", low_memory=False)
    
    # Renommer la colonne Accident_Id en Num_Acc
    carac = carac.rename(columns={"Accident_Id": "Num_Acc"})
    
    return carac, usagers, vehicules, lieux

def load_final_dataset(filepath=None):
    """Charge le dataset final des accidents avec gestion des types et nettoyage basique

    This will coerce non-numeric values to NaN, drop problematic columns like 'pr' and 'pr1',
    and impute numeric NaNs with the column median so models like KNN won't fail on NaNs.
    """
    if filepath is None:
        filepath = os.path.join(_data_dir(), 'accidents_dataset_final.csv')

    # Charger les données avec le bon séparateur
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    print(f"✓ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Apply robust cleaning to avoid strings like '(1)' and NaNs causing model errors
    df = clean_final_dataset(df)

    return df

# =========================
# 2. Créer la variable cible (gravité accident)
# =========================
def create_target(usagers):
    """Crée la variable cible: gravité la plus grave parmi les usagers"""
    grav_acc = (
        usagers
        .groupby("Num_Acc")["grav"]
        .min()
        .reset_index()
    )
    return grav_acc

# =========================
# 3. Agréger les usagers (features)
# =========================
def aggregate_usagers(usagers):
    """Agrège les informations des usagers par accident"""
    def _presence_pieton(x):
        # returns 1 if any row in the group has catu == 3, else 0
        try:
            return 1 if (x == 3).any() else 0
        except Exception:
            return 0

    usagers_agg = (
        usagers
        .groupby("Num_Acc")
        .agg(
            nb_usagers=("grav", "count"),
            age_moyen=("an_nais", lambda x: 2024 - x.mean()),
            presence_pieton=("catu", _presence_pieton)
        )
        .reset_index()
    )
    return usagers_agg

# =========================
# 4. Agréger les véhicules (features)
# =========================
def aggregate_vehicules(vehicules):
    """Agrège les informations des véhicules par accident"""
    vehicules_agg = (
        vehicules
        .groupby("Num_Acc")
        .agg(
            nb_vehicules=("id_vehicule", "count")
        )
        .reset_index()
    )
    return vehicules_agg

# =========================
# 5. Fusionner toutes les données
# =========================
def merge_all_data(carac, grav_acc, usagers_agg, vehicules_agg, lieux):
    """Fusionne toutes les données en un seul DataFrame"""
    data = (
        carac
        .merge(grav_acc, on="Num_Acc", how="inner")
        .merge(usagers_agg, on="Num_Acc", how="left")
        .merge(vehicules_agg, on="Num_Acc", how="left")
        .merge(lieux, on="Num_Acc", how="left")
    )
    return data

# =========================
# 6. Créer une cible binaire
# =========================
def create_binary_target(data):
    """Crée une cible binaire: 1 = grave (tué ou blessé hospitalisé), 0 = non grave"""
    # Ensure grav is numeric (coerce strings like '(1)' to numbers where possible)
    if 'grav' in data.columns:
        data['grav'] = pd.to_numeric(data['grav'], errors='coerce')
    data["grave"] = data["grav"].apply(lambda x: 1 if x in [2, 3] else 0)
    return data

# =========================
# 7. Nettoyage simple (robuste)
# =========================
def clean_final_dataset(data, drop_cols=None, impute=True):
    """Nettoie le DataFrame final pour qu'il soit utilisable par des modèles ML.

    Steps:
    - Optionally drop columns (default drops ['pr','pr1'])
    - Coerce target 'grav' to numeric and drop rows with missing target
    - Convert feature columns to numeric where possible (non-numeric -> NaN)
    - Impute numeric NaNs with column median (if impute=True)
    - Fill remaining non-numeric NaNs with a placeholder
    """
    if drop_cols is None:
        drop_cols = ['pr', 'pr1']

    # Drop requested columns if they exist
    for c in drop_cols:
        if c in data.columns:
            data = data.drop(columns=[c])

    # Ensure target exists and is numeric
    if 'grav' in data.columns:
        data['grav'] = pd.to_numeric(data['grav'], errors='coerce')
        # Drop rows where target is missing
        data = data.dropna(subset=['grav'])
    else:
        # If no 'grav' present, nothing to coerce; caller must handle
        pass

    # Attempt to coerce all other columns to numeric where reasonable
    for col in data.columns:
        if col == 'grav':
            continue
        # If column is object or contains non-numeric, coerce
        if data[col].dtype == object:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Impute numeric columns with median to avoid NaNs for models like KNN
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if impute and len(numeric_cols) > 0:
        medians = data[numeric_cols].median()
        data[numeric_cols] = data[numeric_cols].fillna(medians)

    # For any remaining non-numeric columns, fill NaNs with a placeholder
    non_numeric_cols = [c for c in data.columns if c not in numeric_cols]
    for c in non_numeric_cols:
        # keep strings but replace missing values with 'missing'
        data[c] = data[c].fillna('missing')

    return data

# =========================
# Pipeline complet
# =========================
def prepare_dataset():
    """Pipeline complet de préparation des données"""
    print("🔄 Chargement des données...")
    carac, usagers, vehicules, lieux = load_data()
    
    print("🎯 Création de la variable cible...")
    grav_acc = create_target(usagers)
    
    print("📊 Agrégation des usagers...")
    usagers_agg = aggregate_usagers(usagers)
    
    print("🚗 Agrégation des véhicules...")
    vehicules_agg = aggregate_vehicules(vehicules)
    
    print("🔗 Fusion des données...")
    data = merge_all_data(carac, grav_acc, usagers_agg, vehicules_agg, lieux)
    
    print("✨ Création de la cible binaire...")
    data = create_binary_target(data)
    
    print("🧹 Nettoyage des données...")
    data = clean_final_dataset(data)

    print(f"✅ Dataset final créé : {data.shape[0]} lignes, {data.shape[1]} colonnes")
    
    # Sauvegarder
    output_path = os.path.join(_data_dir(), 'accidents_dataset_final.csv')
    data.to_csv(output_path, index=False, sep=";")
    print(f"💾 Sauvegardé dans : {output_path}")
    
    return data

if __name__ == "__main__":
    data = prepare_dataset()
    print("\n📋 Aperçu des données :")
    print(data.head())
