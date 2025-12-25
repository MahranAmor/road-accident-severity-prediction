import pandas as pd
import numpy as np
import os

# =========================
# 1. Charger les données
# =========================
def load_data():
    """Charge tous les fichiers CSV des accidents"""
    data_path = "../data"
    
    carac = pd.read_csv(os.path.join(data_path, "carcteristiques-2022.csv"), sep=";")
    usagers = pd.read_csv(os.path.join(data_path, "usagers-2022 (1).csv"), sep=";")
    vehicules = pd.read_csv(os.path.join(data_path, "vehicules-2022.csv"), sep=";")
    lieux = pd.read_csv(os.path.join(data_path, "lieux-2022.csv"), sep=";", low_memory=False)
    
    # Renommer la colonne Accident_Id en Num_Acc
    carac = carac.rename(columns={"Accident_Id": "Num_Acc"})
    
    return carac, usagers, vehicules, lieux

def load_final_dataset(filepath="../data/accidents_dataset_final.csv"):
    """Charge le dataset final des accidents avec gestion des types"""
    # Charger les données avec le bon séparateur
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    
    print(f"✓ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
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
    usagers_agg = (
        usagers
        .groupby("Num_Acc")
        .agg(
            nb_usagers=("grav", "count"),
            age_moyen=("an_nais", lambda x: 2024 - x.mean()),
            presence_pieton=("catu", lambda x: int((x == 3).any()))
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
    data["grave"] = data["grav"].apply(lambda x: 1 if x in [2, 3] else 0)
    return data

# =========================
# 7. Nettoyage simple
# =========================
def clean_data(data):
    """Nettoie les données en supprimant les valeurs manquantes"""
    data = data.dropna(subset=["grav"])
    data = data.fillna(0)
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
    data = clean_data(data)
    
    print(f"✅ Dataset final créé : {data.shape[0]} lignes, {data.shape[1]} colonnes")
    
    # Sauvegarder
    output_path = "../data/accidents_dataset_final.csv"
    data.to_csv(output_path, index=False, sep=";")
    print(f"💾 Sauvegardé dans : {output_path}")
    
    return data

if __name__ == "__main__":
    data = prepare_dataset()
    print("\n📋 Aperçu des données :")
    print(data.head())
