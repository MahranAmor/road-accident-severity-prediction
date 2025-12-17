import pandas as pd

# =========================
# 1. Charger les données
# =========================
carac = pd.read_csv("carcteristiques-2022.csv", sep=";")
usagers = pd.read_csv("usagers-2022 (1).csv", sep=";")
vehicules = pd.read_csv("vehicules-2022.csv", sep=";")
lieux = pd.read_csv("lieux-2022.csv", sep=";", low_memory=False)

# Renommer la colonne Accident_Id en Num_Acc dans carac pour correspondre aux autres fichiers
carac = carac.rename(columns={"Accident_Id": "Num_Acc"})

# =========================
# 2. Créer la variable cible (gravité accident)
# gravité accident = gravité la plus grave parmi les usagers
# =========================
grav_acc = (
    usagers
    .groupby("Num_Acc")["grav"]
    .min()
    .reset_index()
)

# =========================
# 3. Agréger les usagers (features)
# =========================
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

# =========================
# 4. Agréger les véhicules (features)
# =========================
vehicules_agg = (
    vehicules
    .groupby("Num_Acc")
    .agg(
        nb_vehicules=("id_vehicule", "count")
    )
    .reset_index()
)

# =========================
# 5. Fusionner toutes les données
# =========================
data = (
    carac
    .merge(grav_acc, on="Num_Acc", how="inner")
    .merge(usagers_agg, on="Num_Acc", how="left")
    .merge(vehicules_agg, on="Num_Acc", how="left")
    .merge(lieux, on="Num_Acc", how="left")
)

# =========================
# 6. Créer une cible binaire (optionnel)
# 1 = accident grave (tué ou blessé hospitalisé)
# 0 = accident non grave
# =========================
data["grave"] = data["grav"].apply(lambda x: 1 if x in [2, 3] else 0)

# =========================
# 7. Nettoyage simple
# =========================
data = data.dropna(subset=["grav"])
data = data.fillna(0)

# =========================
# 8. Résultat
# =========================
print("Shape du dataset final :", data.shape)
print(data.head())

# =========================
# 9. Sauvegarder le dataset final
# =========================
data.to_csv("accidents_dataset_final.csv", index=False, sep=";")
print("\nLe dataset final a été sauvegardé dans 'accidents_dataset_final.csv'")
