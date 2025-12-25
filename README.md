# 🚗 Prédiction de la Gravité des Accidents de la Route

Projet de Machine Learning pour prédire la gravité des accidents de la route en France (données 2022).

## 📁 Structure du Projet

```
accidents-fr/
├── data/                           # Données brutes et traitées
│   ├── carcteristiques-2022.csv   # Caractéristiques des accidents
│   ├── usagers-2022 (1).csv       # Informations sur les usagers
│   ├── vehicules-2022.csv         # Informations sur les véhicules
│   ├── lieux-2022.csv             # Informations sur les lieux
│   └── accidents_dataset_final.csv # Dataset fusionné et nettoyé
├── notebooks/                      # Jupyter notebooks pour exploration
├── src/                           # Code source Python
│   └── data_preparation.py        # Pipeline de préparation des données
├── models/                        # Modèles ML entraînés
├── .gitignore                     # Fichiers à ignorer
├── requirements.txt               # Dépendances Python
└── README.md                      # Ce fichier
```

## 🎯 Objectif

Prédire la gravité d'un accident (grave/non grave) en fonction de :
- Caractéristiques de l'accident (jour, heure, météo, luminosité)
- Informations sur les usagers (âge, nombre, présence de piétons)
- Informations sur les véhicules (nombre, type)
- Informations sur le lieu (type de route, conditions)

## 🔧 Installation

1. **Cloner le repository**
```bash
git clone https://github.com/MahranAmor/road-accident-severity-prediction.git
cd road-accident-severity-prediction
```

2. **Créer un environnement virtuel**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## 🚀 Utilisation

### Préparer les données
```bash
cd src
python data_preparation.py
```

### Explorer les données
Ouvrir les notebooks dans `notebooks/` 

## 📊 Dataset Final

- **Lignes** : 55 302 accidents
- **Colonnes** : 38 features
- **Cible** : `grave` (0 = non grave, 1 = grave)

### Features principales
- `nb_usagers` : Nombre d'usagers impliqués
- `age_moyen` : Âge moyen des usagers
- `nb_vehicules` : Nombre de véhicules impliqués
- `presence_pieton` : Présence d'un piéton (0/1)
- `lum` : Luminosité
- `atm` : Conditions atmosphériques
- etc.

## 👥 Contributeurs

- Mahran Amor
- Ayoub Kallel

## 📝 Licence

Ce projet utilise des données publiques de l'ONISR (Observatoire National Interministériel de la Sécurité Routière).

## 🔗 Sources

- [Données accidents ONISR](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/)
