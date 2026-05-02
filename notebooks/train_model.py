import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ÉTAPE 1 : Charger le dataset


df = pd.read_csv("data/patients_dakar.csv")

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# ÉTAPE 2 : Préparer les features
le_sexe   = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded']   = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = [
    'age', 'sexe_encoded', 'temperature', 'tension_sys',
    'toux', 'fatigue', 'maux_tete', 'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

print(f"\nFeatures : {X.shape}")
print(f"Cible    : {y.shape}")

# ÉTAPE 3 : Séparer train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nEntraînement : {X_train.shape[0]} patients")
print(f"Test         : {X_test.shape[0]} patients")

# ÉTAPE 4 : Entraîner le modèle
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print(f"\nModèle entraîné !")
print(f"Nombre d'arbres  : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

# ÉTAPE 5 : Évaluer le modèle
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(f"\nMatrice de confusion :\n{cm}")

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Visualisation matrice de confusion
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel('Prédiction du modèle')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SénSanté')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("Figure sauvegardée dans figures/confusion_matrix.png")

# ÉTAPE 6 : Sérialiser le modèle
os.makedirs("models", exist_ok=True)

joblib.dump(model,        "models/model.pkl")
joblib.dump(le_sexe,      "models/encoder_sexe.pkl")
joblib.dump(le_region,    "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

size = os.path.getsize("models/model.pkl")
print(f"\nModèle sauvegardé : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")
print("Encodeurs et metadata sauvegardés.")

# ÉTAPE 7 : Tester le modèle rechargé

model_loaded    = joblib.load("models/model.pkl")
le_sexe_loaded  = joblib.load("models/encoder_sexe.pkl")
le_region_loaded= joblib.load("models/encoder_region.pkl")

print(f"\nModèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# Nouveau patient test
nouveau_patient = {
    'age'        : 28,
    'sexe'       : 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux'       : True,
    'fatigue'    : True,
    'maux_tete'  : True,
    'region'     : 'Dakar'
}

sexe_enc   = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

diagnostic = model_loaded.predict([features])[0]
probas     = model_loaded.predict_proba([features])[0]
proba_max  = probas.max()

print(f"\n--- Résultat du pré-diagnostic ---")
print(f"Patient    : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilité : {proba_max:.1%}")

print(f"\nProbabilités par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:8s} : {proba:.1%}  {bar}")