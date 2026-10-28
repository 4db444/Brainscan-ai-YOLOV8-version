# 🧠 BrainScan AI — Classification de tumeurs cérébrales sur IRM

Projet de deep learning qui classifie des IRM cérébrales en 4 catégories :
**gliome**, **méningiome**, **pas de tumeur**, **tumeur hypophysaire**.
Un CNN (TensorFlow/Keras) est entraîné sur le dataset puis déployé via une
application Streamlit pour prédire une classe à partir d'une image uploadée.

## Structure des fichiers

```
brainscan-ai/
├── pyproject.toml            # dépendances + config setuptools
├── download_data.py          # téléchargement du dataset (Google Drive)
├── .env / .env.example       # RANDOM_SEED (git-ignored)
├── data/
│   ├── raw/                  # images brutes (glioma/, meningioma/, notumor/, pituitary/)
│   └── processed/            # x_train.npy, x_test.npy, y_train.npy, y_test.npy, class_names.json
├── notebooks/                # orchestration uniquement
│   ├── 01_EDA.ipynb          # chargement + exploration
│   ├── 02_Preprocessing.ipynb# rééquilibrage, encodage, split, normalisation
│   ├── 03_Training.ipynb     # build + entraînement du CNN
│   └── 04_Evaluation.ipynb   # évaluation + matrices + prédictions
├── src/
│   ├── __init__.py           # re-exporte l'API publique
│   ├── config.py             # chemins, constantes, seed
│   ├── data/                 # load_data, explore, preprocess, save_data
│   ├── models/               # build_model, train_model, evaluate_model, predict_model
│   └── visualization/        # learning curves, confusion matrix, examples
├── models/                   # final_model.h5, best_model.h5
├── reports/
│   ├── tables/               # classification_report.csv, evaluation_stats.json
│   └── figures/              # class_distribution.png, learning_curves.png, ...
└── app/
    └── app.py                # interface Streamlit
```

## Instructions d'exécution

```bash
# 1. Installer les dépendances
pip install -e .

# 2. Configurer le seed
cp .env.example .env

# 3. Télécharger le dataset (~7000 images IRM)
python download_data.py

# 4. Exécuter les notebooks dans l'ordre
#    01_EDA -> 02_Preprocessing -> 03_Training -> 04_Evaluation

# 5. Lancer l'application
streamlit run app/app.py
```

## Modèle

- Architecture : Conv2D (32→64→128→256 filtres) + MaxPooling + BatchNormalization
  + Dropout(0.3) + Dense(256→128) + Softmax(4)
- Entrée : images RGB redimensionnées en 224×224, normalisées [0, 1]
- Optimiseur : Adam (lr = 1e-3), perte : categorical_crossentropy
- Callbacks : ModelCheckpoint (meilleur val_accuracy) + EarlyStopping (patience=5)
- Artefact : `models/final_model.h5` + `data/processed/class_names.json`
