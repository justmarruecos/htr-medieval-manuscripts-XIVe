# HTR Medieval Manuscripts XIVe-XVe

Pipeline complet de reconnaissance de texte manuscrit (HTR) sur des manuscrits
médiévaux français des XIVe et XVe siècles. Projet MD5 2026 — Master Data/IA, HETIC.

## Structure du projet

htr-medieval-manuscripts-XIVe/

├── dataset_nlp/

│   ├── catmus_french_14_15_clean.csv   # CATMuS filtré (25 812 lignes)

│   ├── himanis_clean.csv               # HIMANIS-Guérin extrait (7 698 lignes)

│   ├── corpus_final_clean.csv          # Corpus unifié (33 510 lignes)

│   ├── splits/

│   │   ├── train.csv                   # 23 444 lignes

│   │   ├── val.csv                     # 5 954 lignes

│   │   └── test.csv                    # 4 112 lignes (SCELLÉ)

│   └── binary_data/                    # Fichiers Arrow pour ketos train

├── experiments/

│   ├── 01b_corpus_catmus_fix.ipynb     # Reconstruction mapping CATMuS

│   ├── 01c_corpus_himanis_extract.ipynb # Extraction PAGE-XML HIMANIS

│   ├── 01d_corpus_merge_split.ipynb    # Fusion + split 70/15/15

│   ├── 02b_test_preprocessing.ipynb    # Validation prétraitement + IoU

│   ├── 03b_compile_ketos.ipynb         # Compilation Arrow + entraînement

│   └── journal.jsonl                   # Journal des expériences

├── segmentations/                      # Export PAGE-XML Kraken BLLA

├── src/

│   └── preprocessing.py               # Fonctions prétraitement (deskew/CLAHE/Sauvola)

├── tests/

│   └── test_preprocessing.py          # 16 tests pytest (tous PASSED)

├── models/

│   └── finetune_cremma_v2/            # Checkpoints fine-tuning (lrate=1e-4)

├── DATA_SOURCES.md                    # Sources, licences, répartition

├── CONVENTIONS_TRANSCRIPTION.md       # Conventions transcription + prétraitement

└── MODEL_CARD.md                      # Description du modèle final

## Reproduire les résultats

### Prérequis

```bash
git clone https://github.com/justmarruecos/htr-medieval-manuscripts-XIVe.git
cd htr-medieval-manuscripts-XIVe
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Données

Télécharger HIMANIS-Guérin depuis Zenodo (DOI: 10.5281/zenodo.5535306) :
```bash
# Placer le fichier Guerin(2).zip dans dataset_nlp/raw_himanis/guerin2/
# puis extraire
```

CATMuS Medieval est téléchargé automatiquement via HuggingFace dans
`01b_corpus_catmus_fix.ipynb`.

### Reconstruction du corpus

```bash
jupyter nbconvert --to notebook --execute experiments/01b_corpus_catmus_fix.ipynb
jupyter nbconvert --to notebook --execute experiments/01c_corpus_himanis_extract.ipynb
jupyter nbconvert --to notebook --execute experiments/01d_corpus_merge_split.ipynb
```

**Seeds reproductibilité** : `gss1=47, gss2=57`

**SHA-256 des splits** :
- `train.csv` : `4b30cdb9aece87cac60d835986e0e5bfa331c8c47b1ccd72d473796d882325e3`
- `val.csv` : `592f2e69fb5df7ecb5f7225d012352c32abee860ac13a276cd8bd2159045668e`
- `test.csv` : `3df155b380d8316c29b0f758192fd71dcb9e6f6620e42c090d9f4331cf0a6f08`

### Tests

```bash
python -m pytest tests/ -v
# 16 passed in 1.20s
```

### Entraînement

```bash
# Voir experiments/03b_compile_ketos.ipynb pour le pipeline complet
# Modèle de base : cremma-medieval.mlmodel
# Learning rate : 1e-4 (fine-tuning)
# Batch size : 8
# Early stopping : patience=5
```

## Résultats

### Segmentation (Kraken BLLA, zéro-shot)

| Métrique | Valeur |
|----------|--------|
| IoU moyen | 0.604 |
| IoU médian | 0.614 |
| Lignes IoU > 0.75 | 0/40 |
| Lignes IoU > 0.85 | 0/40 |

Page de référence : `FRAN_0021_33533_A.jpg` (HIMANIS JJ207)
Vérité-terrain : annotations Transkribus (40 lignes)

### Reconnaissance HTR

| Modèle | val_metric (accuracy) |
|--------|----------------------|
| cremma-medieval (zéro-shot) | à évaluer |
| Kraken fine-tuné lrate=1e-3 | 0.2402 (epoch 6) |
| Kraken fine-tuné lrate=1e-4 | en cours |

## Notes techniques

### Bug Kraken 7.0.2

Le CLI `ketos train -f binary` ne transmet pas `binary_dataset_split=True`
à `ArrowIPCRecognitionDataset`, rendant les splits intégrés aux fichiers
`.arrow` inopérants. Contournement : injection directe des datasets via
l'API Python (`VGSLRecognitionDataModule.train_set`/`val_set`).

### Tridis Medieval EarlyModern

Le fichier `Tridis_Medieval_EarlyModern.mlmodel` présent dans le repo
est un fichier vide (0 octets) — le téléchargement a échoué silencieusement.
La baseline zéro-shot est donc évaluée avec `cremma-medieval.mlmodel`.

## Licences corpus

| Corpus | Licence |
|--------|---------|
| CATMuS Medieval | CC-BY 4.0 |
| HIMANIS-Guérin | CC-BY 4.0 |
