# HTR Medieval Manuscripts XIVe-XVe

Pipeline complet HTR + NLP sur manuscrits médiévaux français des XIVe-XVe siècles.  
Projet MD5 2026 — Master Data/IA, HETIC.

**Équipe** :   
**Version** : `v1.0.0` — Branches : `main` ✅

---

## Structure du projet

htr-medieval-manuscripts-XIVe/

├── dataset_nlp/

│   ├── splits/

│   │   ├── train.csv                      # 23 444 lignes (SCELLÉ)

│   │   ├── val.csv                        # 5 954 lignes  (SCELLÉ)

│   │   └── test.csv                       # 4 112 lignes  (SCELLÉ)

│   └── SPLIT_MANIFEST.txt

├── experiments/

│   ├── 01b_corpus_catmus_fix.ipynb

│   ├── 01c_corpus_himanis_extract.ipynb

│   ├── 01d_corpus_merge_split.ipynb

│   ├── 02b_test_preprocessing.ipynb

│   ├── 03b_compile_ketos.ipynb

│   ├── 04_nlp_normalisation.ipynb

│   ├── 05_nlp_topic_modeling.ipynb

│   └── journal.jsonl                      # 10 entrées

├── segmentations/                         # Export PAGE-XML Kraken BLLA

├── src/

│   ├── preprocessing.py                   # deskew + CLAHE + Sauvola

│   ├── normalisation.py                   # Normalisation orthographique moyen français

│   ├── ner.py                             # NERPipeline (CamemBERT zéro-shot)

│   ├── api.py                             # FastAPI /analyze + /health

│   └── tei_export.py                      # Export TEI-XML

├── tests/

│   ├── test_preprocessing.py              # 16 tests

│   ├── test_normalisation.py              # 22 tests

│   └── test_ner.py                        # 15 tests — 53/53 PASSED total

├── models/

│   └── finetune_cremma_v2/                # Checkpoint epoch 18 (val_metric=0.7898)

├── CONVENTIONS_TRANSCRIPTION.md

├── CONVENTIONS_NLP.md

├── DATA_SOURCES.md

├── MODEL_CARD.md

├── data_contract.json

├── Dockerfile

├── docker-compose.yml

└── requirements.txt                       # 205 dépendances figées

---

## Reproduire les résultats

### Prérequis

```bash
git clone https://github.com/justmarruecos/htr-medieval-manuscripts-XIVe.git
cd htr-medieval-manuscripts-XIVe
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Données

CATMuS Medieval est téléchargé automatiquement via HuggingFace dans
`experiments/01b_corpus_catmus_fix.ipynb`.

HIMANIS-Guérin (DOI: 10.5281/zenodo.5535306) :
```bash
# Placer Guerin(2).zip dans dataset_nlp/raw_himanis/guerin2/ puis extraire
```

### Reconstruction du corpus

```bash
jupyter nbconvert --to notebook --execute experiments/01b_corpus_catmus_fix.ipynb
jupyter nbconvert --to notebook --execute experiments/01c_corpus_himanis_extract.ipynb
jupyter nbconvert --to notebook --execute experiments/01d_corpus_merge_split.ipynb
```

**Seeds reproductibilité** : `gss1=47, gss2=57`

**SHA-256 des splits** :
- `train.csv` : `4b30cdb9aece87cac60d835986e0e5bfa331c8c47b1ccd72d473796d882325e3`
- `val.csv`   : `592f2e69fb5df7ecb5f7225d012352c32abee860ac13a276cd8bd2159045668e`
- `test.csv`  : `3df155b380d8316c29b0f758192fd71dcb9e6f6620e42c090d9f4331cf0a6f08`

### Tests

```bash
pytest tests/ -v
# 53 passed
```

### Volet 1 — HTR

```bash
# Prétraitement + segmentation
jupyter nbconvert --to notebook --execute experiments/02b_test_preprocessing.ipynb

# Fine-tuning Kraken (lrate=1e-4)
jupyter nbconvert --to notebook --execute experiments/03b_compile_ketos.ipynb
# Checkpoint : models/finetune_cremma_v2/model-epoch=18-val_metric=0.7898.ckpt
```

### Volet 2 — NLP

```bash
# Normalisation orthographique
jupyter nbconvert --to notebook --execute experiments/04_nlp_normalisation.ipynb

# Topic modeling BERTopic
jupyter nbconvert --to notebook --execute experiments/05_nlp_topic_modeling.ipynb

# Lancer l'API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Ou via Docker
docker-compose up --build
```

---

## Résultats

### Volet 1 — HTR

#### Segmentation (Kraken BLLA, zéro-shot)

| Métrique | Valeur |
|---|---|
| IoU moyen | 0.604 |
| IoU médian | 0.614 |
| Page de référence | FRAN_0021_33533_A.jpg (HIMANIS JJ207) |

#### Reconnaissance HTR

| Modèle | val_metric | Statut |
|---|---|---|
| cremma-medieval baseline zéro-shot | 0.3048 (CER=69.5%) | baseline |
| Kraken fine-tuné lrate=1e-3 (v1) | 0.2402 | sous-optimal |
| Kraken fine-tuné lrate=1e-4 (v2) | **0.7898** | ✅ seuil > 0.75 |

**Delta** : +48.5 points accuracy (0.305 → 0.790)

### Volet 2 — NLP

#### Normalisation orthographique

| Métrique | Valeur | Seuil | Statut |
|---|---|---|---|
| CER modification | 2.51% | < 10% | ✅ VALIDE |
| Lignes modifiées | 122/200 (61%) | — | — |
| Tests pytest | 22/22 | — | ✅ |

#### NER zéro-shot (CamemBERT)

| Métrique | Valeur |
|---|---|
| Entités détectées (200 lignes val) | 215 |
| Distribution | PER=143, LOC=54, MISC=14, ORG=4 |
| Score moyen | 0.880 |
| Précision estimée globale | ~50% |
| Statut | Exploratoire — fine-tuning requis pour F1 > 0.65 |

#### Topic Modeling (BERTopic)

| Topic | Label | Docs |
|---|---|---|
| 0 | Littérature courtoise / récits chevaleresques | 11 |
| 1 | Actes royaux / registres de chancellerie | 7 |
| 2 | Textes hagiographiques / religieux | 6 |
| 3 | Chroniques / récits historiques | 6 |
| 4 | Textes moraux / didactiques | 4 |

Cible ≥ 3 topics cohérents : ✅ **5 topics**

---

## API

```bash
# Health check
curl http://localhost:8000/health

# Analyser un texte
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Jehan Rousseau demeure a Paris"}'
```

---

## Notes techniques

### Bug Kraken 7.0.2
Le CLI `ketos train -f binary` ne transmet pas `binary_dataset_split=True`.
Contournement : injection directe via API Python (`train_set`/`val_set`).

### Tridis Medieval EarlyModern
Fichier 0 octet — baseline zéro-shot réalisée avec `cremma-medieval.mlmodel`.

---

## Licences corpus

| Corpus | Licence |
|---|---|
| CATMuS Medieval | CC-BY 4.0 |
| HIMANIS-Guérin | CC-BY 4.0 |