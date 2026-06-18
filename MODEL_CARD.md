# Model Card — Pipeline HTR + NLP Medieval French

## Informations générales

| Champ | Valeur |
|---|---|
| Modèle HTR | Kraken VGSL (fine-tuné) |
| Modèle NER | Jean-Baptiste/camembert-ner (zéro-shot) |
| Modèle embedding | paraphrase-multilingual-mpnet-base-v2 |
| Tâche | HTR + Normalisation + NER + Topic Modeling |
| Langues | Français médiéval (XIVe-XVe siècles) |
| Licence données | CC-BY 4.0 (CATMuS + HIMANIS-Guérin) |
| Projet | MD5 2026, Master Data/IA, HETIC |
| Version | v1.0.0 |

---

## Données d'entraînement

- **Corpus** : CATMuS Medieval (25 812 lignes) + HIMANIS-Guérin JJ207/JJ210 (7 698 lignes)
- **Total** : 33 510 lignes, 40 groupes manuscrits distincts
- **Période** : XIVe siècle (28%) + XVe siècle (72%)
- **Types** : manuscrits littéraires + registres de chancellerie royale
- **Split** : 70/17.8/12.3% par manuscrit (GroupShuffleSplit, seeds 47/57)

| Split | Lignes | SHA-256 |
|---|---|---|
| train | 23 444 | `4b30cdb9...` |
| val | 5 954 | `592f2e69...` |
| test | 4 112 | `3df155b3...` |

---

## Volet 1 — HTR

### Architecture Kraken VGSL

| Paramètre | Valeur |
|---|---|
| Type | VGSL + CTC loss |
| Couches | Conv×4 → MaxPool×3 → BiLSTM×3 → Linear |
| Paramètres | 4.0M |
| Entrée | Image ligne niveaux de gris (h=120px, w variable) |

### Entraînement

| Run | lrate | val_metric | Epoch | Statut |
|---|---|---|---|---|
| v1 | 1e-3 | 0.2402 | 6 | sous-optimal |
| v2 | 1e-4 | **0.7898** | 18 | ✅ retenu |

### Performances HTR

| Métrique | Valeur | Seuil | Statut |
|---|---|---|---|
| val_metric (accuracy) | 0.7898 | > 0.75 | ✅ |
| CER baseline zéro-shot | 69.5% | — | référence |
| Delta fine-tuning | +48.5 pts | — | ✅ |
| IoU segmentation (BLLA) | 0.604 | > 0.75 | ⚠️ zéro-shot |

---

## Volet 2 — NLP

### Normalisation orthographique

| Métrique | Valeur | Seuil | Statut |
|---|---|---|---|
| CER modification | 2.51% | < 10% | ✅ VALIDE |
| Lignes modifiées | 122/200 (61%) | — | — |
| Tests pytest | 22/22 | — | ✅ |

**Pipeline règles** (ordre critique) :
1. `expand_abbreviations()` — avant NFD
2. `normalize_unicode()` — NFD + espaces
3. `normalize_uv()` — u initial → v
4. `normalize_ij()` — j initial devant a/o/u → i
5. `normalize_punctuation()` — ·→espace, ¶→.

### NER zéro-shot (CamemBERT)

| Métrique | Valeur |
|---|---|
| Modèle | Jean-Baptiste/camembert-ner |
| Entités détectées (200 lignes) | 215 |
| Distribution | PER=143, LOC=54, MISC=14, ORG=4 |
| Score moyen | 0.880 |
| Précision estimée PER | 60% |
| Précision estimée LOC | 35% |
| Précision globale estimée | ~50% |
| Tests pytest | 15/15 | ✅ |

### Topic Modeling (BERTopic)

| Paramètre | Valeur |
|---|---|
| Modèle embedding | paraphrase-multilingual-mpnet-base-v2 |
| Clustering | KMeans(n=5) |
| UMAP | n_neighbors=5, n_components=3 |
| Documents | 34 (agrégés par shelfmark) |
| Topics trouvés | 5 |

| Topic | Label | Docs |
|---|---|---|
| 0 | Littérature courtoise | 11 |
| 1 | Actes royaux / chancellerie | 7 |
| 2 | Textes hagiographiques | 6 |
| 3 | Chroniques historiques | 6 |
| 4 | Textes moraux / didactiques | 4 |

---

## Tests

| Suite | Tests | Statut |
|---|---|---|
| test_preprocessing.py | 16 | ✅ |
| test_normalisation.py | 22 | ✅ |
| test_ner.py | 15 | ✅ |
| **Total** | **53** | ✅ |

---

## Utilisations prévues

- Transcription automatique de manuscrits médiévaux français
- Extraction d'entités nommées (personnes, lieux)
- Modélisation thématique de corpus médiévaux
- Recherche en humanités numériques

---

## Limitations

- IoU segmentation sous le seuil de validation (0.75) en zéro-shot
- NER précision ~50% en zéro-shot — fine-tuning requis pour F1 > 0.65
- Pas de type DATE supporté par camembert-ner
- Faux positifs LOC sur mots médiévaux (ioye, espaulles, cuer)
- Normalisation lexicale hors périmètre (nécessite mT5+LoRA)

---

## Notes techniques

- Bug Kraken 7.0.2 : `binary_dataset_split` non transmis par CLI
- Tridis Medieval EarlyModern : fichier 0 octet, baseline avec cremma-medieval
- HDBSCAN échoue sur 34 docs → KMeans forcé à 5 clusters