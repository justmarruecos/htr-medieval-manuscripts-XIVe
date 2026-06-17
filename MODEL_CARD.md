# Model Card — Kraken HTR Medieval French

## Informations générales

| Champ | Valeur |
|-------|--------|
| Modèle | Kraken VGSL (fine-tuné) |
| Modèle de base | cremma-medieval.mlmodel |
| Tâche | Reconnaissance de texte manuscrit (HTR) |
| Langues | Français médiéval (XIVe-XVe siècles) |
| Licence données | CC-BY 4.0 (CATMuS + HIMANIS-Guérin) |
| Projet | MD5 2026, Master Data/IA, HETIC |

## Données d'entraînement

- **Corpus** : CATMuS Medieval (25 812 lignes) + HIMANIS-Guérin JJ207/JJ210 (7 698 lignes)
- **Total** : 33 510 lignes, 40 groupes manuscrits distincts
- **Période** : XIVe siècle (28%) + XVe siècle (72%)
- **Types de documents** : manuscrits littéraires (CATMuS) + registres de chancellerie royale (HIMANIS)
- **Split train/val/test** : 70/17.8/12.3% par manuscrit (GroupShuffleSplit, seeds 47/57)

## Architecture

- **Type** : VGSL (Variable-size Graph Specification Language) + CTC loss
- **Couches** : Conv × 4 → MaxPool × 3 → BiLSTM × 3 → Linear
- **Paramètres** : 4.0M
- **Entrée** : image de ligne en niveaux de gris (hauteur 120px, largeur variable)

## Entraînement

### Run v1 (lrate=1e-3)
| Paramètre | Valeur |
|-----------|--------|
| Learning rate | 1e-3 |
| Batch size | 8 |
| Optimizer | AdamW |
| Early stopping | patience=5 |
| Meilleur epoch | 6 |
| val_metric | 0.2402 |
| Arrêt | epoch 16 |

### Run v2 (lrate=1e-4) — en cours
| Paramètre | Valeur |
|-----------|--------|
| Learning rate | 1e-4 |
| Batch size | 8 |
| Optimizer | AdamW |
| Early stopping | patience=5 |
| Meilleur epoch | à compléter |
| val_metric | à compléter |

## Performances

### Segmentation (Kraken BLLA, zéro-shot)
| Métrique | Valeur |
|----------|--------|
| IoU moyen | 0.604 |
| IoU médian | 0.614 |
| Lignes IoU > 0.75 | 0/40 |

### Reconnaissance HTR
| Modèle | val_accuracy |
|--------|-------------|
| cremma-medieval (baseline zéro-shot) | à évaluer |
| Fine-tuné lrate=1e-3 (v1) | 0.2402 |
| Fine-tuné lrate=1e-4 (v2) | à compléter |

## Utilisations prévues

- Transcription automatique de manuscrits médiévaux français
- Recherche en humanités numériques
- Benchmark comparatif avec TrOCR+LoRA (travail du groupe)

## Limitations

- Performances limitées sur les manuscrits très dégradés ou à encres très pâles
- Alphabet de 1 148 caractères — certains caractères rares (ex: `Λ`, `δ`) peu représentés
- IoU segmentation sous le seuil de validation (0.75) en zéro-shot —
  amélioration possible via `ketos segtrain` sur les 7 698 lignes HIMANIS annotées

## Notes techniques

- Bug Kraken 7.0.2 : `binary_dataset_split` non transmis par le CLI —
  contournement via API Python (injection directe `train_set`/`val_set`)
- Tridis Medieval EarlyModern : fichier 0 octet, baseline zéro-shot
  réalisée avec cremma-medieval à la place