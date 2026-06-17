# Sources de données

Ce document décrit les corpus utilisés pour l'entraînement et l'évaluation
du pipeline HTR sur manuscrits médiévaux français (XIVe-XVe siècles).

## Corpus retenus

### 1. CATMuS Medieval
- **URL** : https://huggingface.co/datasets/CATMuS/medieval
- **Licence** : CC-BY 4.0
- **Lignes retenues** : 25 812 (filtre : langue=French, siècle∈{14,15})
- **Manuscrits** : 38 groupes distincts (shelfmarks)
- **Description** : Fédération de 7 sous-projets HTR :
  HTRomance, HTRogène, CREMMA, Biblissima+ Fabliaux,
  ANR LIBER, GalliCorpora, PSL-Chartes Students.
- **Note** : l'index HF d'origine (`source_idx`) a été préservé pour
  reconstruire le mapping image↔transcription (voir `01b_corpus_catmus_fix.ipynb`).

### 2. HIMANIS-Guérin (registres JJ207 + JJ210)
- **URL** : https://zenodo.org/record/5535306
- **DOI** : 10.5281/zenodo.5535306
- **Licence** : CC-BY 4.0
- **Lignes retenues** : 7 698 (224 lignes vides exclues)
- **Pages** : 284 (139 pages JJ207 + 145 pages JJ210)
- **Description** : Registres du Trésor des Chartes (Archives nationales),
  série JJ176-235, couvrant la période 1441-1502 (XVe siècle).
  Annotations PAGE-XML produites par Transkribus (vérité-terrain humaine).
- **Type de document** : registre de chancellerie royale (cursive gothique tardive)

## Corpus écartés

### e-NDP (Registres Notre-Dame de Paris)
- **Raison** : licence CC-BY-NC-SA incompatible avec la contrainte #7 du brief
  (seules les licences CC-BY, CC-BY-SA et domaine public sont acceptées).

### CREMMA-BnF fr.22549 (extraction BeautifulSoup4)
- **Raison** : doublon quasi-total avec le sous-projet CREMMA déjà inclus
  dans CATMuS (2 615/2 956 lignes identiques confirmées).

## Répartition finale

| Corpus | Lignes | % | Siècles |
|--------|--------|---|---------|
| CATMuS Medieval | 25 812 | 77.0% | XIVe (37%) + XVe (63%) |
| HIMANIS-Guérin | 7 698 | 23.0% | XVe (100%) |
| **Total** | **33 510** | **100%** | XIVe (28%) + XVe (72%) |

## Split train/val/test

| Split | Lignes | % | Groupes manuscrits |
|-------|--------|---|-------------------|
| Train | 23 444 | 70.0% | 28 |
| Val | 5 954 | 17.8% | 6 |
| Test | 4 112 | 12.3% | 6 |

- **Méthode** : `GroupShuffleSplit` × 2 (groupe = shelfmark),
  seeds `gss1=47, gss2=57` (recherche multi-objectif : équilibre
  proportions 70/15/15 + répartition par siècle).
- **SHA-256 test set** : `3df155b380d8316c29b0f758192fd71dcb9e6f6620e42c090d9f4331cf0a6f08`
- **Test scellé** : ne pas inspecter avant l'évaluation finale.