# Pipeline HTR + NLP pour la transcription automatique de manuscrits médiévaux français (XIVe-XVe siècles)

Master Data/IA, HETIC, 2026  
**Dépôt** : https://github.com/justmarruecos/htr-medieval-manuscripts-XIVe

---

## Résumé

La transcription automatique de manuscrits médiévaux représente un défi majeur pour les humanités numériques : des millions de pages numérisées restent inaccessibles à la recherche plein texte faute de transcriptions exploitables. Ce travail présente un pipeline de bout en bout combinant reconnaissance de texte manuscrit (HTR) et traitement automatique du langage (NLP) sur un corpus de manuscrits français des XIVe et XVe siècles.

Le corpus d'entraînement réunit 33 510 lignes issues de CATMuS Medieval et HIMANIS-Guérin (CC-BY 4.0), partitionnées par manuscrit via GroupShuffleSplit pour garantir l'absence de contamination. Le pipeline HTR repose sur le fine-tuning du modèle Kraken cremma-medieval avec un taux d'apprentissage de 1e-4, atteignant une val_metric de 0.7898 contre 0.3048 en zéro-shot, soit un gain de 48.5 points. Le pipeline NLP ajoute une normalisation orthographique par règles (CER de modification : 2.51%), une extraction d'entités nommées zéro-shot par CamemBERT-NER, et une modélisation thématique BERTopic identifiant 5 topics cohérents sur 34 documents agrégés. L'ensemble est exposé via une API FastAPI et un export TEI-XML conforme aux standards des humanités numériques. L'intégralité du code, des données et des modèles est publiée sous licence ouverte.

---

## Introduction

En 2026, la Bibliothèque nationale de France recense près de 380 000 manuscrits dans ses collections, dont environ 11 millions de documents accessibles en ligne via Gallica. À l'échelle mondiale, plusieurs centaines de millions de pages ont été numérisées et diffusées grâce aux API IIIF. Pourtant, pour un chercheur, cette masse documentaire reste en grande partie inaccessible : une image n'est pas un texte. La recherche plein texte, l'extraction d'entités nommées ou l'analyse statistique supposent des transcriptions, et la transcription manuelle par des paléographes — facturée autour de 50 €/h — est hors de portée à cette échelle.

C'est dans ce contexte qu'est née la recherche en reconnaissance automatique de l'écriture manuscrite (HTR, *Handwritten Text Recognition*). Des projets comme CREMMA, CATMuS et HIMANIS ont constitué des corpus annotés de référence pour le français médiéval, permettant le fine-tuning de modèles génériques sur des écritures spécifiques. Malgré ces avancées, un verrou demeure : chaque manuscrit présente ses propres abréviations, graphies et spécificités linguistiques, rendant difficile la généralisation des modèles entraînés sur un corpus homogène.

Ce travail présente un pipeline de bout en bout articulé en deux volets. Le Volet 1 couvre la chaîne HTR : constitution d'un corpus de 33 510 lignes issues de CATMuS Medieval et HIMANIS-Guérin, prétraitement des images (deskew, CLAHE, binarisation Sauvola), segmentation par Kraken BLLA, et fine-tuning du modèle cremma-medieval. Le Volet 2 prolonge ce travail par une analyse linguistique : normalisation orthographique par règles du moyen français, extraction d'entités nommées par CamemBERT-NER en mode zéro-shot, et modélisation thématique par BERTopic sur les documents agrégés par manuscrit.

Les contributions principales de ce travail sont les suivantes : (1) un pipeline HTR reproductible atteignant une val_metric de 0.7898 (+48.5 points sur la baseline zéro-shot) ; (2) un module de normalisation orthographique avec un CER de modification de 2.51% ; (3) une identification de 5 topics thématiquement cohérents sur un corpus de 34 manuscrits ; (4) une API REST et un export TEI-XML conformes aux standards des humanités numériques ; (5) l'ensemble du code et des données publié sous licence CC-BY 4.0.

---

## État de l'art

### Reconnaissance de texte manuscrit (HTR)

La reconnaissance automatique de l'écriture manuscrite a connu des avancées majeures au cours de la dernière décennie. Deux architectures dominent aujourd'hui le domaine : Kraken et TrOCR.

Kraken (Kiessling, 2019) est un moteur HTR open source reposant sur une architecture VGSL (*Variable-size Graph Specification Language*) combinant couches convolutives, pooling et BiLSTM avec une fonction de perte CTC. Son interface eScriptorium facilite la production de vérité terrain et le fine-tuning sur des corpus spécifiques. Le projet CREMMA (Pinche et al., 2022) a démontré l'efficacité de cette approche sur des manuscrits français médiévaux des XIIIe-XVe siècles, produisant des modèles atteignant des CER inférieurs à 5% sur leurs corpus d'entraînement.

TrOCR (Li et al., 2021) adopte une architecture Transformer encoder-decoder : un ViT (*Vision Transformer*) encode l'image de ligne, un décodeur de type GPT-2 génère la transcription token par token. Cette architecture bénéficie du pré-entraînement massif sur des données synthétiques et réelles. Le fine-tuning par LoRA (Hu et al., 2022) permet d'adapter ces modèles à des domaines spécifiques en n'entraînant qu'une fraction des paramètres, réduisant considérablement le coût computationnel.

### Corpus médiévaux

CATMuS Medieval (Clérice et al., 2023) constitue aujourd'hui le corpus de référence pour le français médiéval : plus de 160 000 lignes couvrant 200 manuscrits du VIIIe au XVIIe siècle, produit par la fédération des projets CREMMA, GalliCorpora, HTRomance et DEEDS. Sa licence CC-BY 4.0 et sa normalisation inter-corpus en font la ressource de choix pour l'entraînement de modèles génériques.

HIMANIS (Stutzmann et al., 2019) a fourni les premiers grands corpus annotés de registres de chancellerie royale française (registres JJ de la série des Archives nationales), combinant des écritures cursives denses et un vocabulaire juridico-administratif spécifique. Les 7 698 lignes du sous-corpus Guérin (JJ207-JJ210) utilisées dans ce travail représentent ce type documentaire.

### Métriques d'évaluation

Le CER (*Character Error Rate*), calculé comme la distance de Levenshtein normalisée par la longueur de la référence, constitue la métrique principale en HTR. Le WER (*Word Error Rate*) est plus sensible aux erreurs lexicales. L'accord inter-annotateurs (IAA), mesuré par le CER entre deux transcriptions humaines du même document, fournit un plafond de performance théorique : en deçà de ce seuil, les erreurs du modèle sont indiscernables de l'ambiguïté inhérente au document.

### Traitement automatique du langage sur textes médiévaux

La normalisation orthographique du moyen français pose des défis spécifiques : alternances graphiques systématiques (u/v, i/j), abréviations scribales, absence de standardisation orthographique. Des approches par règles (Souvay & Pierrel, 2009, via le DMF) couvrent 60 à 75% des divergences sans recours à des modèles neuronaux. Les approches neuronales — notamment mT5 fine-tuné par LoRA — permettent de couvrir les cas non traités par les règles au prix d'un corpus de paires annotées.

Pour la NER sur textes médiévaux, CamemBERT (Martin et al., 2020) et ses variantes fine-tunées constituent l'état de l'art en français. En mode zéro-shot, des précisions de 50 à 60% sur les entités de type PER sont rapportées dans la littérature, le type LOC étant plus difficile en raison des nombreux faux positifs liés aux formes graphiques médiévales.

La modélisation thématique par BERTopic (Grootendorst, 2022) combine des embeddings de phrases denses, une réduction de dimensionnalité par UMAP et un clustering par HDBSCAN ou KMeans, surpassant LDA sur les corpus courts où les sacs de mots sont trop creux pour capturer la sémantique.

---

## Données

### Sources et licences

Le corpus de ce travail est constitué de deux sources complémentaires, toutes deux sous licence CC-BY 4.0, garantissant la conformité avec la contrainte de réutilisabilité ouverte.

**CATMuS Medieval** (Clérice et al., 2023) est téléchargé via l'API HuggingFace (`CATMuS/medieval`). Nous retenons les entrées en français des XIVe et XVe siècles, soit 25 812 lignes couvrant 33 manuscrits distincts. Ce sous-corpus représente une grande diversité de types d'écritures (gothique textura, cursive, bâtarde) et de genres documentaires (littérature courtoise, textes hagiographiques, chroniques).

**HIMANIS-Guérin** (Stutzmann et al., 2019, DOI: 10.5281/zenodo.5535306) fournit 7 698 lignes extraites des registres de chancellerie royale JJ207 et JJ210 des Archives nationales. Les coordonnées spatiales (polygones PAGE-XML) sont extraites par parsing BeautifulSoup depuis les fichiers XML d'annotation. Ce sous-corpus apporte des écritures cursives denses et un vocabulaire juridico-administratif absent de CATMuS.

Les corpus e-NDP (licence CC-BY-NC-SA, incompatible avec la contrainte #7) et CREMMA-BS4 (doublon de BnF fr.22549 déjà présent dans CATMuS) ont été explicitement écartés.

### Volume et distribution

| Source | Lignes | Manuscrits | Siècle |
|---|---|---|---|
| CATMuS Medieval | 25 812 | 33 | XIVe (28%) + XVe (72%) |
| HIMANIS-Guérin | 7 698 | 2 (JJ207, JJ210) | XIVe |
| **Total** | **33 510** | **35** | XIVe (28%) + XVe (72%) |

La distribution temporelle reflète le déséquilibre inhérent aux corpus disponibles : le XVe siècle est surreprésenté (72%) en raison du volume plus important de CATMuS pour cette période. Ce biais est documenté et discuté en section Discussion.

### Partitionnement

Le corpus est partitionné en trois ensembles disjoints selon le shelfmark (cote du manuscrit), garantissant qu'aucun manuscrit n'apparaît dans deux ensembles différents. Cette approche par `GroupShuffleSplit` (scikit-learn) prévient la fuite de données liée à la corrélation entre lignes d'un même manuscrit.

| Split | Lignes | Proportion |
|---|---|---|
| Train | 23 444 | 69.9% |
| Validation | 5 954 | 17.8% |
| Test | 4 112 | 12.3% |

Les seeds ont été sélectionnés par recherche multi-objectif (gss1=47, gss2=57) pour optimiser simultanément les proportions globales et l'équilibre par siècle entre les splits. Le test set est scellé depuis la constitution du corpus et n'a pas été consulté pendant le développement.

**SHA-256 des splits** :
- `train.csv` : `4b30cdb9aece87cac60d835986e0e5bfa331c8c47b1ccd72d473796d882325e3`
- `val.csv`   : `592f2e69fb5df7ecb5f7225d012352c32abee860ac13a276cd8bd2159045668e`
- `test.csv`  : `3df155b380d8316c29b0f758192fd71dcb9e6f6620e42c090d9f4331cf0a6f08`

### Conventions de transcription

Les conventions suivent le niveau semi-diplomatique : les abréviations sont conservées dans leur forme graphique originale (non développées), les alternances u/v et i/j sont transcrites telles qu'elles apparaissent dans le manuscrit. Les lacunes sont marquées par `[...]`. Ces conventions sont documentées intégralement dans `CONVENTIONS_TRANSCRIPTION.md`.

---

## Méthodes

### Vue d'ensemble du pipeline

Le pipeline est structuré en deux volets séquentiels. Le Volet 1 produit des transcriptions textuelles à partir d'images de manuscrits numérisés. Le Volet 2 prend ces transcriptions en entrée pour en extraire des informations linguistiques structurées. Les deux volets partagent le même corpus et le même schéma de données (*data contract* JSON).

### Volet 1 — HTR

#### Prétraitement des images

Trois transformations sont appliquées séquentiellement à chaque image de ligne :

1. **Correction d'inclinaison (deskew)** : détection de l'angle dominant par transformée de Hough, rotation corrective. Paramètre : angle maximal 10°.

2. **Amélioration du contraste (CLAHE)** : *Contrast Limited Adaptive Histogram Equalization* avec `clipLimit=2.0` et `tileGridSize=(8,8)`. Cette paramétrie évite la sur-amplification du bruit sur les zones d'encre pâle tout en améliorant la lisibilité des zones denses.

3. **Binarisation adaptative (Sauvola)** : algorithme de seuillage local avec fenêtre de 25 pixels. Deux valeurs de sensibilité `k` ont été testées : `k=0.1` pour les manuscrits à encre pâle (CATMuS), `k=0.2` pour les registres à grands espaces vides (HIMANIS) afin d'éviter la génération de bruit.

#### Segmentation

La segmentation des lignes de texte utilise Kraken BLLA (*Baseline Layout Analysis*) en mode zéro-shot. BLLA détecte les lignes de base et produit les polygones englobants de chaque ligne. Les résultats sont exportés au format PAGE-XML (`segmentations/`), conforme au standard de la communauté des humanités numériques.

L'évaluation de la segmentation utilise l'IoU (*Intersection over Union*) entre les polygones prédits et une vérité terrain de 40 lignes annotées manuellement sur la page `FRAN_0021_33533_A.jpg` (HIMANIS JJ207).

#### Fine-tuning HTR

Le modèle de base `cremma-medieval.mlmodel` (Kraken VGSL, 4.0M paramètres) est fine-tuné sur le split d'entraînement via `ketos train`. Deux runs ont été conduits :

- **Run v1** (lrate=1e-3) : convergence incomplète, val_metric plafonnant à 0.2402 à l'epoch 6.
- **Run v2** (lrate=1e-4) : convergence nette, val_metric de 0.7898 à l'epoch 18.

Un bug identifié dans Kraken 7.0.2 empêche le CLI `ketos train -f binary` de transmettre correctement `binary_dataset_split=True`. Le contournement consiste à injecter directement les datasets via l'API Python (`VGSLRecognitionDataModule.train_set`/`val_set`).

L'arrêt de l'entraînement du run v2 est manuel (plateau détecté), l'early stopping avec patience=5 aurait déclenché 1 à 2 epochs plus tard.

### Volet 2 — NLP

#### Normalisation orthographique

La normalisation suit un pipeline de règles déterministes appliquées dans un ordre critique pour éviter les interactions entre transformations :

1. `expand_abbreviations()` — résolution des abréviations scribales (ñ→nn, ꝵ→rum) avant la décomposition NFD
2. `normalize_unicode()` — décomposition NFD et nettoyage des espaces
3. `normalize_uv()` — u initial devant voyelle → v (règle médiale désactivée pour limiter les faux positifs)
4. `normalize_ij()` — j initial devant a/o/u → i (conservé devant e/i)
5. `normalize_punctuation()` — ·→espace, ¶→point

Le CER de modification (distance de Levenshtein entre texte brut et texte normalisé, normalisée par la longueur) mesure l'impact de la normalisation sur le corpus de validation. Un CER de modification inférieur à 10% est défini comme seuil de validation pour garantir que la normalisation ne dégrade pas les transcriptions existantes.

#### Extraction d'entités nommées

L'extraction d'entités nommées utilise le modèle `Jean-Baptiste/camembert-ner` en mode zéro-shot via le module `src/ner.py`. Ce modèle, fine-tuné sur le corpus WikiNER en français, supporte quatre types d'entités : PER (personnes), LOC (lieux), ORG (organisations), MISC (divers). L'agrégation des tokens en spans utilise la stratégie `simple` de HuggingFace Transformers.

La classe `NERPipeline` implémente trois méthodes : `predict(text)` pour un texte unique, `predict_batch(texts)` pour le traitement par lots, et `evaluate(texts, gold_labels)` calculant précision, rappel et F1 au niveau des spans exacts.

#### Modélisation thématique

Les 33 510 lignes du corpus (train + validation) sont agrégées par shelfmark, produisant 34 documents d'une taille moyenne de 48 192 caractères. BERTopic est appliqué sur ces documents avec les paramètres suivants :

- **Embeddings** : `paraphrase-multilingual-mpnet-base-v2` (768 dimensions)
- **Réduction de dimensionnalité** : UMAP avec `n_neighbors=5`, `n_components=3`, métrique cosinus
- **Clustering** : KMeans avec `n_clusters=5` (HDBSCAN échoue sur 34 documents, corpus trop petit pour le clustering par densité)
- **Représentation** : KeyBERTInspired + CountVectorizer avec liste de 60+ stopwords du moyen français

#### Export TEI-XML et API

Le module `src/tei_export.py` convertit un texte normalisé et ses entités NER en document TEI-XML conforme, avec injection des balises `<persName>`, `<placeName>`, `<orgName>` aux offsets caractères correspondants et ancrage du manuscrit source via `<msIdentifier>`.

L'API REST (`src/api.py`, FastAPI) expose deux endpoints : `POST /analyze` applique la normalisation puis la NER sur un texte brut et retourne un objet JSON structuré ; `GET /health` retourne le statut de l'API. Le service est conteneurisé via Docker avec pré-téléchargement du modèle NER au build.

---

## Résultats

### Volet 1 — HTR

#### Segmentation (Kraken BLLA, zéro-shot)

L'évaluation de la segmentation est conduite sur 40 lignes annotées manuellement
sur la page `FRAN_0021_33533_A.jpg` (registre JJ207, HIMANIS-Guérin).

| Métrique | Valeur |
|---|---|
| IoU moyen | 0.604 |
| IoU médian | 0.614 |
| Lignes IoU > 0.75 | 0/40 |
| Lignes IoU > 0.85 | 0/40 |

L'IoU moyen de 0.604 est en dessous du seuil de validation fixé à 0.75. Ce résultat
s'explique par la nature cursive dense de l'écriture des registres de chancellerie,
pour lesquels le modèle BLLA zéro-shot produit des polygones trop larges englobant
partiellement les lignes adjacentes. Un fine-tuning de la segmentation via
`ketos segtrain` sur les annotations HIMANIS constitue une piste d'amélioration directe.

#### Reconnaissance HTR

| Modèle | val_metric | CER estimé | Epoch |
|---|---|---|---|
| cremma-medieval (zéro-shot) | 0.3048 | 69.5% | — |
| Kraken fine-tuné lrate=1e-3 (v1) | 0.2402 | — | 6 |
| Kraken fine-tuné lrate=1e-4 (v2) | **0.7898** | **~21%** | 18 |

Le run v2 (lrate=1e-4) dépasse le seuil de validation fixé à 0.75, avec un gain
de 48.5 points d'accuracy sur la baseline zéro-shot. La courbe d'apprentissage
montre une progression régulière : 0.1531 à l'epoch 0, 0.5531 à l'epoch 1,
0.7414 à l'epoch 6, 0.7898 à l'epoch 18. Le plateau observé à partir de l'epoch
15 justifie l'arrêt manuel à l'epoch 19.

La comparaison des runs v1 et v2 illustre la sensibilité critique du fine-tuning
Kraken au taux d'apprentissage : un lrate dix fois plus élevé conduit à une
convergence incomplète malgré un nombre d'epochs similaire.

### Volet 2 — NLP

#### Normalisation orthographique

| Métrique | Valeur | Seuil | Statut |
|---|---|---|---|
| CER modification moyen | 2.51% | < 10% | ✅ VALIDE |
| Lignes modifiées | 122/200 (61%) | — | — |
| Tests pytest | 22/22 | — | ✅ |

Sur 200 lignes du set de validation, 122 (61%) sont modifiées par au moins une
règle de normalisation. Le CER de modification de 2.51% confirme que les
transformations sont ciblées et non destructives. Les règles les plus actives
sont l'expansion des abréviations (ñ→nn, ꝵ→rum) et la normalisation u/v.

Deux bugs ont été identifiés et corrigés pendant le développement : l'abréviation
`sr`→`sire` générait des faux positifs sur des formes comme `desrober` ; la règle
i/j ne devait pas s'appliquer devant e/i pour préserver des formes valides comme `je`.

#### NER zéro-shot (CamemBERT)

| Métrique | Valeur |
|---|---|
| Entités détectées (200 lignes val normalisées) | 215 |
| PER | 143 (66.5%) |
| LOC | 54 (25.1%) |
| MISC | 14 (6.5%) |
| ORG | 4 (1.9%) |
| Score de confiance moyen | 0.880 |
| Score > 0.8 | 76.3% des entités |
| Précision estimée PER (évaluation manuelle) | ~60% |
| Précision estimée LOC (évaluation manuelle) | ~35% |
| Précision globale estimée | ~50% |

Les entités PER bien détectées incluent des noms médiévaux attestés : *Jehan
Rousseau*, *Pierre Norrisson*, *Denis du Vergier*, *Jehan Bouffinière*. Les
principaux faux positifs LOC concernent des mots médiévaux dont la forme
graphique ressemble à des toponymes : `ioye` (joie), `espaulles` (épaules),
`cuer` (cœur), `veneour` (veneur).

L'absence du type DATE dans le modèle zéro-shot constitue une limitation
majeure pour l'analyse des actes royaux, où les dates sont des entités
structurantes essentielles.

#### Topic Modeling (BERTopic)

Sur 34 documents agrégés par shelfmark, BERTopic avec KMeans(5) identifie
5 topics thématiquement cohérents :

| Topic | Label | Docs | Mots-clés représentatifs |
|---|---|---|---|
| 0 | Littérature courtoise | 11 | li rois, escu, liure, roi, lor |
| 1 | Actes royaux / chancellerie | 7 | nostredit, supplicacion, lesdiz, avecques |
| 2 | Textes hagiographiques | 6 | iesu crist, nostre seigneur, euesque, moine |
| 3 | Chroniques historiques | 6 | bertrand, ville, auant, ans |
| 4 | Textes moraux / didactiques | 4 | vie, ainssi, lautre, pource |

La cohérence thématique est validée par recoupement avec les métadonnées
des manuscrits : les registres JJ207 et JJ210 (HIMANIS) sont correctement
classés dans le Topic 1 (Actes royaux), et le manuscrit Bruxelles KBR 9232
(hagiographique) dans le Topic 2. La cible de ≥ 3 topics cohérents est atteinte
avec 5 topics, dépassant les attentes initiales.

### Tests automatisés

| Suite | Tests | Résultat |
|---|---|---|
| test_preprocessing.py | 16 | ✅ PASSED |
| test_normalisation.py | 22 | ✅ PASSED |
| test_ner.py | 15 | ✅ PASSED |
| **Total** | **53** | ✅ **53/53 PASSED** |

---

## Discussion

### Biais de représentation du corpus

Le corpus présente plusieurs biais de représentation qu'il convient de documenter
explicitement. Sur le plan temporel, le XVe siècle est surreprésenté (72%) par
rapport au XIVe (28%), reflet du volume plus important de manuscrits numérisés
et annotés pour cette période dans CATMuS. Ce déséquilibre peut conduire à une
meilleure performance du modèle HTR sur les écritures du XVe siècle (bâtarde,
cursive tardive) que sur celles du XIVe (gothique textura, précotonelle).

Sur le plan géographique, le corpus est très majoritairement centré sur la
production parisienne et de l'Île-de-France : les registres HIMANIS proviennent
de la chancellerie royale, et CATMuS, bien que plus diversifié, reste dominé
par des manuscrits conservés à la BnF. Les dialectes régionaux (picard, normand,
champenois, francien périphérique) sont sous-représentés, ce qui limite la
portée du modèle sur des corpus dialectaux spécifiques.

Sur le plan typologique, deux grandes catégories dominent : la littérature
courtoise/hagiographique (CATMuS) et les registres de chancellerie (HIMANIS).
Les manuscrits scientifiques, médicaux, ou à structure tabulaire complexe sont
absents du corpus d'entraînement, constituant un angle mort du modèle.

### Limitations du pipeline HTR

L'IoU de segmentation de 0.604 en zéro-shot reste en dessous du seuil de
validation de 0.75. Ce résultat illustre la difficulté de Kraken BLLA à gérer
les écritures cursives très denses des registres de chancellerie, où les
interlignes sont réduits et les hastes descendent parfois sur la ligne suivante.
Un fine-tuning de la segmentation via `ketos segtrain` sur les 7 698 lignes
HIMANIS annotées constituerait une amélioration directe et mesurable.

La val_metric de 0.7898 du modèle HTR, bien qu'au-dessus du seuil de validation,
reste éloignée du seuil d'excellence fixé à 0.85. L'analyse des erreurs résiduelles
révèle trois catégories principales : les caractères rares (hapax de l'alphabet
médiéval, présents dans moins de 10 occurrences dans le corpus d'entraînement),
les lignes très courtes (moins de 5 caractères, difficiles à contextualiser),
et les zones dégradées (encre pâlie, parchemin abîmé).

### Limitations du pipeline NLP

La normalisation orthographique par règles couvre 61% des lignes avec un CER
de modification de 2.51%, validant l'approche pour une utilisation en production.
Cependant, la normalisation reste graphique (standardisation des conventions
d'écriture) et non lexicale : des formes comme `doiz`→`dois` ou `sces`→`sais`
nécessiteraient un modèle séquence-à-séquence (mT5+LoRA) entraîné sur des paires
de normalisation annotées, constituant un travail futur.

La précision NER de ~50% en mode zéro-shot est insuffisante pour une utilisation
en production (seuil F1 > 0.65 fixé par le syllabus). Cette limitation s'explique
par trois facteurs : (1) l'absence d'adaptation du modèle au domaine médiéval,
(2) l'absence du type DATE dans camembert-ner, pourtant essentiel pour les actes
royaux, (3) la proximité graphique entre mots médiévaux courants et entités
nommées (faux positifs LOC). Le fine-tuning de CamemBERT avec LoRA sur des
données annotées en IOB constitue la voie d'amélioration principale, mais
nécessite la constitution préalable d'un corpus gold labels — effort estimé à
plusieurs centaines d'heures d'annotation.

Le topic modeling BERTopic sur 34 documents agrégés produit des topics
thématiquement cohérents, mais la granularité est limitée par le petit nombre
de documents. Avec un corpus plus large (plusieurs centaines de manuscrits),
des distinctions plus fines seraient atteignables : séparation des sous-genres
littéraires, distinction dialectale, évolution chronologique des thèmes.

### Comparaison avec l'état de l'art

La val_metric de 0.7898 obtenue avec Kraken fine-tuné est comparable aux
résultats publiés sur des corpus similaires : Pinche et al. (2022) rapportent
des CER de 3 à 8% sur des manuscrits CREMMA avec fine-tuning, notre modèle
atteignant un CER estimé de ~21% sur un corpus plus hétérogène (mélange
littéraire + chancellerie). La différence s'explique principalement par la
diversité plus grande de notre corpus d'entraînement et l'absence de fine-tuning
de la segmentation.

### Reproductibilité et ouverture

L'ensemble du pipeline est conçu pour la reproductibilité : seeds fixés, SHA-256
des splits publiés, versions de toutes les dépendances figées dans
`requirements.txt`, journal d'expériences `experiments/journal.jsonl` documentant
les 10 runs principaux. Le code, les données et les conventions éditoriales sont
publiés sous licence CC-BY 4.0 sur GitHub, conformément aux principes FAIR
(*Findable, Accessible, Interoperable, Reusable*).

---

## Conclusion et travaux futurs

### Conclusion

Ce travail présente un pipeline de bout en bout pour la transcription automatique
et l'analyse linguistique de manuscrits médiévaux français des XIVe et XVe siècles.
Le Volet 1 HTR atteint une val_metric de 0.7898 (+48.5 points sur la baseline
zéro-shot) grâce au fine-tuning du modèle Kraken cremma-medieval avec un taux
d'apprentissage de 1e-4 sur un corpus de 33 510 lignes (CATMuS + HIMANIS-Guérin).
Le Volet 2 NLP produit une normalisation orthographique avec un CER de modification
de 2.51%, une extraction d'entités nommées exploratoire par CamemBERT-NER, et
une modélisation thématique BERTopic identifiant 5 topics cohérents sur 34
documents agrégés par manuscrit.

L'ensemble du pipeline est reproductible, documenté et publié sous licence
CC-BY 4.0. Il constitue une base solide pour des travaux ultérieurs en humanités
numériques, notamment pour l'enrichissement de bases de données patrimoniales
et la mise à disposition de corpus annotés pour la communauté de recherche.

Les contributions principales sont : (1) un pipeline HTR reproductible dépassant
le seuil de validation fixé ; (2) un module de normalisation orthographique du
moyen français validé quantitativement ; (3) une identification thématique
cohérente sur un corpus hétérogène ; (4) une API REST et un export TEI-XML
conformes aux standards des humanités numériques ; (5) 53 tests automatisés
garantissant la non-régression du pipeline.

### Travaux futurs

Plusieurs axes d'amélioration sont identifiés à l'issue de ce travail :

**Fine-tuning NER avec LoRA.** La précision NER de ~50% en zéro-shot est
insuffisante pour une exploitation en production. Le fine-tuning de CamemBERT
avec LoRA sur un corpus gold labels annoté en schéma IOB constitue la priorité.
La cible est un F1 micro > 0.65 sur les types PER, LOC et DATE. La constitution
du corpus d'annotation (estimée à 200-300 heures) représente le principal
obstacle.

**Normalisation lexicale par mT5+LoRA.** La normalisation actuelle est graphique
(standardisation des conventions d'écriture). Une normalisation lexicale
(`doiz`→`dois`, `sces`→`sais`) nécessite un modèle séquence-à-séquence fine-tuné
sur des paires (forme médiévale, forme moderne) issues du Dictionnaire du Moyen
Français (DMF) et de son lemmatiseur LGeRM.

**Fine-tuning de la segmentation.** L'IoU de 0.604 en zéro-shot est en dessous
du seuil de validation. Un fine-tuning de Kraken BLLA via `ketos segtrain` sur
les 7 698 lignes HIMANIS annotées permettrait d'améliorer significativement la
qualité de la segmentation sur les écritures cursives denses.

**Graphe de connaissances.** Les entités NER et les topics BERTopic constituent
les briques d'un graphe de connaissances reliant personnes, lieux, dates et
thèmes à travers les manuscrits. L'export en JSON-LD (NetworkX) et l'intégration
dans une interface de requête SPARQL représentent une valorisation naturelle du
pipeline.

**Extension du corpus.** L'ajout de manuscrits dialectaux (picard, normand,
champenois) et de types documentaires sous-représentés (manuscrits scientifiques,
médicaux, tabulaires) améliorerait la robustesse et la portée du modèle HTR.

**Comparaison TrOCR+LoRA.** Une comparaison systématique entre Kraken VGSL et
TrOCR fine-tuné par LoRA sur le même corpus, avec test de McNemar sur les
erreurs différentielles, apporterait une validation scientifique supplémentaire
et permettrait de bénéficier du bonus d'évaluation prévu par le syllabus.

---

## Références

Camps, J.-B., Pinche, A., & Vidal-Gorène, C. (2021). *Handling Heavily Abbreviated
Manuscripts: HTR engines vs text normalisation approaches*. In *Proceedings of
the 6th International Workshop on Historical Document Imaging and Processing*
(pp. 1–6). ACM.

Clérice, T., Chagué, A., Gille-Levenson, M., Brisville-Fertin, O., Pinche, A.,
Camps, J.-B., Fischer, F., Boschetti, F., Guadagnini, E., Guilhem Couffignal, G.,
Canteaut, O., Romary, L., Reboul, M., Perreaux, N., Poibeau, T., Smith, M.,
Norindr, J., Glaise, A., Navas Farré, M., Bordier, J., Leroy, N., Alba, R., &
Rubin, G. (2023). *CATMuS Medieval: A multilingual large-scale cross-century
dataset in Latin script for handwritten text recognition and beyond* [Data set].
Zenodo. https://doi.org/10.5281/zenodo.8241146

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based
TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.
https://doi.org/10.48550/arXiv.2203.05794

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., &
Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In
*Proceedings of ICLR 2022*. https://doi.org/10.48550/arXiv.2106.09685

Kiessling, B. (2019). Kraken: An universal text recognizer for the humanities.
In *Digital Humanities Conference 2019*. Utrecht.
https://doi.org/10.5281/zenodo.3265754

Kiessling, B., Stoekl Ben Ezra, D., & Miller, M. T. (2019). BADAM: A public
dataset for baseline detection in Arabic-script manuscripts. In *Proceedings of
the 5th International Workshop on Historical Document Imaging and Processing*.
ACM. https://doi.org/10.1145/3352631.3352648

Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., Zhang, C., Li, Z., &
Wei, F. (2021). TrOCR: Transformer-based optical character recognition with
pre-trained models. *arXiv preprint arXiv:2109.10282*.
https://doi.org/10.48550/arXiv.2109.10282

Martin, L., Muller, B., Ortiz Suárez, P. J., Dupont, Y., Romary, L., de la
Clergerie, É., Seddah, D., & Sagot, B. (2020). CamemBERT: a tasty French
language model. In *Proceedings of ACL 2020* (pp. 7203–7219). Association for
Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.645

McNemar, Q. (1947). Note on the sampling error of the difference between
correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
https://doi.org/10.1007/BF02295996

Pinche, A. (2022). *Transcription guidelines for 10th to 15th century manuscripts*
[Technical report]. HAL. https://hal.science/hal-03697382

Pinche, A., Camps, J.-B., Mariotti, V., Nolibois, A., Carnaille, C.,
Deleville, P., Lecomte, S., Meylan, A., Ventura, S., & Dugaz, L. (2022).
*CREMMA Medieval* [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.6331518

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using
Siamese BERT-networks. In *Proceedings of EMNLP 2019* (pp. 3982–3992).
https://doi.org/10.18653/v1/D19-1410

Souvay, G., & Pierrel, J.-M. (2009). LGeRM : Lemmatisation des mots en Moyen
Français. *Traitement Automatique des Langues*, 50(2), 149–172.

Stutzmann, D., Kermorvant, C., Vidal, E., Chanda, S., Hamel, S., Puigcerver,
J., Schomaker, L., & Toselli, A. H. (2019). *HIMANIS: Indexing and searching
the registers of the French royal chancery* [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.5535306

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances
in Neural Information Processing Systems* (Vol. 30). Curran Associates.
https://doi.org/10.48550/arXiv.1706.03762

---

## Annexes

### Annexe A — Exemples de transcriptions

#### Exemple 1 — Manuscrit littéraire (CATMuS, BnF fr. 146)

**Image** : ligne de base extraite par Kraken BLLA

**Transcription brute (HTR v2)** :
Qant el vit sõ escu ꝑcie Lo cheual par laresne tint

**Transcription normalisée (NLP-A)** :
Quant el vit son escu percie Lo cheual par laresne tint

**Entités NER détectées** : aucune (texte narratif sans anthroponymes)

---

#### Exemple 2 — Registre de chancellerie (HIMANIS, JJ207)

**Transcription brute (HTR v2)** :
Jehan Rousseau demourant en la ville de Paris

**Transcription normalisée (NLP-A)** :
Jehan Rousseau demourant en la ville de Paris

**Entités NER détectées** :
- `Jehan Rousseau` → PER (score : 0.94)
- `Paris` → LOC (score : 0.91)

---

#### Exemple 3 — Faux positif LOC typique

**Texte normalisé** :
et la ioye de son cuer estoit grant

**Entités NER détectées (incorrectes)** :
- `ioye` → LOC (score : 0.81) — faux positif, signifie "joie"
- `cuer` → LOC (score : 0.78) — faux positif, signifie "cœur"

*Ces exemples illustrent la limitation principale du NER zéro-shot sur textes
médiévaux : la proximité graphique entre mots du lexique courant et toponymes.*

---

### Annexe B — Schéma du data contract

Le data contract JSON (`data_contract.json`) définit le schéma de validation
des transcriptions produites par le pipeline HTR. Les champs principaux sont :

| Champ | Type | Description |
|---|---|---|
| `id` | string | Identifiant unique de la ligne |
| `shelfmark` | string | Cote du manuscrit source |
| `text` | string | Transcription brute HTR |
| `text_normalized` | string | Transcription normalisée (NLP-A) |
| `confidence` | float | Score de confiance HTR [0,1] |
| `needs_review` | boolean | Flag lignes incertaines |
| `polygon_ref` | list | Coordonnées polygone PAGE-XML |
| `ner_spans` | list | Entités NER détectées |
| `century` | string | Siècle du manuscrit (14/15) |
| `split` | string | Appartenance train/val/test |

---

### Annexe C — Progression de l'entraînement HTR (Run v2)

| Epoch | val_metric |
|---|---|
| 0 | 0.1531 |
| 1 | 0.5531 |
| 2 | 0.6349 |
| 6 | 0.7414 |
| 7 | 0.7576 |
| 12 | 0.7832 |
| 15 | 0.7880 |
| 18 | **0.7898** |

---

### Annexe D — Distribution des topics BERTopic par manuscrit

| Shelfmark | Topic | Label |
|---|---|---|
| Bern, Burgerbibliothek, 354 | 0 | Littérature courtoise |
| Bruxelles, KBR, 9232 | 2 | Textes hagiographiques |
| Paris, AN, JJ207 | 1 | Actes royaux / chancellerie |
| Paris, AN, JJ210 | 1 | Actes royaux / chancellerie |
| Paris, BnF, Arsenal, 3525 | 0 | Littérature courtoise |
| Paris, BnF, Arsenal, 5070 | 4 | Textes moraux / didactiques |
| Paris, BnF, Arsenal, 5103 | 3 | Chroniques historiques |
| Paris, BnF, NAF 27401 | 3 | Chroniques historiques |
| Paris, BnF, NAF 6213 | 3 | Chroniques historiques |
| Paris, BnF, Rés. J-845 | 4 | Textes moraux / didactiques |
| Paris, BnF, Rés. Y2-82 | 3 | Chroniques historiques |
| Paris, BnF, Rés. Y2-930 | 1 | Actes royaux / chancellerie |
| Paris, BnF, Rés. YE-281 | 2 | Textes hagiographiques |
| Paris, BnF, Velins 488 | 0 | Littérature courtoise |
| Paris, BnF, Velins 611 | 0 | Littérature courtoise |
| Paris, BnF, Velins 690 | 2 | Textes hagiographiques |
| Paris, BnF, Velins 906 | 2 | Textes hagiographiques |
| Paris, BnF, fr. 11610 | 0 | Littérature courtoise |
| Paris, BnF, fr. 12551 | 0 | Littérature courtoise |
| Paris, BnF, fr. 12779 | 0 | Littérature courtoise |
| Paris, BnF, fr. 13568 | 1 | Actes royaux / chancellerie |
| Paris, BnF, fr. 146 | 0 | Littérature courtoise |
| Paris, BnF, fr. 1728 | 0 | Littérature courtoise |
| Paris, BnF, fr. 185 | 4 | Textes moraux / didactiques |
| Paris, BnF, fr. 1984 | 3 | Chroniques historiques |
| Paris, BnF, fr. 22549 | 4 | Textes moraux / didactiques |
| Paris, BnF, fr. 263 | 3 | Chroniques historiques |
| Paris, BnF, fr. 411 | 0 | Littérature courtoise |
| Paris, BnF, fr. 5024 | 1 | Actes royaux / chancellerie |
| Paris, BnF, fr. 619 | 2 | Textes hagiographiques |
| Paris, BnF, fr. 777 | 2 | Textes hagiographiques |
| Paris, BnF, fr. 840 | 1 | Actes royaux / chancellerie |
| Philadelphia, UPenn, Codex 660 | 1 | Actes royaux / chancellerie |
| Vatican, BAV, Reg.lat. 1616 | 0 | Littérature courtoise |