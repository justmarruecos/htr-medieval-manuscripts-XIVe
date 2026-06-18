# CONVENTIONS NLP — Pipeline HTR Medieval Manuscripts XIVe

Document de référence pour les choix NLP du Volet 2 (branche `feature/nlp-pipeline`).

---

## 1. Normalisation orthographique (NLP-A)

**Module** : `src/normalisation.py`  
**CER modification** : 2.51% sur 200 lignes val (seuil < 10% ✅)

### Ordre des transformations (CRITIQUE)

1. `expand_abbreviations()` — AVANT NFD (sinon ñ→nn, ꝵ→rum cassés)
2. `normalize_unicode()` — NFD + nettoyage espaces
3. `normalize_uv()` — u initial devant voyelle → v (règle médiale désactivée)
4. `normalize_ij()` — j initial devant a/o/u → i (PAS devant e/i)
5. `normalize_punctuation()` — ·→espace, ¶→.

### Choix justifiés

| Règle | Décision | Justification |
|---|---|---|
| u/v médial | Désactivé | Trop de faux positifs sur HIMANIS |
| j devant e/i | Conservé | Formes comme `je`, `ji` valides en XIVe |
| `sr`→`sire` | Supprimé | Faux positif sur `desrober` |
| Normalisation lexicale | Hors périmètre | Nécessite mT5+LoRA (travail futur) |

---

## 2. NER zéro-shot (NLP-B)

**Module** : `src/ner.py`  
**Modèle** : `Jean-Baptiste/camembert-ner`  
**Précision estimée** : ~50% global (PER=60%, LOC=35%)

### Types d'entités supportés

| Type | Description | Précision estimée |
|---|---|---|
| PER | Personnes (noms médiévaux) | 60% |
| LOC | Lieux (villes, régions) | 35% |
| ORG | Organisations | non évalué |
| MISC | Divers | non évalué |

### Limitations connues

- Pas de type DATE (absent du modèle zéro-shot)
- Faux positifs LOC sur mots médiévaux : `ioye`, `espaulles`, `cuer`
- Fine-tuning requis pour F1 > 0.65 (nécessite gold labels annotés)

---

## 3. Topic Modeling (NLP-C)

**Module** : `experiments/05_nlp_topic_modeling.ipynb`  
**Modèle** : BERTopic v4 — KMeans(5) + paraphrase-multilingual-mpnet-base-v2

### Topics identifiés

| Topic | Label | Docs | Mots-clés |
|---|---|---|---|
| 0 | Littérature courtoise | 11 | li rois, escu, liure, roi |
| 1 | Actes royaux / chancellerie | 7 | nostredit, supplicacion, lesdiz |
| 2 | Textes hagiographiques | 6 | iesu crist, euesque, moine |
| 3 | Chroniques historiques | 6 | bertrand, ville, ans |
| 4 | Textes moraux / didactiques | 4 | vie, ainssi, lautre |

### Choix techniques justifiés

- **KMeans** à la place de HDBSCAN : corpus trop petit (34 docs) pour clustering densité
- **Stopwords moyen français** : liste de 60+ mots grammaticaux ajoutée au CountVectorizer
- **n_components=3** pour UMAP : meilleure séparation sur petit corpus

---

## 4. Export TEI-XML

**Module** : `src/tei_export.py`

### Balises utilisées

| Entité NER | Balise TEI |
|---|---|
| PER | `<persName type="PER">` |
| LOC | `<placeName type="LOC">` |
| ORG | `<orgName type="ORG">` |
| MISC | `<name type="MISC">` |

### Structure du document

```xml
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>...</title></titleStmt>
      <sourceDesc><msDesc><msIdentifier>shelfmark</msIdentifier></msDesc></sourceDesc>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <p>texte avec <persName type="PER">Jehan Rousseau</persName> ...</p>
      </div>
    </body>
  </text>
</TEI>
```

---

## 5. Travaux futurs

- Fine-tuning NER avec LoRA sur gold labels annotés (cible F1 > 0.65)
- Normalisation lexicale avec mT5+LoRA (`doiz`→`dois`, `sces`→`sais`)
- Résolution de coréférence pour fusionner mentions d'une même entité
- Graphe de connaissances (NetworkX + JSON-LD)
