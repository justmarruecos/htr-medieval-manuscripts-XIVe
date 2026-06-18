# Conventions de transcription

Ce document décrit les conventions appliquées aux transcriptions du corpus
HTR médiéval français (XIVe-XVe siècles), conformément aux pratiques des
corpus sources (CATMuS Medieval et HIMANIS-Guérin).

## Principes généraux

Les transcriptions suivent une approche **diplomatique simplifiée** :
fidélité au texte manuscrit sans normalisation orthographique, mais avec
résolution des abréviations courantes.

## Alphabet et encodage

- **Encodage** : UTF-8, normalisation Unicode **NFD** (décomposition canonique)
- **Taille de l'alphabet** : 1 148 caractères distincts (corpus d'entraînement)
- **Caractères spéciaux médiévaux** inclus :
  - Lettres abbréviatives : `ꝵ` (rum), `ꝭ` (us), `Ꝙ` (Q barré)
  - Signes diacritiques : `ͨ` `ͭ` `ͧ` `ͯ` (exposants abréviatifs)
  - Caractères latins étendus : `ħ` `ł` `ẞ` `ẜ`
  - Ponctuation médiévale : `‸` `·` `¶`

## Règles de transcription

### Abréviations
- Les abréviations sont **résolues** et transcrites en toutes lettres
  (ex : `p̃` → `par`, `q̃` → `que`)
- Exception : les signes abréviatifs sans résolution certaine sont
  conservés tels quels avec le caractère Unicode correspondant.

### Majuscules et minuscules
- Respectées telles qu'elles apparaissent dans le manuscrit.
- Pas de normalisation de casse.

### Espaces et ponctuation
- Les espaces sont normalisés (unicode whitespace → U+0020).
- La ponctuation originale est conservée.

### Lignes vides
- Les lignes sans transcription (zones décoratives, numérotation,
  espaces blancs) sont exclues du corpus (224 lignes exclues dans HIMANIS).

## Paramètres de prétraitement

| Étape | Paramètre | Valeur | Justification |
|-------|-----------|--------|---------------|
| Deskew | angle max | ±45° | Correction inclinaison légère |
| CLAHE | clip_limit | 2.0 | Contraste local modéré |
| CLAHE | tile_size | (8, 8) | Grille standard |
| Sauvola | window_size | 25 px | Calibré pour résolution ~300 DPI |
| Sauvola | k | 0.1 (CREMMA) / 0.2 (HIMANIS) | k=0.1 pour encres pâles, k=0.2 pour registres |

**Note** : le prétraitement (deskew → CLAHE → Sauvola) est appliqué
pour la visualisation et l'analyse. Kraken BLLA et le fine-tuning
utilisent l'image originale en niveaux de gris (normalisation interne
au modèle).

## Biais connus et limites

- **Déséquilibre temporel** : 72% XVe siècle / 28% XIVe siècle
  (reflet de la disponibilité des corpus ouverts, pas un choix délibéré).
- **Déséquilibre typologique** : majorité de manuscrits littéraires
  (CATMuS) vs registres administratifs (HIMANIS). La diversité est
  intentionnelle pour tester la généralisation du modèle.
- **window_size Sauvola** : la valeur 25 px est calibrée pour ~300 DPI.
  Sur les images HIMANIS (4374×4374 px, ~1200 DPI effectif), le
  binariseur peut introduire du bruit de texture — à ajuster si
  le prétraitement est utilisé en production.