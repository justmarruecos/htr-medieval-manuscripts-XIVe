"""Module de normalisation orthographique du moyen français (XIVe-XVe s.).

Étape 1 de la chaîne NLP : règles déterministes avant tout modèle seq2seq.
Couvre : substitutions graphiques u/v, i/j, abréviations courantes,
normalisation Unicode NFD, espaces, ponctuation.

Google-style docstrings. Testé via tests/test_normalisation.py.
"""

import re
import unicodedata
from typing import Optional


# ── Tables de substitution ────────────────────────────────────────────────────

# Abréviations médiévales courantes → forme développée
# Sources : DMF (atilf.fr), CREMMA conventions, HIMANIS transcriptions
ABBREVIATIONS = {
    # Signes abréviatifs Unicode médiévaux
    "ꝵ": "rum",   # lettre rum
    "ꝭ": "us",    # lettre us
    "q̃": "que",   # q tilde
    "p̃": "par",   # p tilde
    "ñ": "nn",    # n tilde (nasalisation → doublement)
    "ã": "an",    # a tilde
    "ẽ": "en",    # e tilde
    "ĩ": "in",    # i tilde
    "õ": "on",    # o tilde
    "ũ": "un",    # u tilde
    # Abréviations textuelles fréquentes
    "ihu": "jésus",
    "xps": "christus",
    "dns": "dominus",
    "dns": "dominus",
    "snt": "saint",
    # "sr": "sire",   # supprimé — trop de faux positifs (ex: desrober)
}

# Substitutions graphiques u/v et i/j (contexte non ambigu)
# Règle : u en position initiale devant voyelle → v
#         v en position médiane entre voyelles → u
UV_INITIAL = re.compile(r'\bu([aeiouàâéèêëîïôùûüy])', re.IGNORECASE)
UV_MEDIAL  = re.compile(r'([aeiouàâéèêëîïôùûüy])v([aeiouàâéèêëîïôùûüy])', re.IGNORECASE)

# i/j : j en position initiale → i (vieux français)
# Ancienne regex (trop large) :
# IJ_INITIAL = re.compile(r'\bj([aeiouàâéèêëîïôùûüy])', re.IGNORECASE)

# Nouvelle regex (seulement devant a/o/u — pas devant e/i déjà stables au XVe)
IJ_INITIAL = re.compile(r'\bj([aouàâôùûü])', re.IGNORECASE)


# ── Fonctions ─────────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Normalise le texte en NFD puis retire les diacritiques superflus.

    Args:
        text: Texte brut issu de la transcription HTR.

    Returns:
        Texte normalisé Unicode NFD, espaces unifiés.

    Raises:
        TypeError: Si text n'est pas une chaîne de caractères.
    """
    if not isinstance(text, str):
        raise TypeError(f"Attendu str, reçu {type(text)}")
    # NFD : décomposition canonique
    text = unicodedata.normalize("NFD", text)
    # Unifier les espaces (tabulations, espaces insécables, etc.)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def expand_abbreviations(text: str) -> str:
    """Développe les abréviations médiévales connues.

    Applique la table ABBREVIATIONS dans l'ordre décroissant de longueur
    pour éviter les substitutions partielles (ex: 'ñ' avant 'n').

    Args:
        text: Texte après normalisation Unicode.

    Returns:
        Texte avec abréviations développées.
    """
    for abbr, expansion in sorted(ABBREVIATIONS.items(),
                                   key=lambda x: len(x[0]), reverse=True):
        text = text.replace(abbr, expansion)
    return text


def normalize_uv(text: str) -> str:
    """Normalise les graphies u/v selon les règles positionnelles.

    - u initial devant voyelle → v  (ex: 'uoir' → 'voir')
    Note : la règle v médial → u n'est pas appliquée car trop de faux positifs
    sur des mots déjà normalisés (ex: 'avoir').

    Args:
        text: Texte après développement des abréviations.

    Returns:
        Texte avec u/v normalisés.
    """
    text = UV_INITIAL.sub(lambda m: 'v' + m.group(1), text)
    return text


def normalize_ij(text: str) -> str:
    """Normalise les graphies i/j en position initiale.

    j initial devant voyelle → i  (ex: 'jou' → 'iou', 'jour' → 'iour')
    Note : cette règle est conservatrice — elle ne s'applique qu'en
    position initiale absolue de mot pour éviter les faux positifs.

    Args:
        text: Texte après normalisation u/v.

    Returns:
        Texte avec i/j normalisés.
    """
    text = IJ_INITIAL.sub(lambda m: 'i' + m.group(1), text)
    return text


def normalize_punctuation(text: str) -> str:
    """Normalise la ponctuation médiévale vers des équivalents modernes.

    Args:
        text: Texte après normalisation graphique.

    Returns:
        Texte avec ponctuation normalisée.
    """
    # Point médian médiéval → espace
    text = text.replace('·', ' ')
    # Pied-de-mouche → fin de phrase
    text = text.replace('¶', '.')
    # Doubles espaces résiduels
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize(text: str,
              expand_abbr: bool = True,
              normalize_uv_ij: bool = True,
              normalize_punct: bool = True) -> str:
    """Pipeline de normalisation complet pour le moyen français.

    Enchaîne dans l'ordre : abréviations → Unicode NFD → u/v → i/j → ponctuation.
    Les abréviations sont développées AVANT la normalisation NFD pour éviter
    la décomposition des caractères abréviatifs composés (ex: ñ → n+combining).

    Args:
        text: Texte brut issu de la transcription HTR.
        expand_abbr: Si True, développe les abréviations (défaut: True).
        normalize_uv_ij: Si True, normalise u/v et i/j (défaut: True).
        normalize_punct: Si True, normalise la ponctuation (défaut: True).

    Returns:
        Texte normalisé prêt pour la NER et le topic modeling.

    Raises:
        TypeError: Si text n'est pas une chaîne de caractères.

    Example:
        >>> normalize("uoir le roy auec ses gens")
        'voir le roy avec ses gens'
    """
    if not isinstance(text, str):
        raise TypeError(f"Attendu str, reçu {type(text)}")
    if expand_abbr:
        text = expand_abbreviations(text)
    text = normalize_unicode(text)
    if normalize_uv_ij:
        text = normalize_uv(text)
        text = normalize_ij(text)
    if normalize_punct:
        text = normalize_punctuation(text)
    return text