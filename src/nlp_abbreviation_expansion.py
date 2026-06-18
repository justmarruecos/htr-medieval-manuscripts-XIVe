"""Expansion des abréviations médiévales françaises (XIVe-XVe s.).

Ce module implémente un pipeline d'expansion en deux étapes :
1. Règles déterministes : table de 60+ abréviations + normalisation Unicode NFC
2. Correction MLM optionnelle : CamemBERT masked language model pour les cas ambigus

Le CER (Character Error Rate) est estimé avant/après expansion pour évaluer
l'impact potentiel sur le réentraînement du modèle HTR.

Dépendances optionnelles :
- transformers + torch : correction MLM (CamemBERT)
- editdistance : calcul CER optimisé (fallback Levenshtein manuel sinon)

Google-style docstrings. Compatible Python 3.9+.
"""

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional


# ── Constantes ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Table d'abréviations médiévales françaises (60+ règles) ──────────────────
# Sources : DMF (atilf.fr), CREMMA conventions, Cappelli, HIMANIS
# Classement par catégorie pour lisibilité

ABBREVIATION_TABLE: dict[str, str] = {
    # ─── Signes tironiennes et symboles spéciaux ───
    "⁊": "et",                    # tironian et
    "÷": "est",                   # signe diviseur = est
    "℥": "once",                  # symbole once (médical)

    # ─── Lettres spéciales médiévales (Unicode) ───
    "ꝯ": "con",                   # con/com — contexte par défaut: con
    "ꝑ": "par",                   # par/per — contexte par défaut: par
    "ꝓ": "pro",                   # pro
    "ꝵ": "rum",                   # lettre rum
    "ꝭ": "us",                    # lettre us
    "ꝺ": "d",                     # d insular
    "ꝼ": "f",                     # f insular
    "ꝿ": "tur",                   # lettre tur

    # ─── Voyelles nasalisées (tilde = n/m suivant) ───
    "ẽ": "en",                    # e tilde → en/em
    "õ": "on",                    # o tilde → on/om
    "ĩ": "in",                    # i tilde → in/im
    "ũ": "un",                    # u tilde → un/um
    "ã": "an",                    # a tilde → an/am
    "ñ": "nn",                    # n tilde → nasalisation doublée
    "ỹ": "yn",                    # y tilde → yn

    # ─── Combinaisons fréquentes avec tilde ───
    "q̃": "que",                   # q + combining tilde
    "p̃": "par",                   # p + combining tilde
    "t̃": "tur",                   # t + combining tilde
    "d̃": "de",                    # d + combining tilde

    # ─── Abréviations par contraction (mots courants) ───
    "dns": "dominus",             # seigneur
    "dno": "domino",              # au seigneur
    "dnm": "dominum",             # le seigneur (acc.)
    "scs": "sanctus",             # saint
    "scm": "sanctum",             # saint (acc.)
    "sca": "sancta",              # sainte
    "eps": "episcopus",           # évêque
    "pbr": "presbyter",           # prêtre
    "xps": "christus",            # Christ
    "xpi": "christi",             # du Christ
    "ihu": "jesu",                # Jésus
    "ihc": "jesus christus",      # Jésus Christ

    # ─── Abréviations par suspension (fin de mot coupée) ───
    "grã": "grant",               # grand
    "grãt": "grant",              # grand
    "cõ": "con",                  # con/com prefix
    "cõm": "comm",                # comm- prefix
    "cõe": "comme",               # comme
    "tãt": "tant",                # tant
    "tẽps": "temps",              # temps
    "hõe": "homme",               # homme
    "hões": "hommes",             # hommes
    "bõ": "bon",                  # bon
    "bõne": "bonne",              # bonne

    # ─── Préfixes et suffixes abrégés ───
    "ꝑ": "par",                   # par/per (répété pour priorité)
    "9": "con",                   # 9 = con/com (forme chiffrée)
    "ꝯt": "cont",                # cont-
    "ꝯs": "cons",                # cons-
    "ꝯd": "cond",                # cond-

    # ─── Marques de nasalisation contextuelle ───
    "ãci": "anci",                # ancien
    "ẽt": "ent",                  # -ement, -ent
    "õt": "ont",                  # -ont
    "ĩt": "int",                  # -int
    "ũt": "unt",                  # -unt

    # ─── Abréviations juridiques et administratives ───
    "nre": "nostre",              # notre
    "vre": "vostre",              # votre
    "lre": "lettre",              # lettre
    "tre": "titre",               # titre (contexte juridique)

    # ─── Formes verbales abrégées ───
    "ꝑl": "parle",               # parler
    "ꝑt": "part",                # partir / part
    "pñt": "present",            # présent
    "cõmãd": "command",          # commander

    # ─── Terminaisons communes ───
    "tiõ": "tion",                # -tion
    "ciõ": "cion",                # -cion (ancien français)
    "siõ": "sion",                # -sion
    "mẽt": "ment",               # -ment
}

# Patterns regex pour tildes combinantes (Unicode combining tilde U+0303)
COMBINING_TILDE_PATTERN = re.compile(r"([aeiounAEIOUN])\u0303")

# Mapping voyelle + tilde combinante → expansion nasale
TILDE_NASAL_MAP: dict[str, str] = {
    "a": "an", "A": "An",
    "e": "en", "E": "En",
    "i": "in", "I": "In",
    "o": "on", "O": "On",
    "u": "un", "U": "Un",
    "n": "nn", "N": "Nn",
}


# ── Fonctions d'expansion déterministe ────────────────────────────────────────

def normalize_nfc(text: str) -> str:
    """Applique la normalisation Unicode NFC au texte.

    NFC (Canonical Decomposition + Canonical Composition) recompose
    les séquences base+combining en caractères précomposés quand possible.

    Args:
        text: Texte brut.

    Returns:
        Texte normalisé NFC.
    """
    return unicodedata.normalize("NFC", text)


def expand_combining_tildes(text: str) -> str:
    """Résout les tildes combinantes (U+0303) en expansion nasale.

    Transforme voyelle + combining tilde en voyelle + n/m selon les
    conventions du moyen français.

    Args:
        text: Texte potentiellement contenant des tildes combinantes.

    Returns:
        Texte avec tildes résolues.
    """
    def replace_tilde(match: re.Match) -> str:
        base_char = match.group(1)
        return TILDE_NASAL_MAP.get(base_char, base_char + "n")

    return COMBINING_TILDE_PATTERN.sub(replace_tilde, text)


def apply_abbreviation_table(text: str) -> str:
    """Applique la table d'abréviations par ordre décroissant de longueur.

    Les entrées les plus longues sont traitées en premier pour éviter
    les substitutions partielles (ex: 'cõe' avant 'cõ').

    Args:
        text: Texte après normalisation NFC.

    Returns:
        Texte avec abréviations développées.
    """
    for abbr, expansion in sorted(
        ABBREVIATION_TABLE.items(), key=lambda x: len(x[0]), reverse=True
    ):
        text = text.replace(abbr, expansion)
    return text


def expand_deterministic(text: str) -> str:
    """Pipeline d'expansion déterministe complet.

    Ordre : NFC → tildes combinantes → table d'abréviations.

    Args:
        text: Texte brut de transcription.

    Returns:
        Texte avec toutes les abréviations connues développées.
    """
    text = normalize_nfc(text)
    text = expand_combining_tildes(text)
    text = apply_abbreviation_table(text)
    return text


# ── Correction MLM (CamemBERT) — optionnelle ─────────────────────────────────

def _load_mlm_pipeline():
    """Charge le pipeline CamemBERT fill-mask (lazy loading).

    Returns:
        Pipeline fill-mask ou None si transformers indisponible.
    """
    try:
        from transformers import pipeline as hf_pipeline
        print("[INFO] Chargement du modèle CamemBERT fill-mask...")
        mlm = hf_pipeline("fill-mask", model="camembert-base", top_k=5)
        print("[INFO] CamemBERT chargé avec succès.")
        return mlm
    except ImportError:
        print("[AVERTISSEMENT] transformers non installé — correction MLM désactivée.")
        return None
    except Exception as e:
        print(f"[AVERTISSEMENT] Erreur lors du chargement CamemBERT : {e}")
        return None


def score_candidates_mlm(
    mlm_pipeline,
    context: str,
    candidates: list[str],
    position: int,
) -> list[tuple[str, float]]:
    """Score des candidats d'expansion via CamemBERT fill-mask.

    Masque le mot à la position donnée et retourne les scores
    des candidats proposés par le modèle.

    Args:
        mlm_pipeline: Pipeline HuggingFace fill-mask.
        context: Phrase complète contenant le mot à évaluer.
        candidates: Liste de candidats d'expansion possibles.
        position: Index du mot à masquer dans la phrase.

    Returns:
        Liste de tuples (candidat, score) triés par score décroissant.
    """
    if mlm_pipeline is None:
        return [(candidates[0], 1.0)] if candidates else []

    words = context.split()
    if position >= len(words):
        return [(candidates[0], 1.0)] if candidates else []

    # Remplacer le mot par le masque CamemBERT
    masked_words = words.copy()
    masked_words[position] = "<mask>"
    masked_text = " ".join(masked_words)

    try:
        results = mlm_pipeline(masked_text)
        # Scorer les candidats
        scored = []
        result_tokens = {r["token_str"].strip().lower(): r["score"] for r in results}
        for candidate in candidates:
            score = result_tokens.get(candidate.lower(), 0.0)
            scored.append((candidate, score))
        # Trier par score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored if scored else [(candidates[0], 0.0)]
    except Exception:
        return [(candidates[0], 0.0)]


def expand_with_mlm(text: str, mlm_pipeline) -> str:
    """Expansion avec correction MLM pour les cas ambigus.

    Identifie les positions où l'expansion déterministe produit
    des ambiguïtés (ꝯ → con ou com, ꝑ → par ou per) et utilise
    CamemBERT pour choisir la meilleure option.

    Args:
        text: Texte après expansion déterministe.
        mlm_pipeline: Pipeline CamemBERT (peut être None).

    Returns:
        Texte corrigé par le MLM ou texte inchangé si MLM indisponible.
    """
    if mlm_pipeline is None:
        return text

    # Cas ambigus connus : con/com, par/per
    ambiguous_patterns = [
        (r"\bcon(\w+)", ["con", "com"]),
        (r"\bpar(\w+)", ["par", "per"]),
    ]

    words = text.split()
    for i, word in enumerate(words):
        for pattern, candidates in ambiguous_patterns:
            if re.match(pattern, word, re.IGNORECASE):
                scored = score_candidates_mlm(mlm_pipeline, text, candidates, i)
                if scored and scored[0][1] > 0.01:
                    best = scored[0][0]
                    # Remplacer le préfixe
                    suffix = re.sub(r"^(con|com|par|per)", "", word, flags=re.IGNORECASE)
                    words[i] = best + suffix
                break

    return " ".join(words)


# ── Calcul du CER (Character Error Rate) ─────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calcule la distance de Levenshtein entre deux chaînes.

    Implémentation DP classique O(n*m) en espace O(min(n,m)).

    Args:
        s1: Première chaîne.
        s2: Deuxième chaîne.

    Returns:
        Distance d'édition (insertions, suppressions, substitutions).
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, suppression, substitution
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Calcule le Character Error Rate entre référence et hypothèse.

    CER = distance_levenshtein(ref, hyp) / len(ref)

    Utilise editdistance si disponible (C optimisé), sinon fallback Python.

    Args:
        reference: Texte de référence.
        hypothesis: Texte hypothèse (transcription).

    Returns:
        CER comme float entre 0.0 et potentiellement > 1.0.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0

    try:
        import editdistance
        dist = editdistance.eval(reference, hypothesis)
    except ImportError:
        dist = levenshtein_distance(reference, hypothesis)

    return dist / len(reference)


# ── Traitement principal ─────────────────────────────────────────────────────

def process_all_lines(
    data: list[dict[str, str]],
    use_mlm: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Traite toutes les lignes de transcription avec expansion d'abréviations.

    Args:
        data: Liste de dicts avec clés "page" et "transcription".
        use_mlm: Si True, tente d'utiliser CamemBERT pour les cas ambigus.

    Returns:
        Tuple (résultats expandus, statistiques globales).
    """
    # Charger MLM si demandé
    mlm_pipeline = None
    if use_mlm:
        mlm_pipeline = _load_mlm_pipeline()

    results: list[dict[str, Any]] = []
    cer_values: list[float] = []
    expansion_count = 0
    total_chars_before = 0
    total_chars_after = 0

    total = len(data)
    print(f"[INFO] Traitement de {total} lignes...")

    for i, entry in enumerate(data):
        raw = entry.get("transcription", "")
        page = entry.get("page", "")

        # Expansion déterministe
        expanded = expand_deterministic(raw)

        # Correction MLM (si disponible)
        if mlm_pipeline is not None:
            expanded = expand_with_mlm(expanded, mlm_pipeline)

        # Calculer le CER relatif (raw vs expanded)
        if raw:
            cer = compute_cer(raw, expanded)
            cer_values.append(cer)
        else:
            cer = 0.0

        # Vérifier si une expansion a eu lieu
        if raw != expanded:
            expansion_count += 1

        total_chars_before += len(raw)
        total_chars_after += len(expanded)

        results.append({
            "page": page,
            "transcription_raw": raw,
            "transcription_expanded": expanded,
            "cer_relative": round(cer, 4),
            "modified": raw != expanded,
        })

        # Progress
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  [{i + 1}/{total}] lignes traitées...")

    # Statistiques globales
    mean_cer = sum(cer_values) / len(cer_values) if cer_values else 0.0
    stats = {
        "total_lines": total,
        "lines_modified": expansion_count,
        "modification_rate": round(expansion_count / total * 100, 2) if total > 0 else 0.0,
        "mean_cer_relative": round(mean_cer, 4),
        "total_chars_before": total_chars_before,
        "total_chars_after": total_chars_after,
        "char_increase": total_chars_after - total_chars_before,
        "mlm_used": mlm_pipeline is not None,
        "abbreviation_rules_count": len(ABBREVIATION_TABLE),
    }

    return results, stats


def print_examples(results: list[dict[str, Any]], n: int = 10) -> None:
    """Affiche des exemples avant/après expansion.

    Args:
        results: Liste des résultats d'expansion.
        n: Nombre d'exemples à afficher.
    """
    modified = [r for r in results if r["modified"]]
    examples = modified[:n]

    print(f"\n{'─' * 70}")
    print(f"  EXEMPLES D'EXPANSION ({min(n, len(modified))} sur {len(modified)} lignes modifiées)")
    print(f"{'─' * 70}")

    for i, ex in enumerate(examples, 1):
        print(f"\n  [{i}] Page: {ex['page']}")
        print(f"      RAW:      {ex['transcription_raw']}")
        print(f"      EXPANDED: {ex['transcription_expanded']}")
        print(f"      CER:      {ex['cer_relative']:.4f}")


def estimate_cer_improvement(stats: dict[str, Any]) -> str:
    """Estime l'amélioration potentielle du CER si on réentraîne avec les expansions.

    L'estimation est basée sur le CER relatif moyen entre raw et expanded :
    les abréviations résolues réduisent l'ambiguïté pour le modèle HTR.

    Args:
        stats: Statistiques globales de l'expansion.

    Returns:
        Message d'estimation formaté.
    """
    mean_cer = stats["mean_cer_relative"]
    # Estimation conservative : ~30-50% du CER relatif peut être récupéré
    # en réentraînant avec le ground truth expandu
    estimated_reduction = mean_cer * 0.4  # 40% de récupération estimée
    estimated_points = round(estimated_reduction * 100, 2)

    return (
        f"CER relatif moyen (raw vs expanded) : {mean_cer * 100:.2f}%\n"
        f"Si réentraînement avec ground truth expandu :\n"
        f"  → Réduction CER estimée : ~{estimated_points:.2f} points\n"
        f"  (hypothèse : récupération de 40% du delta via normalisation)"
    )


# ── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> None:
    """Point d'entrée principal du script d'expansion d'abréviations.

    Charge les transcriptions, applique l'expansion, affiche les
    statistiques et sauvegarde les résultats.
    """
    print("[INFO] Démarrage de l'expansion des abréviations médiévales...")
    print(f"[INFO] Table d'abréviations : {len(ABBREVIATION_TABLE)} règles chargées.")

    # Charger les données
    kraken_path = PROJECT_ROOT / "results" / "kraken_output.json"
    print(f"[INFO] Chargement de {kraken_path}...")

    if not kraken_path.exists():
        print(f"[ERREUR] Fichier introuvable : {kraken_path}")
        return

    with open(kraken_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] {len(data)} lignes chargées.")

    # Traitement
    results, stats = process_all_lines(data, use_mlm=True)

    # Affichage
    print(f"\n{'=' * 70}")
    print("  RÉSULTATS D'EXPANSION DES ABRÉVIATIONS")
    print(f"{'=' * 70}")
    print(f"  Lignes traitées     : {stats['total_lines']}")
    print(f"  Lignes modifiées    : {stats['lines_modified']} ({stats['modification_rate']:.1f}%)")
    print(f"  CER relatif moyen   : {stats['mean_cer_relative'] * 100:.2f}%")
    print(f"  Caractères avant    : {stats['total_chars_before']}")
    print(f"  Caractères après    : {stats['total_chars_after']} (+{stats['char_increase']})")
    print(f"  MLM utilisé         : {'Oui' if stats['mlm_used'] else 'Non (règles seules)'}")
    print(f"  Règles appliquées   : {stats['abbreviation_rules_count']}")

    # Exemples
    print_examples(results)

    # Estimation CER
    print(f"\n{'─' * 70}")
    print("  ESTIMATION D'AMÉLIORATION CER")
    print(f"{'─' * 70}")
    print(f"  {estimate_cer_improvement(stats)}")

    # Sauvegarder
    output_path = PROJECT_ROOT / "results" / "transcriptions_expanded.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Format de sortie : juste les transcriptions expandues
    output_data = [
        {
            "page": r["page"],
            "transcription_raw": r["transcription_raw"],
            "transcription_expanded": r["transcription_expanded"],
            "cer_relative": r["cer_relative"],
        }
        for r in results
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Transcriptions expandues sauvegardées : {output_path}")

    # Sauvegarder les stats séparément
    stats_path = PROJECT_ROOT / "results" / "expansion_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Statistiques sauvegardées : {stats_path}")
    print("[INFO] Expansion terminée.")


if __name__ == "__main__":
    main()
