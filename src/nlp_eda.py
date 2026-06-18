"""Analyse exploratoire (EDA) des transcriptions HTR de manuscrits médiévaux.

Ce module charge les résultats de Kraken et les transcriptions NLP,
puis produit un rapport statistique complet :
- Distribution des longueurs de lignes
- Fréquence des caractères abréviatifs médiévaux
- Distribution de confiance et taux de révision nécessaire

Résultats sauvegardés dans results/nlp_eda_report.json.
Google-style docstrings. Compatible Python 3.9+.
"""

import json
import os
import statistics
from pathlib import Path
from typing import Any, Optional


# ── Constantes ────────────────────────────────────────────────────────────────

# Répertoire racine du projet (deux niveaux au-dessus de src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Caractères abréviatifs médiévaux à détecter
MEDIEVAL_ABBREVIATION_CHARS: dict = {
    "⁊": "tironian et (et)",
    "ꝯ": "con/com",
    "ꝑ": "par/per",
    "ꝓ": "pro",
    "÷": "est (signe diviseur)",
    "ẽ": "e tilde (en/em)",
    "õ": "o tilde (on/om)",
    "ĩ": "i tilde (in/im)",
    "ũ": "u tilde (un/um)",
    "ñ": "n tilde (nasalisation)",
    "ã": "a tilde (an/am)",
    "\u0303": "combining tilde (suscript)",
    "ꝵ": "rum",
    "ꝭ": "us",
    "q̃": "q tilde (que)",
    "p̃": "p tilde (par)",
}


# ── Fonctions utilitaires ─────────────────────────────────────────────────────

def load_json(filepath: Path) -> list:
    """Charge un fichier JSON et retourne la liste de dictionnaires.

    Args:
        filepath: Chemin absolu ou relatif vers le fichier JSON.

    Returns:
        Liste de dictionnaires parsés depuis le JSON.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        json.JSONDecodeError: Si le JSON est malformé.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_quartiles(values: list) -> dict[str, float]:
    """Calcule les quartiles Q1, Q2 (médiane), Q3 d'une série numérique.

    Args:
        values: Liste de valeurs numériques.

    Returns:
        Dictionnaire avec clés q1, median, q3.
    """
    if not values:
        return {"q1": 0.0, "median": 0.0, "q3": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4]
    q2 = statistics.median(sorted_vals)
    q3 = sorted_vals[(3 * n) // 4]
    return {"q1": q1, "median": q2, "q3": q3}


# ── Analyse des transcriptions Kraken ─────────────────────────────────────────

def analyze_kraken_output(data: list[dict]) -> dict[str, Any]:
    """Analyse les transcriptions issues de Kraken (kraken_output.json).

    Calcule les statistiques de longueur de ligne et la fréquence
    des caractères abréviatifs médiévaux.

    Args:
        data: Liste de dicts avec clés "page" et "transcription".

    Returns:
        Dictionnaire contenant toutes les statistiques calculées.
    """
    transcriptions = [entry.get("transcription", "") for entry in data]
    total_lines = len(transcriptions)
    line_lengths = [len(t) for t in transcriptions]

    # Statistiques de longueur
    length_stats = {
        "total_lines": total_lines,
        "mean_length": round(statistics.mean(line_lengths), 2) if line_lengths else 0.0,
        "median_length": statistics.median(line_lengths) if line_lengths else 0.0,
        "min_length": min(line_lengths) if line_lengths else 0,
        "max_length": max(line_lengths) if line_lengths else 0,
        "quartiles": compute_quartiles(line_lengths),
    }

    # Comptage des caractères abréviatifs
    abbrev_counts: dict = {char: 0 for char in MEDIEVAL_ABBREVIATION_CHARS}
    lines_with_abbreviations = 0

    for transcription in transcriptions:
        has_abbrev = False
        for char in MEDIEVAL_ABBREVIATION_CHARS:
            count = transcription.count(char)
            if count > 0:
                abbrev_counts[char] += count
                has_abbrev = True
        if has_abbrev:
            lines_with_abbreviations += 1

    abbrev_stats = {
        "character_counts": {
            char: {"count": abbrev_counts[char], "description": desc}
            for char, desc in MEDIEVAL_ABBREVIATION_CHARS.items()
            if abbrev_counts[char] > 0
        },
        "total_abbreviation_chars": sum(abbrev_counts.values()),
        "lines_with_abbreviations": lines_with_abbreviations,
        "percentage_lines_with_abbreviations": round(
            (lines_with_abbreviations / total_lines * 100) if total_lines > 0 else 0.0, 2
        ),
    }

    # Pages uniques
    unique_pages = list(set(entry.get("page", "") for entry in data))

    return {
        "length_statistics": length_stats,
        "abbreviation_statistics": abbrev_stats,
        "unique_pages": len(unique_pages),
    }


# ── Analyse des transcriptions NLP ───────────────────────────────────────────

def analyze_nlp_transcriptions(data: list) -> dict[str, Any]:
    """Analyse les transcriptions NLP (dataset_nlp/transcriptions.json).

    Calcule la distribution de confiance et le taux de lignes
    nécessitant une révision manuelle.

    Args:
        data: Liste de dicts avec clés "page", "line_id", "transcription",
              "confidence", "needs_review", "segmentation_ref".

    Returns:
        Dictionnaire contenant les statistiques de confiance et révision.
    """
    confidences = [
        entry.get("confidence", 0.0)
        for entry in data
        if entry.get("confidence") is not None
    ]
    needs_review_count = sum(
        1 for entry in data if entry.get("needs_review", False)
    )
    total = len(data)

    confidence_stats = {}
    if confidences:
        confidence_stats = {
            "mean": round(statistics.mean(confidences), 4),
            "median": round(statistics.median(confidences), 4),
            "std_dev": round(statistics.stdev(confidences), 4) if len(confidences) > 1 else 0.0,
            "min": round(min(confidences), 4),
            "max": round(max(confidences), 4),
            "quartiles": {
                "q1": round(compute_quartiles(confidences)["q1"], 4),
                "median": round(compute_quartiles(confidences)["median"], 4),
                "q3": round(compute_quartiles(confidences)["q3"], 4),
            },
        }

    return {
        "total_lines": total,
        "confidence_distribution": confidence_stats,
        "needs_review_count": needs_review_count,
        "needs_review_rate": round(
            (needs_review_count / total * 100) if total > 0 else 0.0, 2
        ),
    }


# ── Affichage et rapport ─────────────────────────────────────────────────────

def print_summary(kraken_stats: dict[str, Any],
                  nlp_stats: Optional[dict[str, Any]]) -> None:
    """Affiche un résumé formaté des statistiques EDA.

    Args:
        kraken_stats: Statistiques issues de l'analyse Kraken.
        nlp_stats: Statistiques issues de l'analyse NLP (peut être None).
    """
    print("\n" + "=" * 70)
    print("  RAPPORT EDA — Transcriptions HTR Médiéval Français")
    print("=" * 70)

    ls = kraken_stats["length_statistics"]
    print(f"\n{'─' * 40}")
    print("  1. STATISTIQUES DE LONGUEUR (Kraken)")
    print(f"{'─' * 40}")
    print(f"  Total lignes        : {ls['total_lines']}")
    print(f"  Longueur moyenne    : {ls['mean_length']:.2f} caractères")
    print(f"  Longueur médiane    : {ls['median_length']:.1f} caractères")
    print(f"  Min / Max           : {ls['min_length']} / {ls['max_length']}")
    print(f"  Q1 / Q2 / Q3       : {ls['quartiles']['q1']} / "
          f"{ls['quartiles']['median']} / {ls['quartiles']['q3']}")

    ab = kraken_stats["abbreviation_statistics"]
    print(f"\n{'─' * 40}")
    print("  2. CARACTÈRES ABRÉVIATIFS MÉDIÉVAUX")
    print(f"{'─' * 40}")
    print(f"  Total chars abréviatifs   : {ab['total_abbreviation_chars']}")
    print(f"  Lignes avec abréviations  : {ab['lines_with_abbreviations']} "
          f"({ab['percentage_lines_with_abbreviations']:.1f}%)")
    if ab["character_counts"]:
        print(f"\n  {'Char':<6} {'Count':<8} Description")
        print(f"  {'─' * 40}")
        for char, info in sorted(
            ab["character_counts"].items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        ):
            print(f"  {char:<6} {info['count']:<8} {info['description']}")

    print(f"\n  Pages uniques : {kraken_stats['unique_pages']}")

    if nlp_stats:
        print(f"\n{'─' * 40}")
        print("  3. TRANSCRIPTIONS NLP (dataset_nlp)")
        print(f"{'─' * 40}")
        print(f"  Total lignes         : {nlp_stats['total_lines']}")
        cd = nlp_stats.get("confidence_distribution", {})
        if cd:
            print(f"  Confiance moyenne    : {cd['mean']:.4f}")
            print(f"  Confiance médiane    : {cd['median']:.4f}")
            print(f"  Confiance min/max    : {cd['min']:.4f} / {cd['max']:.4f}")
            q = cd.get("quartiles", {})
            if q:
                print(f"  Q1 / Q2 / Q3        : {q['q1']:.4f} / "
                      f"{q['median']:.4f} / {q['q3']:.4f}")
        print(f"  Taux needs_review    : {nlp_stats['needs_review_rate']:.1f}% "
              f"({nlp_stats['needs_review_count']} lignes)")

    print(f"\n{'=' * 70}")
    print("  Fin du rapport EDA")
    print("=" * 70 + "\n")


def save_report(report: dict[str, Any], output_path: Path) -> None:
    """Sauvegarde le rapport EDA au format JSON.

    Args:
        report: Dictionnaire complet du rapport.
        output_path: Chemin de sortie pour le fichier JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Rapport sauvegardé : {output_path}")


# ── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> None:
    """Point d'entrée principal du script EDA.

    Charge les données, effectue l'analyse et produit le rapport.
    """
    print("[INFO] Démarrage de l'analyse EDA des transcriptions HTR...")

    # 1. Charger kraken_output.json
    kraken_path = PROJECT_ROOT / "results" / "kraken_output.json"
    print(f"[INFO] Chargement de {kraken_path}...")
    try:
        kraken_data = load_json(kraken_path)
        print(f"[INFO] {len(kraken_data)} lignes chargées depuis Kraken.")
    except FileNotFoundError as e:
        print(f"[ERREUR] {e}")
        return

    # 2. Analyser les transcriptions Kraken
    print("[INFO] Analyse des statistiques de longueur et abréviations...")
    kraken_stats = analyze_kraken_output(kraken_data)

    # 3. Charger dataset_nlp/transcriptions.json (optionnel)
    nlp_path = PROJECT_ROOT / "dataset_nlp" / "transcriptions.json"
    nlp_stats = None
    if nlp_path.exists():
        print(f"[INFO] Chargement de {nlp_path}...")
        try:
            nlp_data = load_json(nlp_path)
            print(f"[INFO] {len(nlp_data)} lignes chargées depuis dataset_nlp.")
            nlp_stats = analyze_nlp_transcriptions(nlp_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[AVERTISSEMENT] Impossible de charger dataset_nlp : {e}")
    else:
        print(f"[AVERTISSEMENT] {nlp_path} introuvable — analyse NLP ignorée.")

    # 4. Construire le rapport complet
    report: dict[str, Any] = {
        "source": "nlp_eda.py",
        "kraken_analysis": kraken_stats,
    }
    if nlp_stats:
        report["nlp_analysis"] = nlp_stats

    # 5. Afficher le résumé
    print_summary(kraken_stats, nlp_stats)

    # 6. Sauvegarder le rapport
    output_path = PROJECT_ROOT / "results" / "nlp_eda_report.json"
    save_report(report, output_path)

    print("[INFO] Analyse EDA terminée.")


if __name__ == "__main__":
    main()
