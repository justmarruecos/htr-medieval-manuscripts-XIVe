"""Analyse POS (Part-of-Speech) et graphe d'entités pour manuscrits médiévaux.

Ce module utilise Stanza (modèle Old French `fro` ou fallback `fr`) pour :
1. Étiquetage morphosyntaxique (POS tagging) et lemmatisation
2. Extraction de relations simples (patterns PERSON + verbe + LOCATION)
3. Construction d'un graphe de co-occurrences d'entités par page (NetworkX)

Dépendances :
- stanza : POS tagging et lemmatisation (requis)
- networkx : graphe d'entités (requis)

Résultats sauvegardés dans :
- results/pos_analysis.json
- results/entity_graph.gexf

Google-style docstrings. Compatible Python 3.9+.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional


# ── Constantes ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Nombre de lignes à traiter (échantillon)
SAMPLE_SIZE = 500

# Tags POS universels (Universal Dependencies)
POS_DESCRIPTIONS: dict[str, str] = {
    "NOUN": "Nom commun",
    "VERB": "Verbe",
    "ADJ": "Adjectif",
    "ADV": "Adverbe",
    "ADP": "Adposition (préposition)",
    "DET": "Déterminant",
    "PRON": "Pronom",
    "CONJ": "Conjonction",
    "CCONJ": "Conjonction de coordination",
    "SCONJ": "Conjonction de subordination",
    "NUM": "Numéral",
    "PART": "Particule",
    "INTJ": "Interjection",
    "PUNCT": "Ponctuation",
    "SYM": "Symbole",
    "X": "Autre",
    "AUX": "Auxiliaire",
    "PROPN": "Nom propre",
}

# Patterns regex pour extraction de relations simples
# PERSON + verbe + LOCATION (approximation sans NER fine)
RELATION_PATTERNS = [
    # Nom propre + verbe de mouvement + lieu
    re.compile(
        r"\b([A-Z][a-zàâéèêëîïôùûüç]+)\b\s+"
        r"(?:ala|alla|vint|entra|parti[t]?|arriva|retourna|sen ala)\s+"
        r"(?:a|en|de|à)\s+"
        r"\b([A-Z][a-zàâéèêëîïôùûüç]+)\b",
        re.UNICODE,
    ),
    # Nom propre + verbe d'action + nom propre
    re.compile(
        r"\b([A-Z][a-zàâéèêëîïôùûüç]+)\b\s+"
        r"(?:dist|dit|parla|commanda|envoia|manda)\s+"
        r"(?:a|au|à)\s+"
        r"\b([A-Z][a-zàâéèêëîïôùûüç]+)\b",
        re.UNICODE,
    ),
    # "le roi/sire X" pattern
    re.compile(
        r"(?:le\s+ro[iy]|li\s+ro[iy]|sire|messire)\s+"
        r"\b([A-Z][a-zàâéèêëîïôùûüç]+)\b",
        re.UNICODE,
    ),
]


# ── Chargement de Stanza ─────────────────────────────────────────────────────

def load_stanza_pipeline():
    """Charge le pipeline Stanza avec le modèle Old French ou fallback French.

    Tente d'abord le modèle `fro` (Old French), puis `fr` (French moderne)
    en cas d'échec. Télécharge automatiquement le modèle si nécessaire.

    Returns:
        Tuple (pipeline stanza, nom du modèle utilisé) ou (None, None).
    """
    try:
        import stanza
    except ImportError:
        print("[ERREUR] stanza non installé. Installez avec : pip install stanza")
        return None, None

    # Tenter Old French (fro)
    for lang in ["fro", "fr"]:
        try:
            print(f"[INFO] Tentative de chargement du modèle Stanza '{lang}'...")
            stanza.download(lang, verbose=False)
            nlp = stanza.Pipeline(
                lang,
                processors="tokenize,pos,lemma",
                verbose=False,
                use_gpu=False,
            )
            print(f"[INFO] Modèle Stanza '{lang}' chargé avec succès.")
            return nlp, lang
        except Exception as e:
            print(f"[AVERTISSEMENT] Modèle '{lang}' indisponible : {e}")
            continue

    print("[ERREUR] Aucun modèle Stanza disponible (fro ou fr).")
    return None, None


# ── Analyse POS ──────────────────────────────────────────────────────────────

def analyze_pos(
    transcriptions: list[str],
    nlp_pipeline,
) -> tuple[Counter, list[dict[str, str]]]:
    """Effectue l'analyse POS sur les transcriptions.

    Args:
        transcriptions: Liste de lignes de texte à analyser.
        nlp_pipeline: Pipeline Stanza chargé.

    Returns:
        Tuple (compteur POS, liste de tokens avec pos et lemme).
    """
    pos_counter: Counter = Counter()
    tokens_data: list[dict[str, str]] = []

    total = len(transcriptions)
    print(f"[INFO] Analyse POS de {total} lignes...")

    # Traiter par batchs pour performance
    batch_size = 50
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = transcriptions[batch_start:batch_end]

        # Joindre le batch en un document
        text = "\n".join(line for line in batch if line.strip())
        if not text.strip():
            continue

        try:
            doc = nlp_pipeline(text)
            for sentence in doc.sentences:
                for word in sentence.words:
                    pos = word.upos if word.upos else "X"
                    lemma = word.lemma if word.lemma else word.text
                    pos_counter[pos] += 1
                    tokens_data.append({
                        "text": word.text,
                        "pos": pos,
                        "lemma": lemma,
                    })
        except Exception as e:
            print(f"  [AVERTISSEMENT] Erreur batch {batch_start}-{batch_end}: {e}")
            continue

        if (batch_end) % 100 == 0 or batch_end == total:
            print(f"  [{batch_end}/{total}] lignes analysées...")

    return pos_counter, tokens_data


def format_pos_distribution(
    pos_counter: Counter, top_n: int = 15
) -> list[dict[str, Any]]:
    """Formate la distribution POS en liste de dicts pour le rapport.

    Args:
        pos_counter: Compteur de tags POS.
        top_n: Nombre de tags à inclure.

    Returns:
        Liste de dicts avec tag, count, percentage, description.
    """
    total = sum(pos_counter.values())
    distribution = []

    for tag, count in pos_counter.most_common(top_n):
        distribution.append({
            "tag": tag,
            "count": count,
            "percentage": round(count / total * 100, 2) if total > 0 else 0.0,
            "description": POS_DESCRIPTIONS.get(tag, "Inconnu"),
        })

    return distribution


# ── Extraction d'entités et relations ─────────────────────────────────────────

def extract_proper_nouns(tokens_data: list[dict[str, str]]) -> list[str]:
    """Extrait les noms propres identifiés par le POS tagger.

    Args:
        tokens_data: Liste de tokens avec leur tag POS.

    Returns:
        Liste de noms propres uniques.
    """
    propn = set()
    for token in tokens_data:
        if token["pos"] == "PROPN" and len(token["text"]) > 2:
            propn.add(token["text"])
    return sorted(propn)


def extract_relations(
    transcriptions: list[str],
) -> list[dict[str, str]]:
    """Extrait des relations simples via patterns regex.

    Recherche des patterns PERSON + verbe + LOCATION dans les
    transcriptions brutes.

    Args:
        transcriptions: Liste de lignes de texte.

    Returns:
        Liste de relations extraites (source, relation, target).
    """
    relations: list[dict[str, str]] = []

    for line in transcriptions:
        for pattern in RELATION_PATTERNS:
            matches = pattern.finditer(line)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    relations.append({
                        "source": groups[0],
                        "relation": "associated_with",
                        "target": groups[1],
                        "context": line[:100],
                    })
                elif len(groups) == 1:
                    relations.append({
                        "source": groups[0],
                        "relation": "mentioned_as_noble",
                        "target": "",
                        "context": line[:100],
                    })

    return relations


# ── Graphe de co-occurrences ──────────────────────────────────────────────────

def build_cooccurrence_graph(
    data: list[dict[str, str]],
    tokens_by_page: dict[str, list[dict[str, str]]],
) -> Any:
    """Construit un graphe NetworkX de co-occurrences d'entités par page.

    Deux entités (noms propres) sont reliées si elles apparaissent
    sur la même page. Le poids de l'arête = nombre de co-occurrences.

    Args:
        data: Données brutes avec clé "page".
        tokens_by_page: Dictionnaire page → liste de tokens.

    Returns:
        Graphe NetworkX ou None si networkx indisponible.
    """
    try:
        import networkx as nx
    except ImportError:
        print("[ERREUR] networkx non installé. Installez avec : pip install networkx")
        return None

    G = nx.Graph()

    # Extraire les noms propres par page
    entities_by_page: dict[str, set[str]] = {}
    for page, tokens in tokens_by_page.items():
        propn = set()
        for token in tokens:
            if token["pos"] == "PROPN" and len(token["text"]) > 2:
                propn.add(token["text"])
        if propn:
            entities_by_page[page] = propn

    # Ajouter les nœuds et arêtes
    for page, entities in entities_by_page.items():
        entity_list = sorted(entities)
        for entity in entity_list:
            if not G.has_node(entity):
                G.add_node(entity, label=entity, pages=set())
            G.nodes[entity]["pages"].add(page)

        # Arêtes entre toutes les paires d'entités co-occurrentes
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                e1, e2 = entity_list[i], entity_list[j]
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                    G[e1][e2]["pages"].append(page)
                else:
                    G.add_edge(e1, e2, weight=1, pages=[page])

    # Convertir les sets en listes pour sérialisation
    for node in G.nodes:
        G.nodes[node]["pages"] = list(G.nodes[node].get("pages", set()))
        G.nodes[node]["degree"] = G.degree(node)

    print(f"[INFO] Graphe construit : {G.number_of_nodes()} nœuds, "
          f"{G.number_of_edges()} arêtes.")

    return G


def save_graph(G, output_path: Path) -> None:
    """Sauvegarde le graphe au format GEXF.

    Args:
        G: Graphe NetworkX.
        output_path: Chemin de sortie pour le fichier GEXF.
    """
    try:
        import networkx as nx
    except ImportError:
        print("[ERREUR] networkx non installé — sauvegarde impossible.")
        return

    if G is None or G.number_of_nodes() == 0:
        print("[AVERTISSEMENT] Graphe vide — sauvegarde ignorée.")
        return

    # Nettoyer les attributs pour GEXF (convertir listes en strings)
    for node in G.nodes:
        pages = G.nodes[node].get("pages", [])
        G.nodes[node]["pages"] = ",".join(str(p) for p in pages)
    for u, v in G.edges:
        pages = G[u][v].get("pages", [])
        G[u][v]["pages"] = ",".join(str(p) for p in pages)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, str(output_path))
    print(f"[INFO] Graphe sauvegardé : {output_path}")


# ── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> None:
    """Point d'entrée principal de l'analyse POS et graphe d'entités.

    Charge les transcriptions, effectue le POS tagging avec Stanza,
    extrait les relations et construit le graphe de co-occurrences.
    """
    print("[INFO] Démarrage de l'analyse POS et graphe d'entités...")

    # 1. Charger les données
    kraken_path = PROJECT_ROOT / "results" / "kraken_output.json"
    if not kraken_path.exists():
        print(f"[ERREUR] Fichier introuvable : {kraken_path}")
        return

    with open(kraken_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] {len(data)} lignes chargées.")

    # 2. Échantillonner
    sample = data[:SAMPLE_SIZE]
    transcriptions = [entry.get("transcription", "") for entry in sample]
    transcriptions = [t for t in transcriptions if t.strip()]
    print(f"[INFO] Échantillon : {len(transcriptions)} lignes non vides "
          f"(sur {SAMPLE_SIZE} demandées).")

    # 3. Charger Stanza
    nlp_pipeline, model_name = load_stanza_pipeline()
    if nlp_pipeline is None:
        print("[ERREUR] Impossible de charger un modèle Stanza. Abandon.")
        return

    # 4. Analyse POS
    pos_counter, tokens_data = analyze_pos(transcriptions, nlp_pipeline)

    # 5. Distribution POS
    pos_distribution = format_pos_distribution(pos_counter, top_n=15)

    print(f"\n{'=' * 70}")
    print("  DISTRIBUTION POS (Top 15)")
    print(f"{'=' * 70}")
    print(f"  {'Tag':<8} {'Count':<10} {'%':<8} Description")
    print(f"  {'─' * 60}")
    for entry in pos_distribution:
        print(f"  {entry['tag']:<8} {entry['count']:<10} "
              f"{entry['percentage']:<8.2f} {entry['description']}")

    total_tokens = sum(pos_counter.values())
    print(f"\n  Total tokens analysés : {total_tokens}")

    # 6. Noms propres et lemmes
    proper_nouns = extract_proper_nouns(tokens_data)
    print(f"\n[INFO] Noms propres identifiés : {len(proper_nouns)}")
    if proper_nouns[:20]:
        print(f"  Exemples : {', '.join(proper_nouns[:20])}")

    # 7. Extraction de relations
    relations = extract_relations(transcriptions)
    print(f"\n[INFO] Relations extraites : {len(relations)}")
    for rel in relations[:5]:
        print(f"  {rel['source']} → {rel['relation']} → {rel['target']}")

    # 8. Graphe de co-occurrences
    # Organiser les tokens par page
    tokens_by_page: dict[str, list[dict[str, str]]] = {}
    token_idx = 0
    for entry in sample:
        page = entry.get("page", "unknown")
        text = entry.get("transcription", "")
        if not text.strip():
            continue
        # Approximation : attribuer les tokens au batch correspondant
        if page not in tokens_by_page:
            tokens_by_page[page] = []

    # Re-analyser par page pour le graphe (plus précis)
    for entry in sample:
        page = entry.get("page", "unknown")
        text = entry.get("transcription", "")
        if not text.strip():
            continue
        if page not in tokens_by_page:
            tokens_by_page[page] = []
        # Ajouter les tokens de cette page
        try:
            doc = nlp_pipeline(text)
            for sentence in doc.sentences:
                for word in sentence.words:
                    tokens_by_page[page].append({
                        "text": word.text,
                        "pos": word.upos if word.upos else "X",
                        "lemma": word.lemma if word.lemma else word.text,
                    })
        except Exception:
            continue

    print(f"\n[INFO] Construction du graphe de co-occurrences...")
    G = build_cooccurrence_graph(data, tokens_by_page)

    # 9. Sauvegarder les résultats POS
    pos_results = {
        "model_used": model_name,
        "sample_size": len(transcriptions),
        "total_tokens": total_tokens,
        "pos_distribution": pos_distribution,
        "proper_nouns": proper_nouns[:100],  # Limiter pour le JSON
        "relations_extracted": relations[:50],
        "graph_stats": {
            "nodes": G.number_of_nodes() if G else 0,
            "edges": G.number_of_edges() if G else 0,
        },
    }

    output_path = PROJECT_ROOT / "results" / "pos_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pos_results, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Résultats POS sauvegardés : {output_path}")

    # 10. Sauvegarder le graphe GEXF
    graph_path = PROJECT_ROOT / "results" / "entity_graph.gexf"
    save_graph(G, graph_path)

    print("\n[INFO] Analyse POS et graphe d'entités terminée.")


if __name__ == "__main__":
    main()
