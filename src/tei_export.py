"""Export TEI-XML pour manuscrits médiévaux normalisés + entités NER.

Convertit un texte normalisé et ses entités NER en document TEI/XML
conforme au standard des humanités numériques.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom


TEI_NS = "http://www.tei-c.org/ns/1.0"
TAG = {
    "PER": "persName",
    "LOC": "placeName",
    "ORG": "orgName",
    "MISC": "name",
}


def _pretty(element: ET.Element) -> str:
    """Sérialise un élément XML en chaîne indentée.

    Args:
        element: Élément XML racine.

    Returns:
        Chaîne XML indentée avec déclaration UTF-8.
    """
    raw = ET.tostring(element, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ", encoding=None)


def build_tei(
    text_normalized: str,
    entities: list[dict],
    shelfmark: str = "",
    title: str = "",
) -> ET.Element:
    """Construit un arbre TEI-XML à partir d'un texte normalisé et d'entités NER.

    Args:
        text_normalized: Texte en moyen français normalisé.
        entities: Liste de dicts avec clés word, entity_group, start, end.
        shelfmark: Cote du manuscrit source (ex. Paris, BnF, fr. 146).
        title: Titre du document (optionnel).

    Returns:
        Élément XML racine <TEI>.

    Example:
        >>> el = build_tei("Jehan est a Paris", [], shelfmark="BnF fr.146")
        >>> el.tag
        "TEI"
    """
    ET.register_namespace("", TEI_NS)

    tei = ET.Element("TEI")
    tei.set("xmlns", TEI_NS)

    # --- teiHeader ---
    header = ET.SubElement(tei, "teiHeader")
    file_desc = ET.SubElement(header, "fileDesc")

    title_stmt = ET.SubElement(file_desc, "titleStmt")
    title_el = ET.SubElement(title_stmt, "title")
    title_el.text = title or shelfmark or "Manuscrit médiéval"

    source_desc = ET.SubElement(file_desc, "sourceDesc")
    ms_desc = ET.SubElement(source_desc, "msDesc")
    ms_id = ET.SubElement(ms_desc, "msIdentifier")
    ms_id.text = shelfmark or "Inconnu"

    # --- text / body ---
    text_el = ET.SubElement(tei, "text")
    body = ET.SubElement(text_el, "body")
    div = ET.SubElement(body, "div")

    # Injection des entités dans le texte
    if not entities:
        p = ET.SubElement(div, "p")
        p.text = text_normalized
    else:
        # Trier les entités par position de début
        sorted_ents = sorted(entities, key=lambda e: e.get("start", 0))
        p = ET.SubElement(div, "p")
        cursor = 0

        for ent in sorted_ents:
            start = ent.get("start", 0)
            end = ent.get("end", 0)
            label = ent.get("entity_group", "MISC")
            word = ent.get("word", "")
            tag_name = TAG.get(label, "name")

            # Texte avant l'entité
            if start > cursor:
                if len(p) == 0:
                    p.text = (p.text or "") + text_normalized[cursor:start]
                else:
                    last = list(p)[-1]
                    last.tail = (last.tail or "") + text_normalized[cursor:start]

            # Élément entité
            ent_el = ET.SubElement(p, tag_name)
            ent_el.text = word
            ent_el.set("type", label)
            cursor = end

        # Texte restant après la dernière entité
        if cursor < len(text_normalized):
            if len(p) == 0:
                p.text = (p.text or "") + text_normalized[cursor:]
            else:
                last = list(p)[-1]
                last.tail = (last.tail or "") + text_normalized[cursor:]

    return tei


def export_tei(
    text_normalized: str,
    entities: list[dict],
    output_path: str,
    shelfmark: str = "",
    title: str = "",
) -> str:
    """Exporte un document TEI-XML sur disque.

    Args:
        text_normalized: Texte normalisé à exporter.
        entities: Entités NER avec start/end/word/entity_group.
        output_path: Chemin du fichier .xml de sortie.
        shelfmark: Cote du manuscrit source.
        title: Titre du document.

    Returns:
        Chemin du fichier écrit.

    Raises:
        OSError: Si le fichier ne peut pas être écrit.

    Example:
        >>> path = export_tei("Jehan est a Paris", [], "/tmp/test.xml")
        >>> path
        "/tmp/test.xml"
    """
    tei = build_tei(text_normalized, entities, shelfmark=shelfmark, title=title)
    xml_str = _pretty(tei)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    return output_path
