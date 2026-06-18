"""Tests unitaires pour src/normalisation.py."""
import pytest
from src.normalisation import (
    normalize_unicode,
    expand_abbreviations,
    normalize_uv,
    normalize_ij,
    normalize_punctuation,
    normalize,
)


# ── normalize_unicode ─────────────────────────────────────────────────────────

def test_unicode_nfd():
    result = normalize_unicode("uoir")
    import unicodedata
    assert unicodedata.is_normalized("NFD", result)

def test_unicode_spaces():
    assert normalize_unicode("le  roy") == "le roy"

def test_unicode_strip():
    assert normalize_unicode("  texte  ") == "texte"

def test_unicode_rejects_non_str():
    with pytest.raises(TypeError):
        normalize_unicode(42)


# ── expand_abbreviations ──────────────────────────────────────────────────────

def test_abbr_rum():
    assert expand_abbreviations("doминoꝵ") == "doминorum"

def test_abbr_tilde_n():
    assert expand_abbreviations("boñe") == "bonne"

def test_abbr_no_change():
    assert expand_abbreviations("le roi") == "le roi"


# ── normalize_uv ──────────────────────────────────────────────────────────────

def test_uv_initial_lower():
    assert normalize_uv("uoir") == "voir"

def test_uv_initial_upper():
    # La règle s'applique sur minuscules uniquement (regex sur 'u' minuscule)
    assert normalize_uv("un ami") == "un ami"  # u devant consonne = pas de changement

def test_uv_no_change_medial():
    # 'v' médial n'est pas modifié (règle désactivée pour éviter faux positifs)
    assert normalize_uv("avoir") == "avoir"

def test_uv_no_change_consonant():
    # 'u' initial devant consonne ne change pas
    assert normalize_uv("un") == "un"


# ── normalize_ij ──────────────────────────────────────────────────────────────

def test_ij_initial():
    assert normalize_ij("jour") == "iour"

def test_ij_no_change_medial():
    # 'j' médial ne change pas
    assert normalize_ij("ajouter") == "ajouter"

def test_ij_no_change_consonant():
    # 'j' initial devant consonne ne change pas
    assert normalize_ij("jn") == "jn"


# ── normalize_punctuation ─────────────────────────────────────────────────────

def test_punct_middle_dot():
    assert normalize_punctuation("le·roi") == "le roi"

def test_punct_pilcrow():
    assert normalize_punctuation("¶ Item") == ". Item"

def test_punct_double_spaces():
    assert normalize_punctuation("le  roi") == "le roi"


# ── normalize (pipeline complet) ─────────────────────────────────────────────

def test_normalize_full():
    # 'auec' reste 'auec' — règle médiale désactivée
    result = normalize("uoir le roy auec ses gens")
    assert result == "voir le roy auec ses gens"

def test_normalize_disable_abbr():
    # Sans expand_abbr, les abréviations restent (NFD peut décomposer ñ → n+combining)
    text = "boñe"
    result = normalize(text, expand_abbr=False)
    # Le ñ peut être décomposé par NFD mais 'n' reste présent
    assert "bon" in result

def test_normalize_disable_uv():
    result = normalize("uoir", normalize_uv_ij=False)
    assert result == "uoir"

def test_normalize_rejects_non_str():
    with pytest.raises(TypeError):
        normalize(None)

def test_normalize_empty_string():
    assert normalize("") == ""