"""Tests pytest pour le module NER (src/ner.py).

Cible : 8+ tests PASSED.
"""

import pytest
from src.ner import NERPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ner():
    """Instance NERPipeline partagée entre les tests (chargement unique)."""
    return NERPipeline(device=-1)


# ---------------------------------------------------------------------------
# Tests chargement
# ---------------------------------------------------------------------------

def test_pipeline_instanciation():
    """Le constructeur ne charge pas le modele avant le premier appel."""
    pipe = NERPipeline()
    assert pipe._pipe is None


def test_pipeline_lazy_loading(ner):
    """Le modele est charge apres le premier predict."""
    ner.predict("Jehan Rousseau demeure a Paris")
    assert ner._pipe is not None


def test_model_name_default():
    """Le nom de modele par defaut est correct."""
    pipe = NERPipeline()
    assert pipe.model_name == "Jean-Baptiste/camembert-ner"


def test_model_name_custom():
    """Un nom de modele custom est bien stocke."""
    pipe = NERPipeline(model_name="Jean-Baptiste/camembert-ner-agr")
    assert pipe.model_name == "Jean-Baptiste/camembert-ner-agr"


# ---------------------------------------------------------------------------
# Tests predict
# ---------------------------------------------------------------------------

def test_predict_returns_list(ner):
    """predict() retourne une liste."""
    result = ner.predict("Jehan Rousseau demeure a Paris")
    assert isinstance(result, list)


def test_predict_entities_not_empty(ner):
    """predict() trouve au moins une entite sur un texte medievale connu."""
    result = ner.predict("Jehan Rousseau demeure a Paris")
    assert len(result) > 0


def test_predict_entity_has_required_keys(ner):
    """Chaque entite retournee contient les cles attendues."""
    result = ner.predict("Denis du Vergier est receveur a Paris")
    for entity in result:
        assert "word" in entity
        assert "entity_group" in entity
        assert "score" in entity


def test_predict_per_entity_detected(ner):
    """Une entite PER est detectee sur un nom medieval connu."""
    result = ner.predict("Pierre Norrisson chevalier du roi")
    entity_groups = [e["entity_group"] for e in result]
    assert "PER" in entity_groups


def test_predict_empty_text_raises(ner):
    """predict() leve ValueError sur texte vide."""
    with pytest.raises(ValueError):
        ner.predict("")


def test_predict_whitespace_raises(ner):
    """predict() leve ValueError sur texte blanc."""
    with pytest.raises(ValueError):
        ner.predict("   ")


# ---------------------------------------------------------------------------
# Tests predict_batch
# ---------------------------------------------------------------------------

def test_predict_batch_returns_list_of_lists(ner):
    """predict_batch() retourne une liste de listes."""
    texts = ["Jehan Rousseau est a Paris", "Denis du Vergier va a Lyon"]
    results = ner.predict_batch(texts)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)


def test_predict_batch_empty_raises(ner):
    """predict_batch() leve ValueError sur liste vide."""
    with pytest.raises(ValueError):
        ner.predict_batch([])


# ---------------------------------------------------------------------------
# Tests evaluate
# ---------------------------------------------------------------------------

def test_evaluate_returns_metrics(ner):
    """evaluate() retourne un dict avec les cles de metriques."""
    texts = ["Jehan Rousseau demeure a Paris"]
    gold = [[{"word": "Jehan Rousseau", "entity_group": "PER"}]]
    metrics = ner.evaluate(texts, gold)
    for key in ("precision", "recall", "f1", "n_pred", "n_gold", "n_correct"):
        assert key in metrics


def test_evaluate_mismatched_lengths_raises(ner):
    """evaluate() leve ValueError si texts et gold_labels ont des longueurs differentes."""
    with pytest.raises(ValueError):
        ner.evaluate(["texte1", "texte2"], [[]])


def test_evaluate_f1_between_0_and_1(ner):
    """Le F1 retourne par evaluate() est entre 0 et 1."""
    texts = ["Jehan Rousseau demeure a Paris"]
    gold = [[{"word": "Jehan Rousseau", "entity_group": "PER"}]]
    metrics = ner.evaluate(texts, gold)
    assert 0.0 <= metrics["f1"] <= 1.0
