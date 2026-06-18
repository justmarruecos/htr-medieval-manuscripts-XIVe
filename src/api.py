"""API REST pour le pipeline NLP manuscrits médiévaux.

Endpoints :
    POST /analyze  : normalisation + NER sur un texte
    GET  /health   : statut de l'API
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.normalisation import normalize
from src.ner import NERPipeline


app = FastAPI(
    title="HTR Medieval NLP API",
    description="Pipeline NLP pour manuscrits médiévaux français XIVe-XVe siècle.",
    version="1.0.0",
)

_ner_pipeline: NERPipeline | None = None


def get_ner_pipeline() -> NERPipeline:
    """Retourne l'instance NERPipeline (singleton lazy).

    Returns:
        Instance NERPipeline chargée.
    """
    global _ner_pipeline
    if _ner_pipeline is None:
        _ner_pipeline = NERPipeline(device=-1)
    return _ner_pipeline


class AnalyzeRequest(BaseModel):
    """Schéma de la requête POST /analyze.

    Attributes:
        text: Texte brut à analyser (transcription HTR).
    """
    text: str


class EntityOut(BaseModel):
    """Entité nommée détectée.

    Attributes:
        word: Forme du mot détecté.
        entity_group: Type d'entité (PER, LOC, ORG, MISC).
        score: Score de confiance entre 0 et 1.
        start: Offset caractère de début.
        end: Offset caractère de fin.
    """
    word: str
    entity_group: str
    score: float
    start: int
    end: int


class AnalyzeResponse(BaseModel):
    """Schéma de la réponse POST /analyze.

    Attributes:
        original: Texte original reçu.
        normalized: Texte après normalisation orthographique.
        entities: Liste des entités nommées détectées.
        n_entities: Nombre d'entités détectées.
    """
    original: str
    normalized: str
    entities: list[EntityOut]
    n_entities: int


class HealthResponse(BaseModel):
    """Schéma de la réponse GET /health.

    Attributes:
        status: Statut de l'API.
        model: Nom du modèle NER utilisé.
    """
    status: str
    model: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Vérifie que l'API est opérationnelle.

    Returns:
        HealthResponse avec status ok et nom du modèle.
    """
    return HealthResponse(
        status="ok",
        model=NERPipeline.MODEL_NAME_DEFAULT,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Normalise un texte et extrait les entités nommées.

    Args:
        request: Requête contenant le texte brut.

    Returns:
        AnalyzeResponse avec texte normalisé et entités détectées.

    Raises:
        HTTPException 422: Si le texte est vide.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=422, detail="Le texte ne peut pas être vide.")

    normalized = normalize(request.text)
    ner = get_ner_pipeline()
    raw_entities = ner.predict(normalized)

    entities = [
        EntityOut(
            word=e["word"],
            entity_group=e["entity_group"],
            score=round(float(e["score"]), 4),
            start=e.get("start", 0),
            end=e.get("end", 0),
        )
        for e in raw_entities
    ]

    return AnalyzeResponse(
        original=request.text,
        normalized=normalized,
        entities=entities,
        n_entities=len(entities),
    )
