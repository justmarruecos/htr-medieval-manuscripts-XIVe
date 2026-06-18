"""Module NER pour manuscrits médiévaux français XIVe-XVe siècle.

Utilise le modèle Jean-Baptiste/camembert-ner en mode zéro-shot.
"""

from __future__ import annotations

from typing import Optional
from transformers import pipeline, Pipeline


class NERPipeline:
    """Pipeline NER zéro-shot basé sur CamemBERT-NER.

    Args:
        model_name: Identifiant HuggingFace du modèle NER.
        device: Dispositif d'inférence (-1=CPU, 0=GPU).

    Example:
        >>> ner = NERPipeline()
        >>> ner.predict("Jehan Rousseau demeure a Paris")
        [{'word': 'Jehan Rousseau', 'entity_group': 'PER', 'score': 0.91}]
    """

    MODEL_NAME_DEFAULT = "Jean-Baptiste/camembert-ner"

    def __init__(
        self,
        model_name: str = MODEL_NAME_DEFAULT,
        device: int = -1,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._pipe: Optional[Pipeline] = None

    def _load(self) -> None:
        """Charge le modèle en mémoire (lazy loading).

        Raises:
            OSError: Si le modèle ne peut pas être téléchargé.
        """
        if self._pipe is None:
            self._pipe = pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple",
                device=self.device,
            )

    def predict(self, text: str) -> list[dict]:
        """Prédit les entités nommées dans un texte.

        Args:
            text: Texte à analyser (texte normalisé recommandé).

        Returns:
            Liste de dicts avec clés : word, entity_group, score,
            start, end.

        Raises:
            ValueError: Si text est vide.

        Example:
            >>> ner = NERPipeline()
            >>> results = ner.predict("Denis du Vergier est a Paris")
            >>> results[0]['entity_group']
            'PER'
        """
        if not text or not text.strip():
            raise ValueError("Le texte ne peut pas être vide.")
        self._load()
        return self._pipe(text)

    def predict_batch(self, texts: list[str]) -> list[list[dict]]:
        """Prédit les entités sur une liste de textes.

        Args:
            texts: Liste de textes à analyser.

        Returns:
            Liste de listes d'entités, une par texte.

        Raises:
            ValueError: Si texts est vide.

        Example:
            >>> ner = NERPipeline()
            >>> results = ner.predict_batch(["Jehan est a Paris", "Denis va a Lyon"])
            >>> len(results)
            2
        """
        if not texts:
            raise ValueError("La liste de textes ne peut pas être vide.")
        self._load()
        return [self._pipe(t) for t in texts]

    def evaluate(
        self,
        texts: list[str],
        gold_labels: list[list[dict]],
    ) -> dict:
        """Évalue les prédictions contre des labels gold.

        Compare uniquement les types d'entités (PER, LOC, ORG, MISC).
        Calcule précision, rappel et F1 au niveau des spans exacts.

        Args:
            texts: Liste de textes à évaluer.
            gold_labels: Liste de listes de dicts gold avec clés
                word et entity_group.

        Returns:
            Dict avec précision, recall, f1, n_pred, n_gold.

        Raises:
            ValueError: Si texts et gold_labels ont des longueurs différentes.

        Example:
            >>> ner = NERPipeline()
            >>> metrics = ner.evaluate(
            ...     ["Jehan est a Paris"],
            ...     [[{"word": "Jehan", "entity_group": "PER"}]]
            ... )
            >>> "f1" in metrics
            True
        """
        if len(texts) != len(gold_labels):
            raise ValueError(
                f"texts ({len(texts)}) et gold_labels ({len(gold_labels)}) "
                "doivent avoir la même longueur."
            )
        self._load()

        n_correct = 0
        n_pred = 0
        n_gold = 0

        for text, gold in zip(texts, gold_labels):
            preds = self._pipe(text)
            pred_spans = {(e["word"].lower(), e["entity_group"]) for e in preds}
            gold_spans = {(e["word"].lower(), e["entity_group"]) for e in gold}

            n_correct += len(pred_spans & gold_spans)
            n_pred += len(pred_spans)
            n_gold += len(gold_spans)

        precision = n_correct / n_pred if n_pred > 0 else 0.0
        recall = n_correct / n_gold if n_gold > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_pred": n_pred,
            "n_gold": n_gold,
            "n_correct": n_correct,
        }
