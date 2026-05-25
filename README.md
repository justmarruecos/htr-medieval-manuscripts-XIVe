📜 Pipeline HTR : Reconnaissance de Manuscrits Médiévaux (XIVe-XVe siècles)
Ce dépôt contient l'intégralité d'un pipeline d'ingénierie de données et de Machine Learning (HTR - Handwritten Text Recognition) développé pour la transcription automatique de manuscrits et registres du Moyen Français.

Ce projet vise à démontrer la maîtrise d'un pipeline de bout en bout : de la numérisation brute jusqu'à la production d'un contrat de données JSON.

🛠️ Architecture et Environnement Technique
Système d'exploitation : WSL (Ubuntu 24.04) sous Windows. Choix architectural indispensable pour assurer la stabilité des dépendances natives de kraken (notamment coremltools).

Hardware : Accélération GPU via NVIDIA GeForce RTX 4060 (cuda:0).

Stack Principale : Python 3.12, Kraken (ketos), OpenCV, scikit-image, BeautifulSoup (parsing XML), pandas.

🚀 État d'Avancement et Réalisations
✅ Étape 1 : Ingénierie du Corpus (32 465 lignes)
Pour répondre à l'exigence de diversité documentaire et dialectale, le dataset a été forgé à partir de trois sources distinctes, nécessitant des stratégies d'extraction hétérogènes :

CATMuS Medieval (Français XIVe/XVe) : 25 812 lignes extraites via l'API HuggingFace, formant le cœur du réseau. (Licence : CC-BY 4.0 / Etalab 2.0).

CREMMA Médiéval : 2 956 lignes extraites par un parseur BeautifulSoup sur-mesure depuis les fichiers ALTO/PAGE-XML. Apporte la complexité des enluminures et des lettrines. (Licence : CC-BY 4.0).

e-NDP (Registres du chapitre de Notre-Dame) : 3 697 lignes extraites de la même manière depuis Zenodo. Apporte le défi majeur de la structure tabulaire et des écritures abrégées. (Licence : CC-BY-NC-SA).

Note : L'ajout de CREMMA et e-NDP a permis de corriger un biais temporel identifié lors de l'analyse exploratoire, ramenant la distribution temporelle à un équilibre parfait (50.7% pour le XVe s. contre 49.3% pour le XIVe s).

✅ Étape 2 : Computer Vision, Prétraitement et Segmentation
Le pipeline de traitement d'image a été confronté à deux types de documents radicalement différents, nécessitant une optimisation dynamique des hyperparamètres :

Amélioration du contraste : Application systématique du filtre CLAHE (clipLimit=2.0, tileGridSize=(8,8)).

Binarisation adaptative (Sauvola) : Ajustement critique du paramètre de sensibilité k. Un k=0.1 a été défini pour récupérer les encres pâles sur les manuscrits denses (CREMMA), tandis qu'un k=0.2 a été rendu obligatoire pour éviter la génération de bruit sur les grands espaces vides des registres tabulaires (e-NDP).

Segmentation Structurelle : Détection réussie des lignes de base (baselines) et des polygones via l'IA Kraken BLLA, isolant parfaitement les colonnes et contournant les enluminures.

🚧 Étape 3 : Évaluation Zéro-Shot et Fine-Tuning (En cours)
Baseline (Zéro Fine-tuning) : Le modèle Tridis_Medieval_EarlyModern a été évalué sur les données brutes. L'inférence a démontré une excellente reconnaissance sémantique du vieux français, mais une incapacité totale à interpréter la mise en page tabulaire de e-NDP (génération de bruit et de sauts de lignes erronés), justifiant scientifiquement le besoin de réentraînement.

Stratégie de Fine-Tuning : L'ingestion des données par ketos train a nécessité une refonte du formatage. Pour contourner les limites systèmes de Linux (Argument list too long) et les restrictions de parsing de l'API Kraken, l'intégralité du corpus a été re-packagée via un script Python générant des structures PAGE-XML intégrant des coordonnées géométriques factices. Le calcul intensif sur GPU est imminent.

🗺️ Feuille de Route (Prochaines Étapes)
Calcul GPU : Lancement du fine-tuning adaptatif de Kraken (--resize both) sur les 32 465 fichiers PAGE-XML générés.

Métriques HTR : Calcul comparatif du CER (Character Error Rate) et du WER (Word Error Rate) entre la baseline Tridis et le modèle fine-tuné sur le split d'évaluation scellé.

Architecture alternative : Implémentation optionnelle d'un fine-tuning TrOCR avec LoRA pour comparer les architectures récurrentes vs. Transformers (Objectif : validation scientifique et bonus d'évaluation).

Data Contract JSON : Assemblage des prédictions finales dans un schéma JSON validé, intégrant un tag de confiance (needs_review) pour isoler les dégradations irrémédiables.