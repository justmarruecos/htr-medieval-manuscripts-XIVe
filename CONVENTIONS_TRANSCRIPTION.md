# Conventions de Transcription (Corpus CATMuS - Français XIVe/XVe)

Pour ce projet, nous avons adopté une approche de transcription **semi-diplomatique**, basée sur les recommandations du projet CATMuS, afin d'optimiser l'entraînement du modèle HTR tout en conservant la structure historique.

## 1. Niveau de transcription
* **Semi-diplomatique** : Le texte est transcrit tel qu'il apparaît sur le manuscrit, en conservant l'orthographe d'origine, mais en normalisant certains aspects graphiques pour faciliter l'apprentissage machine.

## 2. Traitement des abréviations
* Les abréviations médiévales (tildes, boucles, lettres suscrites) sont **développées silencieusement**. 
* *Justification* : Le modèle HTR doit apprendre à associer un graphème complexe à son sens complet textuel (ex: un "q" avec un tilde est transcrit "que").

## 3. Casse et Ponctuation
* La casse (majuscules/minuscules) est conservée telle qu'elle apparaît dans le manuscrit original, notamment pour les lettrines.
* La ponctuation historique (points médians, virgules) est conservée. Aucune ponctuation moderne n'est ajoutée artificiellement.

## 4. Gestion des lacunes et dommages
* Les caractères illisibles en raison de dommages sur le parchemin (trous, taches sévères) ou d'une encre trop effacée sont encodés avec un marqueur spécifique ou ignorés si la ligne est jugée inexploitable pour l'entraînement.