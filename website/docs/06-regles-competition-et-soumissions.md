---
id: regles-competition-et-soumissions
title: Regles Officielles et Soumissions
sidebar_position: 6
---

# Regles Officielles et Soumissions Kaggle

La participation au Bobathon ESILV est soumise aux regles officielles editees par IBM et Probabl. Ces modalites garantissent l'equite entre equipes et l'authenticite des modeles deposes.

---

## Conditions d'Eligibilite et Chaine d'Outils

Tous les modeles et fichiers de prediction doivent etre conçus et executes conformement aux exigences du reglement :

:::caution Chaine d'Outils Obligatoire
1. **Bob IDE et Bob CLI** sont les seuls environnements et agents dans lesquels le code experimentale peut etre redige et execute.
2. **skore** est l'unique framework d'evaluation et de suivi d'experiences autorise (`skore.evaluate`, `skore.Project`).
3. Les librairies autorisees au sein de cet environnement comprennent notamment `scikit-learn`, `skrub`, `pandas` et `numpy`.
4. Tout notebook Kaggle autonome, script execute hors de Bob, ou agent externe (Cursor, Claude Code, GitHub Copilot Chat, ChatGPT, Gemini CLI) est strictement prohibe.
:::

---

## Tracabilite Obligatoire sur Skore Hub

Chaque soumission deversee sur le leaderboard Kaggle doit imperativement etre documentee par un rapport d'evaluation en ligne :

1. Lors de l'evaluation d'un modele, le script pousse le rapport sur l'espace d'equipe Skore Hub :
   ```python
   project = Project(name="bobathon-esilv", mode="hub", workspace=workspace)
   project.put("07_ensemble", report)
   ```
2. La commande affiche une URL unique de type :
   `https://skore.probabl.ai/workspace/estimations/...`
3. **Dans la description du fichier depose sur Kaggle**, l'URL complete de ce rapport doit obligatoirement etre renseignee.
4. Une soumission Kaggle depourvue d'URL ou pointant vers un rapport invalide est desactivee par les organisateurs, meme si Kaggle lui attribue un score.

---

## Format Standard du Fichier de Soumission

Les soumissions doivent strictement respecter la structure definie par `sample_submission.csv` :

- **Format** : Fichier CSV separe par des virgules.
- **En-tete** : `Index,target`
- **Contenu** : Exactement 11 013 lignes correspondant aux visites de `X_test.csv`.
- **Nommage recommande** : `submissions/<cle_rapport>.csv` (par exemple `submissions/07_ensemble.csv`) pour eviter tout ecrasement involontaire entre experiences.

Exemple d'apercu :
```csv
Index,target
44590,36.42
44591,41.18
44592,29.80
...
```

---

## Leaderboard Public vs Prive

- **Leaderboard Public** : Calcule en temps reel durant la competition sur une tranche de patients du jeu de test.
- **Leaderboard Prive** : Révélé a la cloture de l'evenement sur les patients restants.
- **Selection finale** : Chaque equipe designe une (1) soumission officielle finale pour le calcul du classement definitif.
