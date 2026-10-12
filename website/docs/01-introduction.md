---
id: introduction
title: Introduction et Contexte Medical
sidebar_position: 1
---

# Introduction et Contexte Medical

## Vue d'ensemble du Bobathon ESILV

Le projet **Bobathon ESILV** est une competition privee d'apprentissage automatique de haut niveau organisee conjointement par **Probabl** et **IBM**. L'objectif central est d'aider la recherche medicale a concevoir de meilleures evaluations de la maladie de Parkinson en corrigeant les biais cliniques et d'observation dans l'echelle motrice de reference.

:::info Objectif Principal
Pour chaque visite clinique d'un patient donne, predire le score moteur theorique impartial (**true OFF MDS-UPDRS**), en minimisant l'erreur quadratique moyenne (**RMSE**) sur un jeu de test dont les patients sont completement disjoints de l'entrainement.
:::

---

## La Maladie de Parkinson et la Denervation Dopaminergique

La maladie de Parkinson est la deuxieme affection neurodegenerative la plus frequente a l'echelle mondiale apres la maladie d'Alzheimer. Elle touche plus de 10 millions de personnes sur Terre, dont environ 200 000 en France avec 25 000 nouveaux cas diagnostiques chaque annee.

Les symptomes moteurs cardinaux regroupent :
- **L'akinesie et la bradykinesie** : lenteur d'initiation et de realisation des mouvements volontaires.
- **La rigidite plastique** : resistance continue aux mouvements passifs d'un membre.
- **Le tremblement de repos** : oscillation involontaire a basse frequence (4 a 6 Hz).

Ces troubles derivent de la perte progressive des neurones dopaminergiques de la substance noire (*substantia nigra pars compacta*). La dopamine joue un role regulateur fondamental dans les boucles cortico-sous-corticales des noyaux gris centraux (striatum, globus pallidus, noyau sous-thalamique).

![Anatomie et mecanismes de la maladie de Parkinson](/figures/figure1_anatomy.png)
*Figure 1 : Anatomie de la maladie de Parkinson, voies dopaminergiques et transmission synaptique.*

---

## Le Traitement Dopaminergique et les Fluctuations Motrices

La pierre angulaire de la prise en charge therapeutique est l'administration de precurseurs de dopamine, notamment la **L-Dopa** (ou Levodopa), souvent quantifiee par la dose journaliere equivalente en Levodopa (**LEDD**).

La reponse au traitement evolue au fil des annees de progression de la maladie :

1. **Phase initiale (lune de miel)** : Les neurones residuels tamponnent efficacement la dopamine exogene. Le controle moteur reste stable tout au long de la journee.
2. **Phase intermediaire (wearing-off)** : La capacite de stockage neuronale s'amenuise. L'effet de chaque prise s'attenue prematurement avant la dose suivante.
3. **Phase avancee (fluctuations motrices et dyskinesies)** : La fenetre therapeutique se retrecit drastiquement. Le patient alterne rapidement entre :
   - **L'etat ON** : phase d'efficacite therapeutique ou la motricite est fluide.
   - **L'etat OFF** : chute de la concentration plasmique de dopamine avec reemergence aigue des blocages et tremblements.
   - **Les dyskinesies** : mouvements involontaires excessifs survenant au pic de concentration medicamenteuse.

![Reponse au traitement selon les stades](/figures/figure2_treatment_response.png)
*Figure 2 : Evolution de la fenetre therapeutique et apparition des phases d'epuisement (wearing-off) et de blocage OFF.*

---

## L'Evaluation Motrice : L'Echelle MDS-UPDRS

Pour quantifier la gravite motrice, la communaute neurologique internationale utilise la partie III de l'echelle **MDS-UPDRS** (*Movement Disorder Society-Unified Parkinson's Disease Rating Scale*).

- L'examen comporte **18 items moteurs** evalues de 0 (normal) a 4 (tres severe).
- Il couvre les mouvements faciaux, le langage, la posture, la marche, la stabilite posturale ainsi que la bradykinesie et la rigidite de chaque membre.
- Au total, l'echelle aggrege **33 sous-scores** pour former une note globale comprise entre **0 et 132**.

![Item de l'echelle motrice MDS-UPDRS](/figures/figure3_mds_updrs.png)
*Figure 3 : Exemple d'item standardise de l'examen moteur MDS-UPDRS.*

---

## Le Probleme Scientifique : Le Biais du Score OFF Observe

Dans les protocoles de recherche clinique, l'etat moteur en condition **OFF** (sans effet du traitement) est considere comme le meilleur biomarqueur indirect de l'etat reel de denervation dopaminergique.

Neanmoins, dans la pratique hospitaliere reelle, la mesure brute du score OFF est fortement biaisee :
1. **Subjectivite des evaluateurs (`rater_id`)** : Deux cliniciens distincts evaluant le meme patient a quelques minutes d'intervalle attribuent des scores pouvant varier significativement.
2. **Mesures manquantes** : L'examen en phase OFF exigeant que le patient interrompe son traitement pendant plus de 12 heures, il est eprouvant et inconfortable. Par consequent, il est tres frequemment omis (42% de valeurs manquantes dans notre jeu d'entrainement).
3. **Fluctuation de la concentration sanguine** : Le delai reel entre la derniere prise medicamenteuse et l'evaluation (`time_since_intake_off`, `time_since_intake_on`) module fortement l'intensite des symptomes.

Pour pallier ces ecueils, le jeu de donnees du Bobathon integre un score theorique non biaise estime : le **true OFF**.

![Trajectoire d'un patient et cible true OFF](/figures/figure4_score_evolution.png)
*Figure 4 : Trajectoire temporelle d'un patient, scores mesures ON/OFF, periodes de LEDD et cible impartiale true OFF.*

La mission des modeles de Machine Learning developpes au cours de ce lab consiste donc a modeliser ce processus de debiaisage et a reconstruire fidelement la trajectoire du true OFF pour chaque patient inconnu du jeu de test.
