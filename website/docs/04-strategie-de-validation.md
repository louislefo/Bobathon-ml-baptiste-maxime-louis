---
id: strategie-de-validation
title: Strategie de Validation Croisee
sidebar_position: 4
---

# Strategie de Validation Croisee

La validite d'une demarche en Machine Learning repose sur la capacite du protocole d'evaluation a anticiper fidelement la performance sur les donnees non vues. Dans le contexte de donnees medicales longitudinales, ce protocole revet une criticite absolue.

---

## Le Piege du Decoupage Aleatoire (Data Leakage)

Chaque patient du jeu de donnees effectue entre 4 et 12 visites au cours de son suivi clinique. Les mesures consecutives d'un meme individu partagent des caracteristiques intrinseques stables : morphologie, genetique, vitesse d'evolution de la degenerescence, tolerance aux molecules.

Si l'on decoupe les donnees aleatoirement par ligne (K-Fold traditionnel) :
1. Certaines visites d'un patient $P$ se retrouvent dans le pli d'entrainement, tandis que d'autres visites du meme patient $P$ atterrissent dans le pli de test.
2. Le modele apprend alors l'identite sous-jacente du patient plutot que les lois generales d'evolution de la maladie.
3. Le score de validation croisee affiche un RMSE artificiellement tres bas (surestimation optimiste).
4. Lors de la soumission sur Kaggle (ou les patients du test sont integralement nouveaux), la performance s'effondre drastiquement.

:::caution Interdiction Formelle
Le partitionnement aleatoire par echantillon est formellement proscrit pour la selection et l'arbitrage des modeles.
:::

---

## Protocole Officiel : GroupKFold par `patient_id`

Pour reproduire fidelement la coupure du concours Kaggle, la validation croisee doit imperativement partitionner les donnees en isolant les patients :

```python
from sklearn.model_selection import GroupKFold
from skore import evaluate

cv = GroupKFold(n_splits=5)
report = evaluate(model, X, y, cv=cv, groups=patient_ids)
```

Grâce a ce decoupage :
- A chaque pli, 80 % des patients sont utilises pour l'apprentissage et 20 % des patients (et l'integralite de leurs visites) sont reserves exclusivement pour l'evaluation.
- Aucun patient n'est simultanement present dans l'entrainement et le test d'un meme pli.
- Le RMSE mesure sur les 5 plis reflete precisement le comportement du modele sur le leaderboard prive Kaggle.

---

## Metrique d'Evaluation : RMSE au Niveau Visite

La metrique officielle retenue pour le classement Kaggle est la racine de l'erreur quadratique moyenne (**Root Mean Squared Error**, RMSE) calculee sur l'ensemble des visites du jeu de test :

```text
RMSE(y, y_hat) = sqrt( (1 / N) * sum_{i=1}^N (y_i - y_hat_i)^2 )
```

Ou :
- `N` est le nombre total de visites du jeu de test (11 013 visites).
- `y_i` est le score impartial reel inconnu (*true OFF*).
- `y_hat_i` est la prediction generee par le modele soumis.

### Proprietes Statistiques
- La penalisation quadratique sanctionne plus severement les erreurs importantes (outliers) que l'ecart absolu moyen (MAE).
- Un predicteur constant predisant la moyenne globale obtient un RMSE egal a l'ecart-type empirique de la cible, soit environ 16.50. Ce score constitue le socle de base a battre des le Day 1.

---

## Le Cas Particulier du Stacking d'Ensemble

Lors de l'assemblage de plusieurs modeles (Ridge et HistGradientBoosting) au sein d'un `StackingRegressor`, les predictions hors-pli (*out-of-fold*) servant a entrainer le meta-modele doivent egalement respecter l'isolation des patients.

C'est pourquoi nous avons developpe la classe custom [`PatientGroupKFold`](file:///c:/Users/louis/OneDrive/Desktop/Esilv/A4/Machine%20Learning/Hackathon/bobathon-esilv/src/model_07_ensemble.py#L40-L58) :

```python
class PatientGroupKFold:
    """GroupKFold dynamique indexe sur les identifiants de patient."""

    def __init__(self, groups: pd.Series, n_splits: int = 5):
        self.groups = groups
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        return GroupKFold(n_splits=self.n_splits).split(
            X, y, groups=self.groups.loc[X.index]
        )
```

Ce splitter garantit que le meta-modele de stacking est lui-meme entraine sur des predictions de modeles de base qui n'ont jamais observe les patients evalues.
