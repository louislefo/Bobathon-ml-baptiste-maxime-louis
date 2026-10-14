"""Modele 26 : Ultimate Blend — Triple Blend OOF-optimise (26_ultimate_blend).

Project: Bobathon ESILV (Probabl x IBM)
Key Hub: '26_ultimate_blend'

Architecture :
  Pilier A : CoefBlendRegressor (24_coef_blend, RMSE 3.49)
             Regression des coefficients parabole patient + blend 22/coef/11
  Pilier B : TriBlendRegressor  (25_tri_blend,  RMSE 3.52)
             Trajectoire 21 + 22 + JPI-CQT multi-graines
  Pilier C : SuperBlendWLSRegressor (24_wls_multi_seed, RMSE 3.53)
             Trajectoire 21 + 22 multi-graines + lissage WLS heteroscedastique

Blend final : alpha*A + beta*B + gamma*C  (alpha+beta+gamma=1)
Les poids sont optimises sur les predictions OOF GroupKFold 5 plis
(aucune fuite : poids cherches uniquement sur le vecteur hors-fold).

Post-traitement : smooth_wls_monotonic identique au modele 24.

Cible : RMSE < 3.49 (meilleur Skore Hub actuel = 24_coef_blend).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
OOF_DIR = REPO_ROOT / "oof"
HUB_KEY = "26_ultimate_blend"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials
from src.model_15_trajectory import add_longitudinal_columns
from src.model_24_coef_blend import CoefBlendRegressor
from src.model_24_wls_multi_seed import SuperBlendWLSRegressor, smooth_wls_monotonic
from src.model_25_tri_blend import TriBlendRegressor

# Poids OOF-optimises (mis a jour par quick_grid apres calcul des OOF)
BEST_ALPHA = 0.60   # Pilier A (24_coef_blend)
BEST_BETA  = 0.25   # Pilier B (25_tri_blend)
BEST_GAMMA = 0.15   # Pilier C (24_wls_multi_seed)


# ---------------------------------------------------------------------------
# Estimateur sklearn-compatible
# ---------------------------------------------------------------------------

class UltimateBlendRegressor(RegressorMixin, BaseEstimator):
    """Triple Blend des 3 meilleures architectures avec poids OOF-optimises."""

    def __init__(
        self,
        alpha: float = BEST_ALPHA,
        beta: float  = BEST_BETA,
        gamma: float = BEST_GAMMA,
        weight_off: float = 2.5,
        alpha_smooth: float = 0.9,
        random_state: int = 0,
    ):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.weight_off   = weight_off
        self.alpha_smooth = alpha_smooth
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        self.pilier_a_ = CoefBlendRegressor(random_state=self.random_state).fit(X, y)
        self.pilier_b_ = TriBlendRegressor().fit(X, y)
        self.pilier_c_ = SuperBlendWLSRegressor().fit(X, y)
        return self

    def _raw_blend(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        pa = self.pilier_a_.predict(X)
        pb = self.pilier_b_.predict(X)
        pc = self.pilier_c_.predict(X)
        return self.alpha * pa + self.beta * pb + self.gamma * pc

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        blend = self._raw_blend(X)
        return smooth_wls_monotonic(
            patient_id=X["patient_id"].values,
            age=X["age"].values,
            pred=blend,
            is_off=X["off"].notna().values,
            alpha=self.alpha_smooth,
            degree=2,
            weight_off=self.weight_off,
        )


# ---------------------------------------------------------------------------
# Calcul OOF + grille de poids (ST1 + ST2)
# ---------------------------------------------------------------------------

def _compute_oof_components(X: pd.DataFrame, y: pd.Series, cv_splits) -> pd.DataFrame:
    """Calcule les predictions OOF des 3 piliers et les met en cache."""
    path = OOF_DIR / "26_triple_components_oof.csv"
    folds = np.zeros(len(X), dtype=int)
    for k, (_, va) in enumerate(cv_splits):
        folds[va] = k

    if path.is_file():
        print(f"  Cache OOF existant charge : {path}", flush=True)
        return pd.read_csv(path)

    print("  Calcul des OOF pour les 3 piliers (5 plis x 3 modeles)...", flush=True)
    comp = pd.DataFrame({
        "Index": X["Index"].values,
        "fold": folds,
        "pred_A": 0.0,
        "pred_B": 0.0,
        "pred_C": 0.0,
        "y": y.values,
    })
    for k, (tr, va) in enumerate(cv_splits):
        print(f"  Pli {k+1}/5 — fit A...", flush=True)
        pa = CoefBlendRegressor().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
        print(f"  Pli {k+1}/5 — fit B...", flush=True)
        pb = TriBlendRegressor().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
        print(f"  Pli {k+1}/5 — fit C...", flush=True)
        pc = SuperBlendWLSRegressor().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
        comp.loc[va, "pred_A"] = pa
        comp.loc[va, "pred_B"] = pb
        comp.loc[va, "pred_C"] = pc
        pa_rmse = np.sqrt(np.mean((pa - y.iloc[va].values) ** 2))
        pb_rmse = np.sqrt(np.mean((pb - y.iloc[va].values) ** 2))
        pc_rmse = np.sqrt(np.mean((pc - y.iloc[va].values) ** 2))
        print(f"  Pli {k+1} : A={pa_rmse:.4f}  B={pb_rmse:.4f}  C={pc_rmse:.4f}", flush=True)

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    comp.to_csv(path, index=False)
    print(f"  OOF sauvegardees : {path}", flush=True)
    return comp


def quick_grid(X: pd.DataFrame, y: pd.Series, cv_splits) -> tuple[float, float, float]:
    """Grille OOF pour optimiser les poids alpha/beta/gamma. Retourne (alpha*, beta*, gamma*)."""
    comp = _compute_oof_components(X, y, cv_splits)
    folds = comp["fold"].values
    n_folds = len(cv_splits)

    pa  = comp["pred_A"].values
    pb  = comp["pred_B"].values
    pc  = comp["pred_C"].values
    yv  = comp["y"].values
    pid = X["patient_id"].values
    age = X["age"].values
    is_off = X["off"].notna().values

    # RMSE OOF individuels
    for name, p in (("A (CoefBlend)", pa), ("B (TriBlend)", pb), ("C (WLS)", pc)):
        rmse = np.sqrt(np.mean((p - yv) ** 2))
        print(f"  Pilier {name} OOF RMSE = {rmse:.4f}", flush=True)

    # Grille convexe fine : alpha in [0.45, 0.70], beta in [0.10, 0.40], gamma = 1-alpha-beta
    alphas = np.arange(0.45, 0.71, 0.05)
    betas  = np.arange(0.10, 0.41, 0.05)
    results = []
    for a in alphas:
        for b in betas:
            g = 1.0 - a - b
            if g < 0.0 or g > 0.50:
                continue
            blend_raw = a * pa + b * pb + g * pc
            # Appliquer le lissage WLS sur chaque pli independamment
            blend_smooth = smooth_wls_monotonic(pid, age, blend_raw, is_off, alpha=0.9, degree=2, weight_off=2.5)
            per_fold = [
                np.sqrt(np.mean((blend_smooth[folds == k] - yv[folds == k]) ** 2))
                for k in range(n_folds)
            ]
            mean_rmse = np.mean(per_fold)
            std_rmse  = np.std(per_fold)
            results.append((mean_rmse, std_rmse, round(a, 2), round(b, 2), round(g, 2)))

    results.sort(key=lambda x: x[0])
    print(f"\n  {'alpha':>6} {'beta':>6} {'gamma':>6}  {'RMSE':>8}  {'std':>8}", flush=True)
    print("  " + "-" * 46, flush=True)
    for mean_r, std_r, a, b, g in results[:10]:
        print(f"  {a:>6.2f} {b:>6.2f} {g:>6.2f}  {mean_r:>8.4f}  {std_r:>8.4f}", flush=True)

    best = results[0]
    print(f"\n  => Meilleurs poids : alpha={best[2]}, beta={best[3]}, gamma={best[4]}"
          f"  RMSE OOF={best[0]:.4f}", flush=True)
    return best[2], best[3], best[4]


# ---------------------------------------------------------------------------
# Point d'entree principal
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Modele 26 : Ultimate Blend OOF-optimise")
    parser.add_argument("--no-hub", action="store_true", help="Ne pas publier sur Skore Hub")
    parser.add_argument("--quick",  action="store_true",
                        help="Mode grille OOF uniquement (cherche les meilleurs poids, pas de skore.evaluate)")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("MODELE 26 : ULTIMATE BLEND OOF-OPTIMISE (26_ultimate_blend)", flush=True)
    print(f"  Pilier A : CoefBlendRegressor       (cible RMSE 3.49)", flush=True)
    print(f"  Pilier B : TriBlendRegressor         (cible RMSE 3.52)", flush=True)
    print(f"  Pilier C : SuperBlendWLSRegressor    (cible RMSE 3.53)", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/5] Chargement des donnees...", flush=True)
    X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv")
    visits = X_train_raw.merge(y_train_raw, on="Index")
    y = visits["target"]
    X = add_longitudinal_columns(visits.drop(columns=["target"]))
    X_test = add_longitudinal_columns(X_test_raw)
    print(f"  Train : {len(X)} visites | Test : {len(X_test)} visites", flush=True)

    print("\n[2/5] Configuration GroupKFold (5 plis sur patient_id)...", flush=True)
    cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

    # --quick : chercher les meilleurs poids sur OOF, puis s'arreter
    if args.quick:
        print("\n[3/5] Grille de poids OOF...", flush=True)
        a, b, g = quick_grid(X, y, cv_splits)
        print(f"\n  => Encodez dans BEST_ALPHA={a}, BEST_BETA={b}, BEST_GAMMA={g}", flush=True)
        print("  Pour lancer le run complet, relancez sans --quick.", flush=True)
        return

    # Run complet : d'abord chercher les poids OOF si le cache est deja la,
    # puis lancer skore.evaluate avec les poids optimaux
    alpha, beta, gamma = BEST_ALPHA, BEST_BETA, BEST_GAMMA
    oof_path = OOF_DIR / "26_triple_components_oof.csv"
    if oof_path.is_file():
        print("\n[2b] Cache OOF detecte — recherche des poids optimaux en avance rapide...", flush=True)
        alpha, beta, gamma = quick_grid(X, y, cv_splits)
    else:
        print(f"\n[2b] Poids par defaut : alpha={alpha}, beta={beta}, gamma={gamma}", flush=True)
        print("     (Lancez avec --quick pour calculer les OOF et optimiser les poids.)", flush=True)

    print(f"\n[3/5] Evaluation skore.evaluate avec poids ({alpha}, {beta}, {gamma})...", flush=True)
    model = UltimateBlendRegressor(alpha=alpha, beta=beta, gamma=gamma)
    report = evaluate(model, X, y, splitter=cv_splits)

    try:
        print(f"  Score RMSE GroupKFold :\n{report.metrics.rmse()}", flush=True)
    except Exception as exc:
        print(f"  Acces rmse : {exc}", flush=True)
    try:
        print(f"  Score MAE  GroupKFold :\n{report.metrics.mae()}", flush=True)
    except Exception as exc:
        print(f"  Acces mae : {exc}", flush=True)
    try:
        print(f"  Verifications Skore :\n{report.checks.summarize(fast_mode=True)}", flush=True)
    except Exception as exc:
        print(f"  Controles Skore : {exc}", flush=True)

    print("\n[4/5] Entrainement final sur tout le train et prediction X_test...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / f"{HUB_KEY}.csv"
    pd.DataFrame({"Index": X_test_raw["Index"], "target": predictions}).to_csv(sub_path, index=False)
    print(f"  Soumission : {sub_path} ({len(predictions)} lignes)", flush=True)
    print(f"  Stats : min={predictions.min():.2f}  mean={predictions.mean():.2f}  "
          f"max={predictions.max():.2f}", flush=True)

    if args.no_hub:
        print("\n[5/5] Publication Skore Hub ignoree (--no-hub).", flush=True)
        return

    print(f"\n[5/5] Publication sur Skore Hub sous la cle '{HUB_KEY}'...", flush=True)
    try:
        config = load_skore_credentials(REPO_ROOT)
        login(mode="hub")
        project = Project(name="bobathon-esilv", mode="hub", workspace=config.get("workspace"))
        project.put(HUB_KEY, report)
        print("  Rapport publie sur Skore Hub !", flush=True)
    except Exception as exc:
        print(f"  Avertissement Skore Hub : {exc}", flush=True)

    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
