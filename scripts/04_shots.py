"""Phase 6 — shot features + xFG model."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src import config as C
from src.shots import features as SF, model as SM

E = C.PROCESSED_DIR / "events"
T = {p.stem: pd.read_parquet(p) for p in E.glob("*.parquet")}
sp = pd.read_parquet(C.DERIVED_DIR / "spatial_shots.parquet")

sh = SF.build_shot_features(T, sp)
print("shot features:", sh.shape)
print("\ncreation_type:\n", sh.creation_type.value_counts().to_string())

fit = sh[sh.is_box_score_fga & sh.location_x.notna()].copy()
print(f"\nmodelling set: {len(fit)} box-score FGA (base FG% {fit.outcome.mean():.3f})")

res = {}
for tag, fn, spat in [("logistic+spatial", SM.make_logistic, True),
                      ("logistic_no_spatial", lambda: SM.make_logistic(False), False),
                      ("gbm", SM.make_gbm, True)]:
    r = SM.cross_val_scores(fit, fn, with_spatial=spat)
    oof = r.pop("oof")
    if tag == "logistic+spatial":
        fit["xfg"] = oof
    res[tag] = r
    print(f"  {tag:22s} logloss {r['log_loss']:.4f} (base {r['log_loss_baseline']:.4f})  "
          f"brier {r['brier']:.4f} (base {r['brier_baseline']:.4f})  auc {r['auc']:.3f}")

# SkillCorner's own shotQuality as a benchmark on the same rows
sq = fit.dropna(subset=["shot_quality"])
from sklearn.metrics import roc_auc_score, brier_score_loss
sq_auc = roc_auc_score(sq.outcome.astype(int), sq.shot_quality)
print(f"  {'SkillCorner shotQuality':22s} auc {sq_auc:.3f}  (n={len(sq)}) "
      f"corr with our xFG: {sq.shot_quality.corr(sq.xfg):.3f}")
res["skillcorner_shotquality_auc"] = float(sq_auc)
res["skillcorner_vs_our_xfg_corr"] = float(sq.shot_quality.corr(sq.xfg))

model = SM.fit_full(fit)
coefs = SM.coefficient_report(model)
print("\nTop coefficients (log-odds):")
print(coefs.head(16).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

cal = SM.calibration_table(fit.outcome.astype(int).to_numpy(), fit.xfg.to_numpy())
print("\nCalibration (out-of-fold):")
print(cal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# score every shot (incl. fouled misses) with the full-sample model for display
Xall, _ = SM.design_frame(sh)
sh["xfg"] = model.predict_proba(Xall)[:, 1]
sh["xpts"] = sh.xfg * sh.points_if_made
sh["shot_value_added"] = sh.points - sh.xpts

sh.to_parquet(C.DERIVED_DIR / "shots_features.parquet", index=False)
coefs.to_parquet(C.DERIVED_DIR / "xfg_coefficients.parquet", index=False)
cal.assign(bin=cal.bin.astype(str)).to_parquet(C.DERIVED_DIR / "xfg_calibration.parquet", index=False)
Path(C.DERIVED_DIR / "xfg_metrics.json").write_text(json.dumps(res, indent=1))
print("\nwrote shots_features.parquet", sh.shape)
