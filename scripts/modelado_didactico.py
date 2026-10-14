import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 2026
rng = np.random.default_rng(SEED)
OUT = Path("results/modelado_didactico")
OUT.mkdir(parents=True, exist_ok=True)
FEATURES = ["intensity_mean", "texture_contrast", "edge_density"]
x = rng.normal(size=(240, 3))
df = pd.DataFrame(x, columns=FEATURES)
df["sample_id"] = [f"SIM_{i:03d}" for i in range(len(df))]
df["subject_id"] = df["sample_id"]
df["target_continuous"] = (
    10
    + 2 * x[:, 0]
    - 1.5 * x[:, 1]
    + 0.8 * x[:, 2]
    + 1.2 * x[:, 0] ** 2
    + rng.normal(0, 1.2, len(df))
)
probability = 1 / (1 + np.exp(-(-0.6 + 1.4 * x[:, 0] - x[:, 1])))
df["target_binary"] = rng.binomial(1, probability)
# Las etiquetas se crean antes de introducir ausencias en predictores.
df.loc[rng.random(len(df)) < 0.06, "texture_contrast"] = np.nan
# Partición fija por sujeto; no se repite buscando un resultado favorable.
order = rng.permutation(len(df))
df["split"] = "test"
df.loc[order[:144], "split"] = "train"
df.loc[order[144:192], "split"] = "validation"
df.to_csv(OUT / "features_sinteticas.csv", index=False)
train = df.loc[df["split"] == "train"].copy()
validation = df.loc[df["split"] == "validation"].copy()
# La prueba no se usa en ajuste, diagnóstico ni elección.


def continuous_pipeline(degree):
    """Imputa y transforma dentro del ajuste del conjunto recibido."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", LinearRegression()),
        ]
    )


def binary_pipeline(c):
    """Regresión logística L2 con predictores en escala comparable."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def regression_metrics(y, prediction):
    return {
        "MAE": float(mean_absolute_error(y, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
        "R2": float(r2_score(y, prediction)),
    }


def binary_metrics(y, probability, threshold=0.5):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("La evaluación binaria requiere ambas clases.")
    tn, fp, fn, tp = confusion_matrix(
        y, probability >= threshold, labels=[0, 1]
    ).ravel()
    return {
        "ROC_AUC": float(roc_auc_score(y, probability)),
        "AP": float(average_precision_score(y, probability)),
        "Brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, probability >= threshold)
        ),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "threshold": float(threshold),
    }


Xtr, Xva = train[FEATURES], validation[FEATURES]
ytr_r, yva_r = train["target_continuous"], validation["target_continuous"]
ytr_b, yva_b = train["target_binary"], validation["target_binary"]
for labels in (ytr_b, yva_b):
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Faltan clases; revisar protocolo sin mirar prueba.")

regression_models = {f"degree_{d}": continuous_pipeline(d) for d in (1, 2)}
classification_models = {f"C_{c}": binary_pipeline(c) for c in (0.1, 1.0)}
regression_models["baseline"] = DummyRegressor(strategy="mean")
classification_models["baseline"] = DummyClassifier(strategy="prior")
rows = []
for name, model in regression_models.items():
    model.fit(Xtr, ytr_r)
    rows.append(
        {
            "route": "continuous",
            "model": name,
            **regression_metrics(yva_r, model.predict(Xva)),
        }
    )
for name, model in classification_models.items():
    model.fit(Xtr, ytr_b)
    rows.append(
        {
            "route": "binary",
            "model": name,
            **binary_metrics(yva_b, model.predict_proba(Xva)[:, 1]),
        }
    )
comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / "validation_metrics.csv", index=False)
# Criterios predeclarados: RMSE continuo y AP binario.
best_r_name = min(
    ("degree_1", "degree_2"),
    key=lambda name: regression_metrics(yva_r, regression_models[name].predict(Xva))[
        "RMSE"
    ],
)
best_b_name = max(
    ("C_0.1", "C_1.0"),
    key=lambda name: binary_metrics(
        yva_b, classification_models[name].predict_proba(Xva)[:, 1]
    )["AP"],
)
best_r = regression_models[best_r_name]
best_b = classification_models[best_b_name]

prediction_va = best_r.predict(Xva)
residual_va = yva_r.to_numpy() - prediction_va
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(yva_r, prediction_va, alpha=0.7)
limits = [min(yva_r.min(), prediction_va.min()), max(yva_r.max(), prediction_va.max())]
axes[0].plot(limits, limits, "k--")
axes[0].set(
    xlabel="Referencia (u.a.)",
    ylabel="Predicción (u.a.)",
    title=f"Validación, n={len(validation)}",
)
axes[1].scatter(prediction_va, residual_va, alpha=0.7)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set(xlabel="Predicción (u.a.)", ylabel="Residuo (u.a.)")
fig.tight_layout()
fig.savefig(OUT / "regression_validation.png", dpi=160)
plt.close(fig)

prob_va = best_b.predict_proba(Xva)[:, 1]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
RocCurveDisplay.from_predictions(yva_b, prob_va, ax=axes[0])
PrecisionRecallDisplay.from_predictions(yva_b, prob_va, ax=axes[1])
fraction, mean_probability = calibration_curve(yva_b, prob_va, n_bins=5)
axes[2].plot(mean_probability, fraction, "o-")
axes[2].plot([0, 1], [0, 1], "k--")
axes[2].set(
    xlabel="Probabilidad media",
    ylabel="Fracción positiva",
    title="Calibración exploratoria",
)
fig.suptitle(f"Validación sintética, n={len(validation)}")
fig.tight_layout()
fig.savefig(OUT / "classification_validation.png", dpi=160)
plt.close(fig)

terms = best_r.named_steps["poly"].get_feature_names_out(FEATURES)
pd.DataFrame({"term": terms, "coefficient": best_r.named_steps["model"].coef_}).to_csv(
    OUT / "regression_coefficients.csv", index=False
)
beta = best_b.named_steps["model"].coef_[0]
pd.DataFrame(
    {"feature": FEATURES, "beta_standardized": beta, "odds_ratio_per_SD": np.exp(beta)}
).to_csv(OUT / "logistic_coefficients.csv", index=False)

THRESHOLD = 0.5  # Fijado como referencia didáctica antes de abrir prueba.
protocol = {
    "seed": SEED,
    "features": FEATURES,
    "continuous_model": best_r_name,
    "binary_model": best_b_name,
    "threshold": THRESHOLD,
    "regression_intercept": float(best_r.named_steps["model"].intercept_),
    "logistic_intercept": float(best_b.named_steps["model"].intercept_[0]),
    "fit_population": "train",
    "selection_population": "validation",
    "data_kind": "synthetic_not_clinical",
}
(OUT / "frozen_protocol.json").write_text(
    json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
)
# No se reajusta: se evalúa el modelo exactamente seleccionado.
test = df.loc[df["split"] == "test"].copy()
Xte = test[FEATURES]
prediction_te = best_r.predict(Xte)
prob_te = best_b.predict_proba(Xte)[:, 1]
test_rows = []
for name in (best_r_name, "baseline"):
    test_rows.append(
        {
            "route": "continuous",
            "model": name,
            **regression_metrics(
                test["target_continuous"], regression_models[name].predict(Xte)
            ),
        }
    )
for name in (best_b_name, "baseline"):
    test_rows.append(
        {
            "route": "binary",
            "model": name,
            **binary_metrics(
                test["target_binary"],
                classification_models[name].predict_proba(Xte)[:, 1],
                THRESHOLD,
            ),
        }
    )
pd.DataFrame(test_rows).to_csv(OUT / "test_metrics.csv", index=False)
result = test[
    ["sample_id", "subject_id", "split", "target_continuous", "target_binary"]
].copy()
result["prediction_continuous"] = prediction_te
result["residual"] = test["target_continuous"] - prediction_te
result["probability_positive"] = prob_te
result["decision"] = (prob_te >= THRESHOLD).astype(int)
result.to_csv(OUT / "test_predictions.csv", index=False)
print(comparison.to_string(index=False))
print("Modelos congelados:", best_r_name, best_b_name)
