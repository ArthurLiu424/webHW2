"""
HW2 – Multiple Linear Regression on Kaggle Flood Prediction Dataset (CRISP-DM)

Dataset : https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset
          (train.csv = Kaggle Playground S4E5 "Regression with a Flood Prediction Dataset",
           https://www.kaggle.com/competitions/playground-series-s4e5 ; we use a 50k random sample)
Target  : FloodProbability (continuous, 0~1)
Features: 20 numeric risk factors (integer scores)

Only numpy / scipy / matplotlib are required, so the script runs anywhere.
All OLS statistics (coefficients, SE, t, p, CI, PI, VIF, diagnostics) are
computed from the closed-form solution  beta = (X'X)^-1 X'y.

Run:  python analysis.py
Out:  figures/*.png, outputs/results.json, outputs/model.json, docs/data.js
"""
import csv
import json
import os

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "flood_s4e5_sample50k.csv")   # 50k random rows of train.csv
ORIG = os.path.join(ROOT, "data", "flood.csv")                   # original 50k-row flood.csv
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "outputs")
DOCS = os.path.join(ROOT, "docs")
for d in (FIG, OUT, DOCS):
    os.makedirs(d, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2a6fdb", "#e8833a", "#8a8f98"


# ---------------------------------------------------------------- helpers
def load_csv(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        arr = np.array([[float(v) for v in row] for row in r])
    return header, arr


def add_const(X):
    return np.column_stack([np.ones(len(X)), X])


class OLS:
    """Ordinary least squares with classical inference."""

    def fit(self, X, y):
        Xc = add_const(X)
        self.n, self.p = Xc.shape
        self.XtX_inv = np.linalg.inv(Xc.T @ Xc)
        self.beta = self.XtX_inv @ Xc.T @ y
        resid = y - Xc @ self.beta
        self.df = self.n - self.p
        self.sigma2 = resid @ resid / self.df
        self.se = np.sqrt(np.diag(self.XtX_inv) * self.sigma2)
        self.t = self.beta / self.se
        self.pval = 2 * stats.t.sf(np.abs(self.t), self.df)
        self.tcrit = stats.t.ppf(0.975, self.df)
        self.resid = resid
        self.fitted = Xc @ self.beta
        sst = ((y - y.mean()) ** 2).sum()
        self.r2 = 1 - (resid @ resid) / sst
        self.adj_r2 = 1 - (1 - self.r2) * (self.n - 1) / self.df
        k = self.p - 1
        self.F = (self.r2 / k) / max((1 - self.r2) / self.df, 1e-300)
        self.F_p = stats.f.sf(self.F, k, self.df)
        ll = -self.n / 2 * (np.log(2 * np.pi * (resid @ resid) / self.n) + 1)
        self.aic = -2 * ll + 2 * self.p
        self.bic = -2 * ll + np.log(self.n) * self.p
        return self

    def predict(self, X, interval=False):
        Xc = add_const(X)
        yhat = Xc @ self.beta
        if not interval:
            return yhat
        h = np.einsum("ij,jk,ik->i", Xc, self.XtX_inv, Xc)  # x0'(X'X)^-1 x0
        se_mean = np.sqrt(self.sigma2 * h)
        se_pred = np.sqrt(self.sigma2 * (1 + h))
        return (yhat,
                yhat - self.tcrit * se_mean, yhat + self.tcrit * se_mean,
                yhat - self.tcrit * se_pred, yhat + self.tcrit * se_pred)


def metrics(y, yhat, k):
    e = y - yhat
    mse = float(np.mean(e ** 2))
    r2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    n = len(y)
    return {"MAE": float(np.mean(np.abs(e))), "MSE": mse, "RMSE": mse ** .5,
            "R2": float(r2), "AdjR2": float(1 - (1 - r2) * (n - 1) / (n - k - 1))}


def kfold_rmse(X, y, cols, k=5):
    idx = rng.permutation(len(y)) if not hasattr(kfold_rmse, "idx") else kfold_rmse.idx
    kfold_rmse.idx = idx
    folds = np.array_split(idx, k)
    scores = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        m = OLS().fit(X[tr][:, cols], y[tr])
        scores.append(np.sqrt(np.mean((y[te] - m.predict(X[te][:, cols])) ** 2)))
    return float(np.mean(scores))


def lasso_cd(X, y, lam, iters=300):
    """Coordinate-descent Lasso on standardized X, centred y."""
    n, p = X.shape
    w = np.zeros(p)
    r = y.copy()
    col_sq = (X ** 2).sum(0) / n
    for _ in range(iters):
        w_old = w.copy()
        for j in range(p):
            r += X[:, j] * w[j]
            rho = X[:, j] @ r / n
            w[j] = np.sign(rho) * max(abs(rho) - lam, 0) / col_sq[j]
            r -= X[:, j] * w[j]
        if np.max(np.abs(w - w_old)) < 1e-9:
            break
    return w


def vif(X):
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        r2 = OLS().fit(others, X[:, j]).r2
        out.append(float(1 / (1 - r2)))
    return out


# ================================================================ 1. Data Understanding
header, raw = load_csv(DATA)
features, target = header[:-1], header[-1]
X_all, y_all = raw[:, :-1], raw[:, -1]
n, p = X_all.shape
print(f"Loaded {n} rows, {p} features, target={target}")

# Why not the original flood.csv?  It is (almost) noise-free: OLS gives R2 ~ 1.0,
# so CI / PI would collapse to zero width.  Record that finding for the report.
_, orig = load_csv(ORIG)
orig_r2 = float(OLS().fit(orig[:, :-1], orig[:, -1]).r2)
print(f"Original flood.csv OLS R2 = {orig_r2:.6f}  -> use noisier S4E5 train sample instead")

missing = int(np.isnan(raw).sum())
dup = n - len(np.unique(raw, axis=0))
desc = {f: {"mean": float(X_all[:, i].mean()), "std": float(X_all[:, i].std()),
            "min": float(X_all[:, i].min()), "max": float(X_all[:, i].max())}
        for i, f in enumerate(features)}
desc[target] = {"mean": float(y_all.mean()), "std": float(y_all.std()),
                "min": float(y_all.min()), "max": float(y_all.max())}
corr_full = np.corrcoef(raw.T)
corr_target = {f: float(corr_full[i, -1]) for i, f in enumerate(features)}
max_offdiag = float(np.max(np.abs(corr_full[:-1, :-1] - np.eye(p))))

# target histogram
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist(y_all, bins=60, color=BLUE, alpha=.85)
ax.set(title="Distribution of FloodProbability", xlabel=target, ylabel="count")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "01_target_hist.png")); plt.close(fig)

# correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(corr_full, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(p + 1)); ax.set_xticklabels(header, rotation=90, fontsize=7)
ax.set_yticks(range(p + 1)); ax.set_yticklabels(header, fontsize=7)
ax.grid(False)
fig.colorbar(im, fraction=.04)
ax.set_title("Pearson correlation matrix")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "02_corr_heatmap.png")); plt.close(fig)

# correlation with target
order = np.argsort([corr_target[f] for f in features])
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh([features[i] for i in order], [corr_target[features[i]] for i in order], color=BLUE)
ax.set(title="Correlation of each feature with FloodProbability", xlabel="Pearson r")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "03_corr_target.png")); plt.close(fig)

# ================================================================ 2. Data Preparation
perm = rng.permutation(n)
n_test = int(n * 0.2)
te_idx, tr_idx = perm[:n_test], perm[n_test:]
X_tr, y_tr, X_te, y_te = X_all[tr_idx], y_all[tr_idx], X_all[te_idx], y_all[te_idx]
mu, sd = X_tr.mean(0), X_tr.std(0)            # fitted on train only
Z_tr, Z_te = (X_tr - mu) / sd, (X_te - mu) / sd
print(f"Train {len(y_tr)}  Test {len(y_te)}  missing={missing} duplicates={dup}")

# ================================================================ 3. Feature Selection (train only)
# (a) Filter: univariate F-test  (== SelectKBest(f_regression))
F_scores = []
for j in range(p):
    r = np.corrcoef(X_tr[:, j], y_tr)[0, 1]
    F_scores.append(float(r ** 2 / (1 - r ** 2) * (len(y_tr) - 2)))
filter_rank = list(np.argsort(F_scores)[::-1])

# (b) Wrapper: forward sequential selection by 5-fold CV RMSE
selected, remaining, forward_path = [], list(range(p)), []
while remaining:
    best = min(remaining, key=lambda j: kfold_rmse(X_tr, y_tr, selected + [j]))
    selected.append(best); remaining.remove(best)
    m = OLS().fit(X_tr[:, selected], y_tr)
    forward_path.append({"k": len(selected), "added": features[best],
                         "cv_rmse": kfold_rmse(X_tr, y_tr, selected),
                         "adj_r2": float(m.adj_r2), "aic": float(m.aic), "bic": float(m.bic)})
forward_order = selected[:]

# (c) Embedded: Lasso path on standardized features
yc = y_tr - y_tr.mean()
lam_max = np.max(np.abs(Z_tr.T @ yc)) / len(yc)
lams = lam_max * np.logspace(0, -3, 40)
lasso_path = [lasso_cd(Z_tr, yc, l) for l in lams]
lasso_entry = {}
for l, w in zip(lams, lasso_path):
    for j in np.nonzero(w)[0]:
        lasso_entry.setdefault(int(j), float(l))
lasso_rank = sorted(range(p), key=lambda j: -lasso_entry.get(j, 0))

# (d) Backward elimination by p-value (alpha=0.05)
back = list(range(p))
while True:
    m = OLS().fit(X_tr[:, back], y_tr)
    worst = int(np.argmax(m.pval[1:]))
    if m.pval[1:][worst] > 0.05 and len(back) > 1:
        back.pop(worst)
    else:
        break
backward_kept = [features[j] for j in back]

# Choose k: smallest k whose CV RMSE is within 0.5% of the best -> "parsimony"
cv = np.array([s["cv_rmse"] for s in forward_path])
best_k = int(np.argmin(cv)) + 1
tol = cv.min() * 1.005
k_parsimonious = int(np.argmax(cv <= tol)) + 1

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, p + 1), cv, "o-", color=BLUE, label="5-fold CV RMSE")
ax.axvline(best_k, color=ORANGE, ls="--", label=f"best k={best_k}")
ax.axhline(tol, color=GREY, ls=":", label="best +0.5%")
ax.set(title="Forward selection: CV RMSE vs. number of features", xlabel="k features", ylabel="RMSE")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, "04_forward_selection.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
W = np.array(lasso_path)
for j in range(p):
    ax.plot(lams, W[:, j], lw=1.2)
ax.set_xscale("log"); ax.invert_xaxis()
ax.set(title="Lasso coefficient path (standardized)", xlabel="lambda (log, decreasing)", ylabel="coefficient")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "05_lasso_path.png")); plt.close(fig)

# ================================================================ 4. Modeling
candidates = {}
for k in sorted({5, 10, 15, k_parsimonious, p}):
    cols = forward_order[:k]
    m = OLS().fit(X_tr[:, cols], y_tr)
    candidates[f"Top-{k} (forward)"] = {"k": k, "cols": cols, "model": m}
# simple linear regression baseline using single best feature
cols1 = forward_order[:1]
candidates["Simple LR (1 feature)"] = {"k": 1, "cols": cols1, "model": OLS().fit(X_tr[:, cols1], y_tr)}

comparison = []
for name, c in candidates.items():
    m = c["model"]
    tr_m = metrics(y_tr, m.predict(X_tr[:, c["cols"]]), c["k"])
    te_m = metrics(y_te, m.predict(X_te[:, c["cols"]]), c["k"])
    comparison.append({"name": name, "k": c["k"], "cv_rmse": kfold_rmse(X_tr, y_tr, c["cols"]),
                       "train": tr_m, "test": te_m, "aic": float(m.aic), "bic": float(m.bic)})
# engineered baseline: simple LR on the sum of all 20 risk scores
S_tr, S_te = X_tr.sum(1, keepdims=True), X_te.sum(1, keepdims=True)
m_sum = OLS().fit(S_tr, y_tr)
comparison.append({"name": "Simple LR (FeatureSum)", "k": 1, "cv_rmse": None,
                   "train": metrics(y_tr, m_sum.predict(S_tr), 1), "test": metrics(y_te, m_sum.predict(S_te), 1),
                   "aic": float(m_sum.aic), "bic": float(m_sum.bic)})
comparison.sort(key=lambda r: r["k"])

# Final model: forward-selected features at the parsimonious k
final_k = k_parsimonious
final_cols = forward_order[:final_k]
final = OLS().fit(X_tr[:, final_cols], y_tr)
final_feats = [features[j] for j in final_cols]
yhat, ci_lo, ci_hi, pi_lo, pi_hi = final.predict(X_te[:, final_cols], interval=True)
test_metrics = metrics(y_te, yhat, final_k)
pi_coverage = float(np.mean((y_te >= pi_lo) & (y_te <= pi_hi)))
ci_coverage = float(np.mean((y_te >= ci_lo) & (y_te <= ci_hi)))

# standardized coefficients (for comparability)
final_std = OLS().fit(Z_tr[:, final_cols], y_tr)

# ================================================================ 5. Evaluation & diagnostics
resid = final.resid
dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))
# Breusch–Pagan (LM = n*R2 of e^2 on X)
bp_r2 = OLS().fit(X_tr[:, final_cols], resid ** 2).r2
bp_lm = float(len(resid) * bp_r2)
bp_p = float(stats.chi2.sf(bp_lm, final_k))
# normality (Jarque–Bera) + skew/kurtosis
jb = stats.jarque_bera(resid)
vifs = vif(X_tr[:, final_cols])

# --- Prediction plot 1: actual vs predicted (sorted) with 95% CI & PI (subsample 300 for legibility)
sub = rng.choice(len(y_te), 300, replace=False)
o = sub[np.argsort(yhat[sub])]
fig, ax = plt.subplots(figsize=(10, 4.8))
xs = np.arange(len(o))
ax.fill_between(xs, pi_lo[o], pi_hi[o], color=BLUE, alpha=.15, label="95% prediction interval")
ax.fill_between(xs, ci_lo[o], ci_hi[o], color=BLUE, alpha=.55, label="95% confidence interval")
ax.plot(xs, yhat[o], color=BLUE, lw=1.5, label="predicted")
ax.scatter(xs, y_te[o], s=9, color=ORANGE, zorder=3, label="actual")
ax.set(title=f"Test-set predictions (300 random samples, sorted)  PI coverage={pi_coverage:.1%}",
       xlabel="sample (sorted by prediction)", ylabel=target)
ax.legend(loc="upper left", fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(FIG, "06_prediction_intervals.png")); plt.close(fig)

# --- Prediction plot 2: actual vs predicted scatter
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(yhat, y_te, s=3, alpha=.25, color=BLUE)
lims = [min(yhat.min(), y_te.min()), max(yhat.max(), y_te.max())]
ax.plot(lims, lims, color=ORANGE, lw=1.5)
ax.set(title=f"Actual vs Predicted (test R²={test_metrics['R2']:.3f})", xlabel="predicted", ylabel="actual")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "07_actual_vs_pred.png")); plt.close(fig)

# --- Prediction plot 3: effect of top feature, others at train mean, with CI & PI
top = final_cols[0]
grid = np.linspace(X_all[:, top].min(), X_all[:, top].max(), 50)
Xg = np.tile(X_tr[:, final_cols].mean(0), (len(grid), 1))
Xg[:, 0] = grid
g_hat, g_cl, g_ch, g_pl, g_ph = final.predict(Xg, interval=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.fill_between(grid, g_pl, g_ph, color=BLUE, alpha=.15, label="95% PI")
ax.fill_between(grid, g_cl, g_ch, color=BLUE, alpha=.5, label="95% CI")
ax.plot(grid, g_hat, color=BLUE, label="prediction")
jit = X_te[:, top] + rng.uniform(-.3, .3, len(y_te))
ax.scatter(jit[:1500], y_te[:1500], s=3, alpha=.25, color=ORANGE, label="test data")
ax.set(title=f"Partial effect of {features[top]} (others at mean)", xlabel=features[top], ylabel=target)
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(FIG, "08_partial_effect_CI_PI.png")); plt.close(fig)

# --- Residual diagnostics
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
axs[0].scatter(final.fitted[:4000], resid[:4000], s=3, alpha=.3, color=BLUE)
axs[0].axhline(0, color=ORANGE); axs[0].set(title="Residuals vs Fitted", xlabel="fitted", ylabel="residual")
axs[1].hist(resid, bins=60, color=BLUE, alpha=.85); axs[1].set(title="Residual histogram")
(osm, osr), (sl, ic, _) = stats.probplot(resid, dist="norm")
axs[2].scatter(osm[::20], osr[::20], s=3, color=BLUE)
axs[2].plot(osm, sl * osm + ic, color=ORANGE); axs[2].set(title="Normal Q-Q", xlabel="theoretical", ylabel="sample")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "09_residual_diagnostics.png")); plt.close(fig)

# --- Coefficient plot with 95% CI (standardized)
b, s_ = final_std.beta[1:], final_std.se[1:]
o2 = np.argsort(b)
fig, ax = plt.subplots(figsize=(7, 6))
ax.errorbar(b[o2], range(final_k), xerr=final_std.tcrit * s_[o2], fmt="o", color=BLUE, ecolor=GREY, capsize=2)
ax.set_yticks(range(final_k)); ax.set_yticklabels([final_feats[i] for i in o2], fontsize=8)
ax.set(title="Standardized coefficients ± 95% CI", xlabel="Δ FloodProbability per 1 SD")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "10_coefficients.png")); plt.close(fig)

# ================================================================ 6. Deployment artefacts
Xc_cols = ["const"] + final_feats
model_json = {
    "features": final_feats, "beta": final.beta.tolist(), "XtX_inv": final.XtX_inv.tolist(),
    "sigma2": float(final.sigma2), "tcrit": float(final.tcrit), "df": int(final.df),
    "ranges": {f: [desc[f]["min"], desc[f]["max"], round(desc[f]["mean"], 2)] for f in final_feats},
}
coef_table = [{"term": t, "coef": float(final.beta[i]), "se": float(final.se[i]),
               "t": float(final.t[i]), "p": float(final.pval[i]),
               "ci_low": float(final.beta[i] - final.tcrit * final.se[i]),
               "ci_high": float(final.beta[i] + final.tcrit * final.se[i]),
               "std_coef": float(final_std.beta[i]) if i else None,
               "vif": vifs[i - 1] if i else None} for i, t in enumerate(Xc_cols)]

results = {
    "dataset": {"name": "Flood Prediction Dataset", "url": "https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset",
                "source_file": "train.csv (Playground S4E5) – 50k random sample", "orig_flood_csv_r2": orig_r2,
                "rows": n, "n_features": p, "target": target, "missing": missing, "duplicates": dup,
                "train": int(len(y_tr)), "test": int(len(y_te))},
    "describe": desc, "corr_target": corr_target, "max_abs_feature_corr": max_offdiag,
    "corr_matrix": {"labels": header, "values": np.round(corr_full, 3).tolist()},
    "target_hist": dict(zip(["counts", "edges"], [a.tolist() for a in np.histogram(y_all, bins=50)])),
    "feature_selection": {
        "filter_F": {features[j]: F_scores[j] for j in filter_rank},
        "forward_path": forward_path, "best_k": best_k, "k_parsimonious": k_parsimonious,
        "lasso_rank": [features[j] for j in lasso_rank],
        "lasso_path": {"lambdas": lams.tolist(), "coefs": {features[j]: W[:, j].tolist() for j in range(p)}},
        "backward_kept": backward_kept,
    },
    "comparison": comparison,
    "final": {"k": final_k, "features": final_feats, "test": test_metrics,
              "r2_train": float(final.r2), "adj_r2_train": float(final.adj_r2),
              "F": float(final.F), "F_p": float(final.F_p), "aic": float(final.aic), "bic": float(final.bic),
              "pi_coverage": pi_coverage, "ci_coverage": ci_coverage, "coef_table": coef_table},
    "diagnostics": {"durbin_watson": dw, "bp_lm": bp_lm, "bp_p": bp_p,
                    "jb_stat": float(jb.statistic), "jb_p": float(jb.pvalue),
                    "skew": float(stats.skew(resid)), "kurtosis": float(stats.kurtosis(resid)),
                    "max_vif": float(max(vifs))},
    "plots": {
        "pred_sorted": {"yhat": yhat[o].tolist(), "y": y_te[o].tolist(), "ci_lo": ci_lo[o].tolist(),
                        "ci_hi": ci_hi[o].tolist(), "pi_lo": pi_lo[o].tolist(), "pi_hi": pi_hi[o].tolist()},
        "scatter": {"yhat": yhat[:2500].round(4).tolist(), "y": y_te[:2500].tolist()},
        "partial": {"feature": features[top], "x": grid.tolist(), "yhat": g_hat.tolist(),
                    "ci_lo": g_cl.tolist(), "ci_hi": g_ch.tolist(), "pi_lo": g_pl.tolist(), "pi_hi": g_ph.tolist()},
        "resid": {"fitted": final.fitted[:2500].round(4).tolist(), "resid": resid[:2500].round(5).tolist(),
                  "qq_x": osm[::100].tolist(), "qq_y": osr[::100].tolist(), "qq_line": [float(sl), float(ic)]},
    },
}
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=1)
with open(os.path.join(OUT, "model.json"), "w") as f:
    json.dump(model_json, f, indent=1)
with open(os.path.join(DOCS, "data.js"), "w") as f:
    f.write("window.RESULTS = " + json.dumps(results) + ";\nwindow.MODEL = " + json.dumps(model_json) + ";\n")

print("\n=== Forward selection path ===")
for s in forward_path:
    print(f"k={s['k']:2d} +{s['added']:<32} CV-RMSE={s['cv_rmse']:.5f} adjR2={s['adj_r2']:.4f}")
print(f"best_k={best_k}  parsimonious k={k_parsimonious}  backward kept={len(backward_kept)}")
print("\n=== Model comparison (test) ===")
for c in comparison:
    print(f"{c['name']:<24} k={c['k']:2d} RMSE={c['test']['RMSE']:.5f} R2={c['test']['R2']:.4f}")
print("\nFinal test:", {k: round(v, 5) for k, v in test_metrics.items()})
print(f"PI coverage={pi_coverage:.3f}  DW={dw:.3f}  BP p={bp_p:.3g}  JB p={jb.pvalue:.3g}  maxVIF={max(vifs):.3f}")
