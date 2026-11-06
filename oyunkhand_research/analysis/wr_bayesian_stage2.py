# 2B. Сорилын хэсгийг бусад тоглогчийн хөдөлгөөний загвараар ойролцоолох
# 2025 оны cohort-ыг энэ сургалтаас тусгаарлаж хадгална.
# phase нь cone/event-ийн жинхэнэ шошго биш, normalized path progress-ийн template zone.
from scipy.ndimage import gaussian_filter1d
import json
MODEL_DIR = OUTPUT_DIR / "bayesian_stage2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DRILLS = ["DAGGER_ROUTE_LEFT", "SLANT_ROUTE_LEFT", "CURL_ROUTE_RIGHT"]
N_BINS = 20
FRAME_STRIDE = 4  # Ойролцоогоор 0.4 сек; хөрш 3-цэгийн өнцгүүдийн давхцлыг багасгана.
SEGMENT_KEYS = SEQ
motion2 = wr_motion.loc[wr_motion["drill_name"].isin(DRILLS)].copy()
motion2 = motion2.sort_values(SEGMENT_KEYS + ["timestamp"]).reset_index(drop=True)
# Хугацааны тасалдал бүрд шинэ segment эхлүүлнэ.
new_segment = ~motion2["dt_prev_s"].sub(EXPECTED_DT_S).abs().le(DT_TOLERANCE_S)
motion2["segment"] = new_segment.groupby([motion2[k] for k in SEGMENT_KEYS]).cumsum().astype(int)
SEG = SEGMENT_KEYS + ["segment"]
g2 = motion2.groupby(SEG, sort=False, observed=True)
motion2["frame_index"] = g2.cumcount()
motion2["prev_turn_rad"] = g2["turn_angle_rad"].shift(1)
motion2["prev_speed_yd_s"] = g2["speed_in_yd_s"].shift(1)
motion2["signed_accel_yd_s2"] = (motion2["speed_in_yd_s"] - motion2["prev_speed_yd_s"]) / motion2["dt_prev_s"]
# Өмнөх цэгээс ирсэн алхмыг хурд * хугацаагаар сэргээнэ.
incoming_distance = motion2["speed_in_yd_s"] * motion2["dt_prev_s"]
incoming_distance = incoming_distance.where(~new_segment, 0).fillna(0)
motion2["path_distance"] = incoming_distance.groupby([motion2[k] for k in SEG]).cumsum()
path_total = motion2.groupby(SEG)["path_distance"].transform("max")
motion2["progress"] = motion2["path_distance"] / path_total.replace(0, np.nan)
motion2["progress_bin"] = np.minimum((motion2["progress"].fillna(0) * N_BINS).astype(int), N_BINS - 1)

candidate = motion2.loc[motion2["valid_turn"] & np.isfinite(motion2["prev_turn_rad"])
    & np.isfinite(motion2["signed_accel_yd_s2"]) & motion2["progress"].notna()].copy()
train_pool = candidate.loc[candidate["draft_year"].isin([2023, 2024])].copy()
if train_pool["nfl_id"].nunique() < 20:
    raise ValueError("Хөгжүүлэлтийн түүвэр хэт бага байна")
# Тоглогч бүр ижил жинтэй: олон frame/оролдлоготой хүн template-ийг эзлэхгүй.
per_player_bin = train_pool.groupby(["drill_name", "nfl_id", "progress_bin"])["turn_angle_rad"].apply(lambda s: np.abs(s).mean())
template_cache = {}
def make_template(drill, excluded_player=None):
    key = (drill, excluded_player)
    if key not in template_cache:
        tab = per_player_bin.loc[drill].unstack("progress_bin")
        if excluded_player is not None:
            tab = tab.drop(index=excluded_player, errors="ignore")
        curve = tab.mean(axis=0).reindex(range(N_BINS)).interpolate(limit_direction="both")
        if curve.isna().any():
            raise ValueError(f"{drill}: template үүсгэх боломжгүй")
        curve = gaussian_filter1d(curve.to_numpy(), sigma=1.0, mode="nearest")
        lower, upper = np.quantile(curve, [0.50, 0.75])
        zones = np.where(curve >= upper, "turn_zone", np.where(curve <= lower, "straight_zone", "transition_zone"))
        template_cache[key] = (curve, zones)
    return template_cache[key]

candidate["phase"] = ""
for (drill, player_id), idx in candidate.groupby(["drill_name", "nfl_id"]).groups.items():
    year = int(candidate.loc[idx, "draft_year"].iloc[0])
    # Сургалтын хүний phase-ийг өөрийн өнцгийг ашиглахгүйгээр тодорхойлно.
    _, zones = make_template(drill, player_id if year in [2023, 2024] else None)
    candidate.loc[idx, "phase"] = zones[candidate.loc[idx, "progress_bin"].to_numpy()]

model_data = candidate.loc[candidate["draft_year"].isin([2023, 2024]) & candidate["frame_index"].mod(FRAME_STRIDE).eq(0)].copy()
holdout_2025 = candidate.loc[candidate["draft_year"].eq(2025) & candidate["frame_index"].mod(FRAME_STRIDE).eq(0)].copy()
# Нэг тоглогчид хангалттай frame байгаа эсэхийг харуулна; бага түүврийг нууцаар эрэмбэлэхгүй.
player_counts = model_data.groupby("nfl_id").size().rename("model_frames")
print("Сургалт: 2023–2024; тусгаарласан cohort: 2025")
print("Train:", len(model_data), "frames;", model_data.nfl_id.nunique(), "players")
print("Held-out 2025:", len(holdout_2025), "frames;", holdout_2025.nfl_id.nunique(), "players")
display(model_data.groupby(["drill_name", "phase"]).agg(players=("nfl_id", "nunique"), frames=("nfl_id", "size")))
phase_template = pd.concat([pd.DataFrame({"drill_name": d, "progress": (np.arange(N_BINS)+0.5)/N_BINS,
    "mean_abs_turn_rad": make_template(d)[0], "phase": make_template(d)[1]}) for d in DRILLS], ignore_index=True)
fig2, axs2 = plt.subplots(1, len(DRILLS), figsize=(15, 4), sharey=True)
for ax, drill in zip(axs2, DRILLS):
    t = phase_template.loc[phase_template.drill_name.eq(drill)]
    ax.plot(t.progress, np.degrees(t.mean_abs_turn_rad), color="gray")
    for phase, color in [("straight_zone", "#2864a0"), ("transition_zone", "#999999"), ("turn_zone", "#d97706")]:
        s = t.loc[t.phase.eq(phase)]
        ax.scatter(s.progress, np.degrees(s.mean_abs_turn_rad), color=color, label=phase)
    ax.set(title=drill, xlabel="Fraction of path distance")
axs2[0].set_ylabel("Template mean absolute turn (degrees)")
axs2[-1].legend(fontsize=7)
plt.tight_layout(); plt.show()
fig2.savefig(MODEL_DIR / "phase_templates.png", dpi=150, bbox_inches="tight")
model_data.to_csv(MODEL_DIR / "model_data_2023_2024.csv", index=False)
holdout_2025.to_csv(MODEL_DIR / "heldout_2025_not_fitted.csv", index=False)
phase_template.to_csv(MODEL_DIR / "phase_templates.csv", index=False)
print("Анхаарах: zones нь бусад тоглогчийн template дээрх харьцангуй их/бага эргэлтийн хэсэг; баталгаатай cone шошго биш.")


# 2C. Bayesian circular mixed-effects загвар: mean болон concentration хоёрыг загварчилна.
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pymc.distributions.dist_math import i0e
print("PyMC", pm.__version__, "ArviZ", az.__version__)

# Continuous covariates-ийг зөвхөн сургалтын өгөгдлөөр стандартчилна.
CONT = ["speed_in_yd_s", "signed_accel_yd_s2", "progress"]
centers = model_data[CONT].mean()
scales = model_data[CONT].std().replace(0, 1)
Z = (model_data[CONT] - centers) / scales
# Байрлалын ганц бүлэг WR тул position variance байхгүй; player random effect байна.
CONTEXT = pd.get_dummies(model_data[["drill_name", "phase"]], drop_first=True, dtype=float)
X_mu = np.column_stack([np.ones(len(model_data)), Z.to_numpy(),
    model_data["prev_turn_rad"].to_numpy(), CONTEXT.to_numpy()])
X_kappa = np.column_stack([np.ones(len(model_data)), Z.to_numpy(), CONTEXT.to_numpy()])
mu_names = ["intercept"] + CONT + ["previous_turn_rad"] + CONTEXT.columns.tolist()
kappa_names = ["intercept"] + CONT + CONTEXT.columns.tolist()
player_ids = np.sort(model_data["nfl_id"].unique())
player_idx = pd.Categorical(model_data["nfl_id"], categories=player_ids).codes
# Оролдлогын random intercept нь нэг оролдлогын үлдэгдэл холбоог хэсэгчлэн тооцно.
attempt_labels = model_data[SEQ].astype(str).agg("|".join, axis=1)
attempt_idx, attempt_levels = pd.factorize(attempt_labels, sort=True)
y_angle = model_data["turn_angle_rad"].to_numpy()
coords = {"player": player_ids.astype(str), "attempt": attempt_levels.tolist(),
    "mu_feature": mu_names, "kappa_feature": kappa_names}
with pm.Model(coords=coords) as circular_model:
    beta_mu = pm.Normal("beta_mu", mu=0, sigma=0.3, dims="mu_feature")
    gamma_prior = np.zeros(X_kappa.shape[1]); gamma_prior[0] = np.log(400)
    gamma_scale = np.full(X_kappa.shape[1], 0.7); gamma_scale[0] = 1.5
    beta_kappa = pm.Normal("beta_kappa", mu=gamma_prior, sigma=gamma_scale, dims="kappa_feature")
    sigma_player = pm.HalfNormal("sigma_player", sigma=0.7)
    player_z = pm.Normal("player_z", 0, 1, dims="player")
    player_effect = pm.Deterministic("player_effect", sigma_player * player_z, dims="player")
    sigma_attempt = pm.HalfNormal("sigma_attempt", sigma=0.7)
    attempt_z = pm.Normal("attempt_z", 0, 1, dims="attempt")
    eta_mu = pt.dot(X_mu, beta_mu)
    mu_angle = 2 * pt.arctan(eta_mu)
    log_kappa = pt.dot(X_kappa, beta_kappa) + player_effect[player_idx] + sigma_attempt * attempt_z[attempt_idx]
    # PyMC 5.28-ийн log(I0(kappa)) overflow-оос сэргийлсэн яг ижил von Mises logp.
    # I0e(kappa) = exp(-kappa) I0(kappa), kappa > 0.
    def stable_vonmises_logp(value, mu, kappa):
        return kappa * (pt.cos(value - mu) - 1) - pt.log(2 * np.pi) - pt.log(i0e(kappa))
    pm.CustomDist("turn", mu_angle, pt.exp(log_kappa),
        logp=stable_vonmises_logp, observed=y_angle)
    initial_logp = circular_model.compile_logp()(circular_model.initial_point())
    if not np.isfinite(initial_logp):
        raise ValueError("Initial logp finite биш; sampling эхлүүлэхгүй")
    print("Finite initial logp:", float(initial_logp))
    idata = pm.sample(draws=750, tune=1000, chains=4, cores=2,
        target_accept=0.95, init="adapt_diag", random_seed=SEED,
        idata_kwargs={"log_likelihood": False})
# Нэн түрүүнд хадгална: оношилгоо хангахгүй байсан ч сургалтын үр дүн алдагдахгүй.
idata.to_netcdf(MODEL_DIR / "circular_model_posterior.nc")
model_spec = {"method": "Bayesian von Mises mean/concentration mixed-effects",
    "training_years": [2023, 2024], "heldout_year": 2025,
    "drills": DRILLS, "frame_stride": FRAME_STRIDE,
    "centers": centers.to_dict(), "scales": scales.to_dict(),
    "mu_features": mu_names, "kappa_features": kappa_names,
    "note": "Exploratory Combine trait model; not validated against NFL game performance."}
(MODEL_DIR / "model_spec.json").write_text(json.dumps(model_spec, ensure_ascii=False, indent=2))
print("Posterior хадгалсан:", MODEL_DIR)


# 2D. Нийлэлт, posterior predictive шалгалт, тоглогчийн uncertainty
# Өнцгийн хэлбэлзэл нь тоглолтын сайн гүйцэтгэл гэсэн баталгаа биш.
VARS = ["beta_mu", "beta_kappa", "sigma_player", "sigma_attempt", "player_effect", "player_z", "attempt_z"]
diagnostics = az.summary(idata, var_names=VARS, hdi_prob=0.95)
display(diagnostics)
divergences = int(idata.sample_stats["diverging"].sum())
rhat_values = az.rhat(idata, var_names=VARS)
max_rhat = max(float(v.max()) for v in rhat_values.data_vars.values())
ess_values = az.ess(idata, var_names=VARS, method="bulk")
min_ess = min(float(v.min()) for v in ess_values.data_vars.values())
bfmi = np.asarray(az.bfmi(idata))
max_tree_depth = int(idata.sample_stats["tree_depth"].max())
SAMPLER_OK = (divergences == 0 and max_rhat <= 1.01 and min_ess >= 400 and bfmi.min() > 0.3 and max_tree_depth < 10)
print({"divergences": divergences, "max_rhat": max_rhat, "min_bulk_ESS": min_ess,
       "min_BFMI": float(bfmi.min()), "max_tree_depth": max_tree_depth, "sampler_checks_pass": SAMPLER_OK})
if not SAMPLER_OK:
    print("Оношилгоо хангаагүй: тоглогчийн дүгнэлт хийхээс өмнө tune/draws/загварыг дахин шалгана.")
diagnostics.to_csv(MODEL_DIR / "mcmc_diagnostics.csv")
# Random effect-ийн exp нь ижил нөхцөл дэх concentration multiplier.
pe = idata.posterior["player_effect"].transpose("chain", "draw", "player").values.reshape(-1, len(player_ids))
multiplier = np.exp(pe)
trait_table = pd.DataFrame({"nfl_id": player_ids, "concentration_multiplier_median": np.median(multiplier, axis=0),
    "ci95_low": np.quantile(multiplier, 0.025, axis=0), "ci95_high": np.quantile(multiplier, 0.975, axis=0)})
trait_table = trait_table.merge(wr_players[["nfl_id", "display_name", "draft_year"]], on="nfl_id", validate="one_to_one")
trait_table = trait_table.merge(player_counts, on="nfl_id", validate="one_to_one")
trait_table["sampler_checks_pass"] = SAMPLER_OK
trait_table["result_status"] = "EXPLORATORY_NOT_VALIDATED"
trait_table.to_csv(MODEL_DIR / "player_turn_traits_exploratory.csv", index=False)
print(">1: илүү төвлөрсөн, <1: илүү хэлбэлзэлтэй өнцөг; сайн/муу эрэмбэ биш.")
display(trait_table.sort_values("display_name").head(12))

# 120 posterior draws-оос one-step conditional predictive шалгалт.
# Энэ нь бүх замын симуляц болон NFL гүйцэтгэлийн шалгалт биш.
post = idata.posterior.stack(sample=("chain", "draw"))
prng = np.random.default_rng(SEED)
chosen = prng.choice(post.sizes["sample"], size=min(120, post.sizes["sample"]), replace=False)
bm = post["beta_mu"].transpose("sample", "mu_feature").values[chosen]
bk = post["beta_kappa"].transpose("sample", "kappa_feature").values[chosen]
u = post["player_effect"].transpose("sample", "player").values[chosen]
sa = post["sigma_attempt"].values[chosen]
azs = post["attempt_z"].transpose("sample", "attempt").values[chosen]
pp_mu = 2 * np.arctan(bm @ X_mu.T)
pp_kappa = np.exp(bk @ X_kappa.T + u[:, player_idx] + sa[:, None] * azs[:, attempt_idx])
y_rep = prng.vonmises(pp_mu, pp_kappa)
pp_checks = pd.DataFrame({"statistic": ["mean_cos", "mean_sin", "abs_angle_90pct"],
    "observed": [np.cos(y_angle).mean(), np.sin(y_angle).mean(), np.quantile(np.abs(y_angle), 0.9)],
    "replicated_2.5pct": [np.quantile(np.cos(y_rep).mean(axis=1), .025), np.quantile(np.sin(y_rep).mean(axis=1), .025), np.quantile(np.quantile(np.abs(y_rep), .9, axis=1), .025)],
    "replicated_97.5pct": [np.quantile(np.cos(y_rep).mean(axis=1), .975), np.quantile(np.sin(y_rep).mean(axis=1), .975), np.quantile(np.quantile(np.abs(y_rep), .9, axis=1), .975)]})
print("Conditional posterior predictive checks:"); display(pp_checks)
pp_checks["inside_predictive_interval"] = pp_checks["observed"].between(pp_checks["replicated_2.5pct"], pp_checks["replicated_97.5pct"])
pp_checks.to_csv(MODEL_DIR / "posterior_predictive_checks.csv", index=False)
PPC_OK = bool(pp_checks["inside_predictive_interval"].all())
trait_table["predictive_checks_pass"] = PPC_OK
trait_table.to_csv(MODEL_DIR / "player_turn_traits_exploratory.csv", index=False)
print("Global predictive checks pass:", PPC_OK)
if not PPC_OK:
    print("Загвар тархалтын зарим шинжийг давтаж чадахгүй байна; traits-ийг баталгаатай үнэлгээ гэж ашиглахгүй.")
# Нэг оролдлогын sampled residual-ууд холбоотой үлдсэн эсэх (descriptive diagnostic).
residual = np.angle(np.exp(1j * (y_angle - np.angle(np.exp(1j * pp_mu).mean(axis=0)))))
res_table = model_data[SEQ + ["timestamp"]].copy()
res_table["residual"] = residual
res_table = res_table.sort_values(SEQ + ["timestamp"])
res_table["previous_residual"] = res_table.groupby(SEQ)["residual"].shift(1)
residual_lag_corr = float(res_table[["residual", "previous_residual"]].corr().iloc[0, 1])
print("Within-attempt sampled residual lag correlation:", residual_lag_corr)
(MODEL_DIR / "diagnostic_status.json").write_text(json.dumps({"sampler_checks_pass": bool(SAMPLER_OK),
    "predictive_checks_pass": PPC_OK, "max_rhat": max_rhat, "min_bulk_ess": min_ess,
    "divergences": divergences, "residual_lag_correlation": residual_lag_corr}, indent=2))
fig3, axs3 = plt.subplots(1, 2, figsize=(13, 5))
# Gaussian-like tight angles-г ойлгомжтой харуулахын тулд degrees ашиглав; likelihood нь circular хэвээр.
bins = np.linspace(-30, 30, 70)
axs3[0].hist(np.degrees(y_angle), bins=bins, density=True, alpha=.5, label="Observed")
axs3[0].hist(np.degrees(y_rep).ravel(), bins=bins, density=True, histtype="step", color="orange", label="Posterior predictive")
axs3[0].set(xlabel="Turn angle (degrees)", title="Conditional fit (display range +/-30 deg)"); axs3[0].legend()
# Бүх хүнийг эрэмбэлэхгүй, нэрийн цагаан толгойн эхний 15-ыг харуулна.
show = trait_table.sort_values("display_name").head(15).iloc[::-1]
axs3[1].errorbar(show.concentration_multiplier_median, np.arange(len(show)),
    xerr=[show.concentration_multiplier_median-show.ci95_low, show.ci95_high-show.concentration_multiplier_median], fmt="o", capsize=2)
axs3[1].set_yticks(np.arange(len(show)), labels=show.display_name)
axs3[1].axvline(1, color="gray", linestyle="--")
axs3[1].set(xlabel="Concentration multiplier (95% credible interval)", title="Exploratory traits; alphabetical subset")
plt.tight_layout(); plt.show()
fig3.savefig(MODEL_DIR / "bayesian_diagnostics_traits.png", dpi=150, bbox_inches="tight")
print("2025 cohort-ыг сургалт, template байгуулахад ашиглаагүй. Held-out үнэлгээг дараагийн шатанд хийнэ.")
print("Convergence хангах нь загвар зөв эсвэл NFL-д ашигтайг дангаараа батлахгүй.")
print("Хадгалсан:", MODEL_DIR)
