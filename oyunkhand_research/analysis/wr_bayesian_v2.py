# 3A. V2: сорил тус бүрийн явцыг spline-ээр, оролдлогын ялгааг mean random effect-ээр тооцох
# V1-ийг өөрчлөхгүй. 2025 өгөгдөл сургалт болон тохируулгад орохгүй.
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
import joblib
V2_DIR = MODEL_DIR / "v2_spline"
V2_DIR.mkdir(parents=True, exist_ok=True)
idata_v1 = idata
X_mu_v1, X_kappa_v1 = X_mu.copy(), X_kappa.copy()

progress_spline = SplineTransformer(n_knots=4, degree=3, include_bias=False, extrapolation="constant")
B = progress_spline.fit_transform(model_data[["progress"]])
D = pd.get_dummies(model_data["drill_name"], dtype=float).reindex(columns=DRILLS)
interaction_blocks = [B * D[d].to_numpy()[:, None] for d in DRILLS]
mu2_names = ["intercept", "speed_z", "signed_accel_z", "previous_turn_rad"]
Xmu2 = np.column_stack([np.ones(len(model_data)), Z["speed_in_yd_s"], Z["signed_accel_yd_s2"],
    model_data["prev_turn_rad"], D.iloc[:, 1:].to_numpy(), *interaction_blocks])
mu2_names += ["drill_" + d for d in DRILLS[1:]]
mu2_names += [f"{d}_progress_basis_{j}" for d in DRILLS for j in range(B.shape[1])]
speed_sq = Z["speed_in_yd_s"].to_numpy() ** 2
speed_sq_center, speed_sq_scale = speed_sq.mean(), speed_sq.std()
speed_sq_z = (speed_sq - speed_sq_center) / speed_sq_scale
phase_dummies = pd.get_dummies(model_data["phase"], drop_first=True, dtype=float)
Xkap2 = np.column_stack([np.ones(len(model_data)), Z["speed_in_yd_s"], speed_sq_z,
    Z["signed_accel_yd_s2"], D.iloc[:, 1:].to_numpy(), phase_dummies.to_numpy()])
kap2_names = ["intercept", "speed_z", "speed_squared_z", "signed_accel_z"]
kap2_names += ["drill_" + d for d in DRILLS[1:]] + phase_dummies.columns.tolist()
# Prior биш: зөвхөн NUTS-ийн эхлэх цэгийг data-д ойр болгох ridge initialization.
initial_beta_mu = Ridge(alpha=2., fit_intercept=False).fit(Xmu2, np.tan(y_angle/2)).coef_
# Orthogonal basis: span нь ижил; prior-ыг QR coefficient дээр тогтоосон V2.
Xmu2_original = Xmu2.copy()
Q2, R2 = np.linalg.qr(Xmu2_original, mode="reduced")
qr_scale2 = np.sqrt(len(model_data))
Xmu2 = Q2 * qr_scale2
R2_scaled = R2 / qr_scale2
initial_beta_mu = R2_scaled @ initial_beta_mu
mu2_original_names = mu2_names.copy()
mu2_names = [f"qr_mean_{j}" for j in range(Xmu2.shape[1])]
np.save(V2_DIR / "mean_basis_R_scaled.npy", R2_scaled)
print("Mean design rank:", np.linalg.matrix_rank(Xmu2_original), "of", Xmu2_original.shape[1])
print("V1 dispersion hyperparameters:")
display(az.summary(idata_v1, var_names=["sigma_player", "sigma_attempt"], hdi_prob=.95))
print("V2 same training rows:", Xmu2.shape[0], "players:", len(player_ids))
print("Mean coefficients:", Xmu2.shape[1], "Concentration coefficients:", Xkap2.shape[1])
joblib.dump(progress_spline, V2_DIR / "progress_spline.joblib")


# 3B. V2 Bayesian circular mixed-effects fit
# Mean нь drill-specific smooth curve + speed/acceleration/previous turn + attempt intercept.
# Concentration нь nonlinear speed + context + player random effect.
coords2 = {"player": player_ids.astype(str), "attempt": attempt_levels.tolist(),
           "mu_feature": mu2_names, "kappa_feature": kap2_names}
with pm.Model(coords=coords2) as circular_model_v2:
    beta_mu2 = pm.Normal("beta_mu", 0., .05, dims="mu_feature")
    k_prior2 = np.zeros(Xkap2.shape[1]); k_prior2[0] = np.log(400.)
    k_sd2 = np.full(Xkap2.shape[1], .7); k_sd2[0] = 1.5
    beta_kappa2 = pm.Normal("beta_kappa", k_prior2, k_sd2, dims="kappa_feature")
    sigma_player2 = pm.HalfNormal("sigma_player", .7)
    player_z2 = pm.Normal("player_z", 0., 1., dims="player")
    player_effect2 = pm.Deterministic("player_effect", sigma_player2 * player_z2, dims="player")
    sigma_attempt_mean2 = pm.HalfNormal("sigma_attempt_mean", .03)
    attempt_z2 = pm.Normal("attempt_z", 0., 1., dims="attempt")
    attempt_mean2 = sigma_attempt_mean2 * (attempt_z2[attempt_idx] - pt.mean(attempt_z2))
    mu2 = 2 * pt.arctan(pt.dot(Xmu2, beta_mu2) + attempt_mean2)
    kappa2 = pt.exp(pt.dot(Xkap2, beta_kappa2) + player_effect2[player_idx])
    pm.CustomDist("turn", mu2, kappa2, logp=stable_vonmises_logp, observed=y_angle)
    assert np.isfinite(circular_model_v2.compile_logp()(circular_model_v2.initial_point()))
    idata_v2 = pm.sample(draws=1200, tune=1000, chains=4, cores=2, target_accept=.95,
        init="adapt_diag", initvals={"beta_mu": initial_beta_mu}, random_seed=SEED+1,
        idata_kwargs={"log_likelihood": False})
idata_v2.to_netcdf(V2_DIR / "posterior.nc")
(V2_DIR / "model_spec.json").write_text(json.dumps({"training_years":[2023,2024], "heldout_year":2025,
    "mu_features":mu2_names, "original_mu_features":mu2_original_names, "mu_QR_prior_sd":.05, "kappa_features":kap2_names, "speed_sq_center":float(speed_sq_center),
    "speed_sq_scale":float(speed_sq_scale), "standardization_centers":centers.to_dict(),
    "standardization_scales":scales.to_dict(), "version":"v2_spline_attempt_mean"},indent=2))
print("V2 posterior хадгаллаа:", V2_DIR)


# 3C. V1 ба V2-ийн ижил түүвэр дээрх оношилгооны харьцуулалт
vars2 = ["beta_mu", "beta_kappa", "sigma_player", "player_z", "player_effect", "sigma_attempt_mean", "attempt_z"]
diag2 = az.summary(idata_v2, var_names=vars2, hdi_prob=.95)
r2 = az.rhat(idata_v2, var_names=vars2)
e2 = az.ess(idata_v2, var_names=vars2, method="bulk")
max_rhat2 = max(float(v.max()) for v in r2.data_vars.values())
min_ess2 = min(float(v.min()) for v in e2.data_vars.values())
div2 = int(idata_v2.sample_stats.diverging.sum())
bfmi2 = float(np.min(az.bfmi(idata_v2)))
depth2 = int(idata_v2.sample_stats.tree_depth.max())
sampler_ok2 = bool(div2 == 0 and max_rhat2 <= 1.01 and min_ess2 >= 400 and bfmi2 > .3 and depth2 < 10)
diag2.to_csv(V2_DIR / "mcmc_diagnostics.csv")
p2 = idata_v2.posterior.stack(sample=("chain","draw"))
rng2 = np.random.default_rng(SEED+1)
sel2 = rng2.choice(p2.sizes["sample"], 600, replace=False)
bm2 = p2.beta_mu.transpose("sample","mu_feature").values[sel2]
bk2 = p2.beta_kappa.transpose("sample","kappa_feature").values[sel2]
u2 = p2.player_effect.transpose("sample","player").values[sel2]
zatt2 = p2.attempt_z.transpose("sample","attempt").values[sel2]
satt2 = p2.sigma_attempt_mean.values[sel2]
pmu2 = 2*np.arctan(bm2 @ Xmu2.T + satt2[:,None]*(zatt2[:,attempt_idx]-zatt2.mean(axis=1,keepdims=True)))
pkap2 = np.exp(bk2 @ Xkap2.T + u2[:,player_idx])
yrep2 = rng2.vonmises(pmu2, pkap2)
obs_stats = np.array([np.cos(y_angle).mean(), np.sin(y_angle).mean(), np.quantile(np.abs(y_angle),.9)])
rep_stats2 = np.column_stack([np.cos(yrep2).mean(axis=1),np.sin(yrep2).mean(axis=1),np.quantile(np.abs(yrep2),.9,axis=1)])
ppc2 = pd.DataFrame({"statistic":["mean_cos","mean_sin","abs_angle_90pct"], "observed":obs_stats,
    "predictive_low":np.quantile(rep_stats2,.025,axis=0),"predictive_high":np.quantile(rep_stats2,.975,axis=0)})
ppc2["inside_interval"] = ppc2.observed.between(ppc2.predictive_low,ppc2.predictive_high)
ppc_ok2 = bool(ppc2.inside_interval.all())
ppc2.to_csv(V2_DIR / "conditional_predictive_checks.csv",index=False)
resid2 = np.angle(np.exp(1j*(y_angle-np.angle(np.exp(1j*pmu2).mean(axis=0)))))
rt2 = model_data[SEQ+["timestamp"]].copy(); rt2["residual"]=resid2
rt2=rt2.sort_values(SEQ+["timestamp"])
rt2["lag"] = rt2.groupby(SEQ).residual.shift(1)
lagcorr2=float(rt2[["residual","lag"]].corr().iloc[0,1])
# Ижил 600 draws ашиглан V1 predictive шалгалтын Monte Carlo ялгааг багасгана.
p1=idata_v1.posterior.stack(sample=("chain","draw"))
s1=np.random.default_rng(SEED+1).choice(p1.sizes["sample"],600,replace=False)
m1=2*np.arctan(p1.beta_mu.transpose("sample","mu_feature").values[s1] @ X_mu_v1.T)
k1=np.exp(p1.beta_kappa.transpose("sample","kappa_feature").values[s1] @ X_kappa_v1.T
    + p1.player_effect.transpose("sample","player").values[s1][:,player_idx]
    + p1.sigma_attempt.values[s1,None]*p1.attempt_z.transpose("sample","attempt").values[s1][:,attempt_idx])
rep1=np.random.default_rng(SEED+1).vonmises(m1,k1)
rstat1=np.column_stack([np.cos(rep1).mean(axis=1),np.sin(rep1).mean(axis=1),np.quantile(np.abs(rep1),.9,axis=1)])
ppc1_pass=bool(np.all((obs_stats>=np.quantile(rstat1,.025,axis=0)) & (obs_stats<=np.quantile(rstat1,.975,axis=0))))
comparison=pd.DataFrame({"version":["V1","V2"],"divergences":[divergences,div2],"max_Rhat":[max_rhat,max_rhat2],
    "min_bulk_ESS":[min_ess,min_ess2],"min_BFMI":[float(bfmi.min()),bfmi2],"max_tree_depth":[max_tree_depth,depth2],
    "sampler_checks_pass":[SAMPLER_OK,sampler_ok2],"PPC_600_draws_pass":[ppc1_pass,ppc_ok2],
    "residual_lag_corr":[residual_lag_corr,lagcorr2]})
display(comparison); display(ppc2)
comparison.to_csv(V2_DIR / "v1_v2_comparison.csv",index=False)
upost=np.exp(idata_v2.posterior.player_effect.transpose("chain","draw","player").values.reshape(-1,len(player_ids)))
traits2=pd.DataFrame({"nfl_id":player_ids,"concentration_multiplier_median":np.median(upost,axis=0),
    "ci95_low":np.quantile(upost,.025,axis=0),"ci95_high":np.quantile(upost,.975,axis=0)})
traits2=traits2.merge(wr_players[["nfl_id","display_name","draft_year"]],on="nfl_id",validate="one_to_one").merge(player_counts,on="nfl_id")
traits2["sampler_checks_pass"]=sampler_ok2; traits2["predictive_checks_pass"]=ppc_ok2
traits2["result_status"]="EXPLORATORY_NOT_NFL_VALIDATED"
traits2.to_csv(V2_DIR / "player_traits_exploratory.csv",index=False)
print("<1: илүү хэлбэлзэлтэй, >1: илүү төвлөрсөн өнцөг; чадварын чансаа биш.")
figv2, axv2=plt.subplots(1,3,figsize=(15,4))
bins=np.linspace(-30,30,70)
for ax,rep,title in [(axv2[0],rep1,"V1 conditional fit"),(axv2[1],yrep2,"V2 conditional fit")]:
    ax.hist(np.degrees(y_angle),bins=bins,density=True,alpha=.4,label="Observed")
    ax.hist(np.degrees(rep).ravel(),bins=bins,density=True,histtype="step",color="orange",label="Predictive")
    ax.set(title=title,xlabel="Turn angle (degrees)"); ax.legend(fontsize=8)
show2=traits2.sort_values("display_name").head(12).iloc[::-1]
axv2[2].errorbar(show2.concentration_multiplier_median,np.arange(len(show2)),
    xerr=[show2.concentration_multiplier_median-show2.ci95_low,show2.ci95_high-show2.concentration_multiplier_median],fmt="o",capsize=2)
axv2[2].set_yticks(np.arange(len(show2)),labels=show2.display_name)
axv2[2].axvline(1,color="gray",linestyle="--")
axv2[2].set(title="V2 traits: alphabetical subset",xlabel="Concentration multiplier (95% CrI)")
plt.tight_layout(); plt.show()
figv2.savefig(V2_DIR / "v1_v2_diagnostics.png",dpi=150,bbox_inches="tight")
print("2025 болон NFL гүйцэтгэлийн өгөгдөл ашиглаагүй. Энэ бол conditional training-fit харьцуулалт.")
print("Нийлэлт сайн байсан ч predictive шалгалт болон residual холбоог тусад нь харна.")
print("Хадгалсан:",V2_DIR)
