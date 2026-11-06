# 5A. V3-ийг сонгохоос өмнөх шалтгаан оношилгоо. Зөвхөн 2023-2024.
V3_DIR = MODEL_DIR / 'v3_dynamic_hurdle'
V3_DIR.mkdir(parents=True, exist_ok=True)
audit3 = model_data.copy()
audit3['exact_zero'] = audit3.turn_angle_rad.eq(0)
audit3['near_zero_05deg'] = audit3.turn_angle_rad.abs().le(np.radians(.5))
audit3['obs_sin'] = np.sin(y_angle)
audit3['v2_pred_sin'] = np.sin(pmu2).mean(axis=0)
audit3['v2_residual'] = resid2
summary3 = audit3.groupby('drill_name').agg(frames=('nfl_id','size'),zero_rate=('exact_zero','mean'),near_zero_rate=('near_zero_05deg','mean'),obs_mean_sin=('obs_sin','mean'),pred_mean_sin=('v2_pred_sin','mean'))
display(summary3)
print('Zero rows:',int(audit3.exact_zero.sum()),'of',len(audit3))
print('Previous raw angle correlation:',np.corrcoef(y_angle,model_data.prev_turn_rad)[0,1])
print('Unique values near zero:',np.unique(np.round(np.degrees(y_angle[np.abs(y_angle)<.005]),8))[:25])
# Lag at four original frames; do not cross a time gap.
md3=motion2.sort_values(SEG+['timestamp']).copy()
g3=md3.groupby(SEG,sort=False)
md3['lag4_turn_rad']=g3.turn_angle_rad.shift(4)
md3['lag4_time']=g3.timestamp.shift(4)
md3['lag4_ok']=(md3.timestamp-md3.lag4_time).dt.total_seconds().sub(.4).abs().le(.025)
lag3=md3.set_index(KEY)[['lag4_turn_rad','lag4_ok']]
audit3=audit3.join(lag3,on=KEY,validate='one_to_one')
print('Lag4 availability:',audit3.lag4_ok.fillna(False).mean())
print('Lag4 raw correlation:',audit3[['turn_angle_rad','lag4_turn_rad']].corr().iloc[0,1])
summary3.to_csv(V3_DIR/'pre_model_audit.csv')
# Preserve coordinate lookup for follow-up measurement checks.
geom3=wr_motion.set_index(KEY)[['x','y']]
print('2025 not inspected for model choice.')


# 5B. Frozen V3 specification + 2023 -> 2024 development validation.
# No 2025 rows or 2024 outcomes enter fitted preprocessing.
from scipy.optimize import minimize
from scipy.special import expit, logit, i0e as sp_i0e, i1e as sp_i1e
from scipy.stats import vonmises
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
import joblib
V3_CONFIG = dict(knots=4, degree=3, mean_prior_sd=.10, zero_prior_rate=.08,
                 validation_train_year=2023, validation_year=2024, bin_width_deg=1.,
                 seed=2030, draws=1000, tune=1000, chains=4, target_accept=.95)
(V3_DIR/'frozen_spec.json').write_text(json.dumps(V3_CONFIG,indent=2))
# Same 2790 rows as V1/V2. Every retained lag must be within its segment.
v3_data = audit3.copy().reset_index(drop=True)
assert v3_data.lag4_ok.all()
v3_data['lag4_missing']=v3_data.lag4_turn_rad.isna().astype(float)
print('Missing valid lag4 angles:',int(v3_data.lag4_missing.sum()),'; zero imputation + explicit missing indicator')
v3_data['lag4_turn_rad']=v3_data.lag4_turn_rad.fillna(0.)
assert not v3_data.draft_year.eq(2025).any()

def prepare_v3_design(fit_df, apply_df, dynamic):
    cols=['speed_in_yd_s','signed_accel_yd_s2']
    c=fit_df[cols].mean(); s=fit_df[cols].std().replace(0,1)
    spline=SplineTransformer(n_knots=4,degree=3,include_bias=False,extrapolation='constant')
    spline.fit(fit_df[['progress']])
    sq_fit=((fit_df.speed_in_yd_s-c.iloc[0])/s.iloc[0])**2
    sq_c,sq_s=float(sq_fit.mean()),float(sq_fit.std())
    lag_abs_c=float(fit_df.lag4_turn_rad.abs().mean()); lag_abs_s=float(fit_df.lag4_turn_rad.abs().std())
    def raw(d):
        z=(d[cols]-c)/s
        dummies=np.column_stack([d.drill_name.eq(k).astype(float) for k in DRILLS])
        basis=spline.transform(d[['progress']])
        mean=np.column_stack([np.ones(len(d)),z.to_numpy(),d.prev_turn_rad,dummies[:,1:],
                               *[basis*dummies[:,j,None] for j in range(len(DRILLS))]])
        if dynamic: mean=np.column_stack([mean,d.lag4_turn_rad,d.lag4_missing])
        kap=np.column_stack([np.ones(len(d)),z.speed_in_yd_s,
                            (z.speed_in_yd_s**2-sq_c)/sq_s,z.signed_accel_yd_s2,dummies[:,1:]])
        if dynamic:
            kap=np.column_stack([kap,(d.lag4_turn_rad.abs()-lag_abs_c)/lag_abs_s])
        zero=np.column_stack([np.ones(len(d)),z.speed_in_yd_s,d.progress-.5,dummies[:,1:],
                              (d.lag4_turn_rad.eq(0)&d.lag4_missing.eq(0)).astype(float),d.lag4_missing])
        return mean.astype(float),kap.astype(float),zero.astype(float)
    mf,kf,zf=raw(fit_df); ma,ka,za=raw(apply_df)
    Q,R=np.linalg.qr(mf,mode='reduced'); Rscaled=R/np.sqrt(len(fit_df))
    assert np.linalg.matrix_rank(mf)==mf.shape[1]
    return (Q*np.sqrt(len(fit_df)),kf,zf),(np.linalg.solve(Rscaled.T,ma.T).T,ka,za),dict(
        spline=spline,centers=c,scales=s,speed_sq_center=sq_c,speed_sq_scale=sq_s,
        lag_abs_center=lag_abs_c,lag_abs_scale=lag_abs_s,R_scaled=Rscaled,dynamic=dynamic)

def fit_population_map(d,design,hurdle):
    X,K,H=design; y=d.turn_angle_rad.to_numpy(); iz=y==0; keep=~iz if hurdle else np.ones(len(y),bool)
    a,b,c=X.shape[1],K.shape[1],H.shape[1]
    kmean=np.zeros(b); kmean[0]=np.log(400.)
    ksd=np.full(b,.7); ksd[0]=1.5
    amean=np.zeros(c); amean[0]=logit(.08)
    asd=np.ones(c); asd[0]=1.5
    ini=Ridge(alpha=2,fit_intercept=False).fit(X[keep],np.tan(y[keep]/2)).coef_
    x0=np.r_[ini,kmean,amean] if hurdle else np.r_[ini,kmean]
    def fun(v):
        beta=v[:a]; gamma=v[a:a+b]
        eta=X@beta; mu=2*np.arctan(eta); kap=np.exp(K@gamma); delta=y-mu
        ll=kap*(np.cos(delta)-1)-np.log(2*np.pi)-np.log(sp_i0e(kap))
        loss=-ll[keep].sum()+.5*np.sum((beta/.10)**2)+.5*np.sum(((gamma-kmean)/ksd)**2)
        gm=kap*np.sin(delta)*2/(1+eta**2); gm[~keep]=0
        gk=kap*(np.cos(delta)-sp_i1e(kap)/sp_i0e(kap));gk[~keep]=0
        grad=np.r_[-X.T@gm+beta/.10**2,-K.T@gk+(gamma-kmean)/ksd**2]
        if hurdle:
            alpha=v[a+b:];eta0=H@alpha;p=expit(eta0)
            loss+=np.sum(np.logaddexp(0,eta0)-iz*eta0)+.5*np.sum(((alpha-amean)/asd)**2)
            grad=np.r_[grad,H.T@(p-iz)+(alpha-amean)/asd**2]
        return float(loss),grad
    bounds=[(None,None)]*a+[(-2,12)]+[(-5,5)]*(b-1)+([(None,None)]*c if hurdle else [])
    par_scale=np.r_[np.full(a,.10),ksd,asd] if hurdle else np.r_[np.full(a,.10),ksd]
    def standardized_fun(w):
        value,gradient=fun(w*par_scale)
        return value,gradient*par_scale
    standardized_bounds=[(None if low is None else low/scale,None if high is None else high/scale) for (low,high),scale in zip(bounds,par_scale)]
    result=minimize(standardized_fun,x0/par_scale,jac=True,method='L-BFGS-B',bounds=standardized_bounds,
                    options={'maxiter':5000,'ftol':1e-9,'gtol':1e-5})
    result.x=result.x*par_scale
    if not result.success: raise RuntimeError(result.message)
    return result

def population_predict(v,design,hurdle):
    X,K,H=design;a=X.shape[1];b=K.shape[1]
    return 2*np.arctan(X@v[:a]),np.exp(K@v[a:a+b]),expit(H@v[a+b:]) if hurdle else np.zeros(len(X))

def common_bin_score(d,mu,kap,pzero):
    # Same observation space for continuous baseline and point-mass hurdle model.
    y=d.turn_angle_rad.to_numpy();w=np.radians(1.)
    centers_bin=np.floor(y/w+.5)*w;lo=centers_bin-w/2;hi=centers_bin+w/2
    continuous=vonmises.cdf(hi,kap,loc=mu)-vonmises.cdf(lo,kap,loc=mu)
    prob=(1-pzero)*np.maximum(continuous,0)+pzero*((lo<=0)&(hi>0))
    return -np.log(np.maximum(prob,1e-15))

fit_dev=v3_data.loc[v3_data.draft_year.eq(2023)].copy()
val_dev=v3_data.loc[v3_data.draft_year.eq(2024)].copy()
print('Development split:',len(fit_dev),'train frames /',len(val_dev),'validation frames;',val_dev.nfl_id.nunique(),'new players')
assert set(fit_dev.nfl_id).isdisjoint(val_dev.nfl_id)
dev_fits={};dev_scores=[];dev_by_player={}
for label,dyn in [('population_spline_baseline',False),('V3_dynamic_hurdle',True)]:
    df,da,trans=prepare_v3_design(fit_dev,val_dev,dyn)
    result=fit_population_map(fit_dev,df,dyn)
    mu,k,pz=population_predict(result.x,da,dyn)
    nll=common_bin_score(val_dev,mu,k,pz)
    predicted_angle=np.angle(pz+(1-pz)*np.exp(1j*mu)*(sp_i1e(k)/sp_i0e(k)))
    errors=np.angle(np.exp(1j*(val_dev.turn_angle_rad.to_numpy()-predicted_angle)))
    per=pd.DataFrame({'nfl_id':val_dev.nfl_id.to_numpy(),'bin_nll':nll}).groupby('nfl_id').mean()
    dev_by_player[label]=per.bin_nll
    dev_fits[label]=dict(optimizer=result,transform=trans,mu=mu,kappa=k,pzero=pz,nll=nll)
    dev_scores.append(dict(model=label,frame_mean_bin_NLL=float(nll.mean()),player_equal_bin_NLL=float(per.bin_nll.mean()),
                           angular_MAE_deg=float(np.degrees(np.abs(errors)).mean()),
                           zero_Brier=float(np.mean((pz-val_dev.turn_angle_rad.eq(0).to_numpy())**2))))
dev_results=pd.DataFrame(dev_scores);display(dev_results)
delta=dev_by_player['V3_dynamic_hurdle']-dev_by_player['population_spline_baseline']
boot_rng=np.random.default_rng(2030)
bootstrap=np.array([boot_rng.choice(delta.to_numpy(),len(delta),replace=True).mean() for _ in range(2000)])
dev_comparison=dict(player_equal_delta_V3_minus_baseline=float(delta.mean()),
                    bootstrap_ci95_low=float(np.quantile(bootstrap,.025)),bootstrap_ci95_high=float(np.quantile(bootstrap,.975)),
                    split='2023->2024 development, previously used for V1/V2 training; not external test',
                    comparison='regularized population MAP models; not a refit of hierarchical V2')
print(dev_comparison)
print('Negative delta = improved common 1-degree-bin predictive log score. No 2025 tuning.')
dev_results.to_csv(V3_DIR/'development_validation.csv',index=False)
(V3_DIR/'development_comparison.json').write_text(json.dumps(dev_comparison,indent=2))
joblib.dump(dev_fits,V3_DIR/'development_models.joblib')


# 5C. V3 full Bayesian fit. Existing V1/V2 posteriors remain unchanged.
# The hurdle component describes recorded exact zeros; it is not proof of a physical state.
if dev_comparison['player_equal_delta_V3_minus_baseline'] >= 0:
    raise RuntimeError('V3 did not improve the development log score; review before full fit.')
design3,_,transform3=prepare_v3_design(v3_data,v3_data,True)
Xm3,Xk3,Xz3=design3
init_map3=fit_population_map(v3_data,design3,True)
y3=v3_data.turn_angle_rad.to_numpy(); zero3=y3==0; nonzero3=np.flatnonzero(~zero3)
a3,b3,c3=Xm3.shape[1],Xk3.shape[1],Xz3.shape[1]
# Stable index ordering exactly matches the current V3 rows.
player3_ids=np.sort(v3_data.nfl_id.unique())
player3_idx=pd.Categorical(v3_data.nfl_id,categories=player3_ids).codes
attempt3_labels=v3_data[SEQ].astype(str).agg('|'.join,axis=1)
attempt3_idx,attempt3_levels=pd.factorize(attempt3_labels,sort=True)
coords3={'mean_feature':np.arange(a3),'kappa_feature':np.arange(b3),
         'zero_feature':np.arange(c3),'player':player3_ids.astype(str),
         'attempt':attempt3_levels.tolist()}
with pm.Model(coords=coords3) as circular_model_v3:
    beta3=pm.Normal('beta_mu',0,.10,dims='mean_feature')
    kg3=np.zeros(b3);kg3[0]=np.log(400.);ks3=np.full(b3,.7);ks3[0]=1.5
    gamma3=pm.Normal('beta_kappa',kg3,ks3,dims='kappa_feature')
    azmean3=np.zeros(c3);azmean3[0]=logit(.08);azsd3=np.ones(c3);azsd3[0]=1.5
    alpha3=pm.Normal('beta_zero',azmean3,azsd3,dims='zero_feature')
    sigma_p3=pm.HalfNormal('sigma_player',.7)
    pz3=pm.Normal('player_z',0,1,dims='player')
    effect_p3=pm.Deterministic('player_effect',sigma_p3*pz3,dims='player')
    sigma_a3=pm.HalfNormal('sigma_attempt_mean',.03)
    za3=pm.Normal('attempt_z',0,1,dims='attempt')
    ame3=sigma_a3*(za3[attempt3_idx]-pt.mean(za3))
    mu3=2*pt.arctan(pt.dot(Xm3,beta3)+ame3)
    kap3=pt.exp(pt.dot(Xk3,gamma3)+effect_p3[player3_idx])
    pm.Bernoulli('recorded_zero',logit_p=pt.dot(Xz3,alpha3),observed=zero3.astype('int8'))
    pm.CustomDist('nonzero_turn',mu3[nonzero3],kap3[nonzero3],
                  logp=stable_vonmises_logp,observed=y3[nonzero3])
    initial3={'beta_mu':init_map3.x[:a3],'beta_kappa':init_map3.x[a3:a3+b3],
              'beta_zero':init_map3.x[a3+b3:]}
    assert np.isfinite(circular_model_v3.compile_logp()(circular_model_v3.initial_point()))
    idata_v3=pm.sample(draws=1000,tune=1000,chains=4,cores=2,target_accept=.95,
                       init='adapt_diag',initvals=initial3,random_seed=2030,
                       idata_kwargs={'log_likelihood':False})
idata_v3.to_netcdf(V3_DIR/'posterior.nc')
joblib.dump(transform3,V3_DIR/'design_transform.joblib')
v3_data.to_csv(V3_DIR/'model_data_2023_2024.csv',index=False)
np.savez_compressed(V3_DIR/'designs.npz',mean=Xm3,kappa=Xk3,zero=Xz3,player_index=player3_idx,attempt_index=attempt3_idx)
print('V3 posterior saved:',V3_DIR)
print('2025 not used. V3 is an exploratory recorded-angle hurdle model, not a physical mechanism or NFL ranking.')


# 5D. V3 diagnostics, same-row V2 comparison, and full saved outputs.
vars3=['beta_mu','beta_kappa','beta_zero','sigma_player','player_z','sigma_attempt_mean','attempt_z']
diag3=az.summary(idata_v3,var_names=vars3,hdi_prob=.95)
r3hat=az.rhat(idata_v3,var_names=vars3);e3ess=az.ess(idata_v3,var_names=vars3,method='bulk')
max_rhat3=max(float(v.max()) for v in r3hat.data_vars.values())
min_ess3=min(float(v.min()) for v in e3ess.data_vars.values())
div3=int(idata_v3.sample_stats.diverging.sum());bfmi3=float(np.min(az.bfmi(idata_v3)))
depth3=int(idata_v3.sample_stats.tree_depth.max())
sampler_ok3=bool(div3==0 and max_rhat3<=1.01 and min_ess3>=400 and bfmi3>.3 and depth3<10)
diag3.to_csv(V3_DIR/'mcmc_diagnostics.csv')
p3post=idata_v3.posterior.stack(sample=('chain','draw'))
s3rng=np.random.default_rng(2031);sel3=s3rng.choice(p3post.sizes['sample'],800,replace=False)
bm3=p3post.beta_mu.transpose('sample','mean_feature').values[sel3]
bk3=p3post.beta_kappa.transpose('sample','kappa_feature').values[sel3]
ba3=p3post.beta_zero.transpose('sample','zero_feature').values[sel3]
u3=p3post.player_effect.transpose('sample','player').values[sel3]
az3=p3post.attempt_z.transpose('sample','attempt').values[sel3]
sa3=p3post.sigma_attempt_mean.values[sel3]
pmu3=2*np.arctan(bm3@Xm3.T+sa3[:,None]*(az3[:,attempt3_idx]-az3.mean(axis=1,keepdims=True)))
pkap3=np.exp(bk3@Xk3.T+u3[:,player3_idx]);pzero3=expit(ba3@Xz3.T)
yrep3=s3rng.vonmises(pmu3,pkap3)
yrep3[s3rng.random(yrep3.shape)<pzero3]=0.

def summary_checks3(obs,rep):
    obs_stat=np.array([np.cos(obs).mean(),np.sin(obs).mean(),np.quantile(np.abs(obs),.9),
                       np.mean(obs==0),np.mean(np.abs(obs)<=np.radians(.5))])
    rep_stat=np.column_stack([np.cos(rep).mean(axis=1),np.sin(rep).mean(axis=1),
                               np.quantile(np.abs(rep),.9,axis=1),(rep==0).mean(axis=1),
                               (np.abs(rep)<=np.radians(.5)).mean(axis=1)])
    result=pd.DataFrame({'statistic':['mean_cos','mean_sin','abs_angle_90pct','exact_zero_rate','within_half_degree'],
                         'observed':obs_stat,'predictive_low':np.quantile(rep_stat,.025,axis=0),
                         'predictive_high':np.quantile(rep_stat,.975,axis=0)})
    result['inside_interval']=result.observed.between(result.predictive_low,result.predictive_high)
    return result
ppc3=summary_checks3(y3,yrep3);display(ppc3)
ppc3.to_csv(V3_DIR/'conditional_predictive_checks.csv',index=False)
ppc3_ok=bool(ppc3.inside_interval.all())
bydrill3=[]
for d in DRILLS:
    mask=v3_data.drill_name.eq(d).to_numpy()
    pcheck=summary_checks3(y3[mask],yrep3[:,mask]);pcheck['drill_name']=d;bydrill3.append(pcheck)
bydrill3=pd.concat(bydrill3,ignore_index=True)
display(bydrill3.loc[bydrill3.statistic.isin(['mean_sin','exact_zero_rate'])])
bydrill3.to_csv(V3_DIR/'drill_predictive_checks.csv',index=False)
# Mixture circular expectation includes both the point mass and continuous component.
expected_complex3=pzero3+(1-pzero3)*(sp_i1e(pkap3)/sp_i0e(pkap3))*np.exp(1j*pmu3)
predangle3=np.angle(expected_complex3.mean(axis=0))
resid3=np.angle(np.exp(1j*(y3-predangle3)))
# Compare only adjacent sampled moments separated by 0.4s; never across a gap/segment.
def lag_residual3(values):
    t=v3_data[SEG+['timestamp']].copy();t['residual']=values
    t=t.sort_values(SEG+['timestamp']);gr=t.groupby(SEG)
    t['lag']=gr.residual.shift(1);t['lag_t']=gr.timestamp.shift(1)
    valid=(t.timestamp-t.lag_t).dt.total_seconds().sub(.4).abs().le(.025)
    return float(t.loc[valid,['residual','lag']].corr().iloc[0,1]),int(valid.sum())
corr_v3,pairs3=lag_residual3(resid3);corr_v2_same,_=lag_residual3(resid2)
# V2 on same five checks (zero-event calibration is an added test).
ppc2_extended=summary_checks3(y3,yrep2)
comparison3=pd.DataFrame([
 dict(version='V2',max_Rhat=max_rhat2,min_bulk_ESS=min_ess2,divergences=div2,sampler_pass=sampler_ok2,
      five_PPC_pass=bool(ppc2_extended.inside_interval.all()),gap_checked_residual_corr=corr_v2_same),
 dict(version='V3',max_Rhat=max_rhat3,min_bulk_ESS=min_ess3,divergences=div3,sampler_pass=sampler_ok3,
      five_PPC_pass=ppc3_ok,gap_checked_residual_corr=corr_v3)])
display(comparison3)
print('Same-row 0.4s residual pairs:',pairs3)
comparison3.to_csv(V3_DIR/'v2_v3_diagnostics.csv',index=False)
# Recorded zeros have no automatic biological interpretation.
status3=dict(sampler_checks_pass=sampler_ok3,conditional_PPC_pass=ppc3_ok,
             residual_corr=corr_v3,previous_residual_corr_same_pairs=corr_v2_same,
             development_comparison=dev_comparison,
             heldout_2025_used=False,NFL_validation_done=False,
             status='EXPLORATORY_NOT_READY_FOR_PLAYER_RANKING')
(V3_DIR/'status.json').write_text(json.dumps(status3,indent=2))
fig3,ax3=plt.subplots(2,2,figsize=(13,8))
for ax,rep,label in [(ax3[0,0],yrep2,'V2 conditional fit'),(ax3[0,1],yrep3,'V3 dynamic hurdle')]:
    bins3=np.linspace(-30,30,81)
    ax.hist(np.degrees(y3),bins=bins3,density=True,alpha=.4,label='Observed')
    ax.hist(np.degrees(rep).ravel(),bins=bins3,density=True,histtype='step',color='darkorange',label='Predictive')
    ax.set(title=label,xlabel='Turn angle (degrees)');ax.legend()
labels3=['Observed','V2 predictive','V3 predictive']
ax3[1,0].bar(labels3,[np.mean(y3==0),np.mean(yrep2==0),np.mean(yrep3==0)],color=['steelblue','gray','teal'])
ax3[1,0].set(title='Exact recorded-zero rate',ylabel='Proportion')
ax3[1,1].bar(['V2','V3'],[corr_v2_same,corr_v3],color=['gray','teal'])
ax3[1,1].axhline(0,color='black',linewidth=.7)
ax3[1,1].set(title='Residual correlation at 0.4s',ylabel='Pearson correlation')
plt.tight_layout();fig3.savefig(V3_DIR/'v3_diagnostics.png',dpi=160,bbox_inches='tight');plt.show()
print('V3 status:',status3['status'])
print('Conditional training fit only; development population models used 2023->2024; final V3 not tested on 2025.')


# 5E. Durable result package and human-readable summary.
# This does not approve a player ranking or use 2025.
import shutil, base64
from IPython.display import HTML,display
notes3=(
    'V3 dynamic recorded-zero hurdle model\n'
    'Data: 69 WR players / 2790 frames, 2023-2024. 2025 untouched.\n'
    'Observed exact recorded zeros: 230 / 2790. A hurdle models recording distribution, not a proven physical state.\n'
    'Mean: drill-specific cubic progress splines, speed, signed acceleration, previous angle, lag-four angle, lag missing indicator, centered attempt random effect.\n'
    'Concentration: speed, speed squared, signed acceleration, drill, absolute lag-four angle, player random effect.\n'
    'Zero probability: speed, progress, drill, previous exact-zero indicator, lag missing indicator.\n'
    '198 lag-four angles failed prior angle QC; zero imputed with explicit missing indicators. No rows removed.\n'
    '2023->2024 validation compares regularized population MAP models, not the stored hierarchical V2.\n'
    f"Development equal-player log-score delta V3-baseline: {dev_comparison['player_equal_delta_V3_minus_baseline']:.6f}; bootstrap CI [{dev_comparison['bootstrap_ci95_low']:.6f}, {dev_comparison['bootstrap_ci95_high']:.6f}].\n"
    'This is development evidence: 2024 was previously used in V1/V2 training, not a pristine external test.\n'
    f'Full V3 max Rhat: {max_rhat3:.6f}; min bulk ESS: {min_ess3:.1f}; divergences: {div3}.\n'
    f'Conditional five-summary PPC all passed: {ppc3_ok}.\n'
    f'Residual lag correlation on same valid 0.4s pairs: V2 {corr_v2_same:.6f}; V3 {corr_v3:.6f}.\n'
    'PPC conditions on observed lagged angles and fitted training effects; no full trajectory simulation or new-cohort Bayesian validation.\n'
    'No validated NFL performance claims or player ranking.\n'
)
(V3_DIR/'README_results.txt').write_text(notes3)
# Save combined files in a separate versioned archive.
archive3=Path(shutil.make_archive(str(DATA_DIR/'wr_bayesian_v3_results'),'zip',root_dir=V3_DIR))
print(notes3)
print('V3 result ZIP:',archive3,'MiB:',round(archive3.stat().st_size/1024**2,2))
# Visible UI download links, not an external transmission.
for pth,title in [(V3_DIR/'v3_diagnostics.png','V3 diagnostics figure'),(V3_DIR/'status.json','V3 status JSON')]:
    mime='image/png' if pth.suffix=='.png' else ('application/json' if pth.suffix=='.json' else 'application/zip')
    payload3=base64.b64encode(pth.read_bytes()).decode('ascii')
    if pth.suffix=='.png':display(HTML(f'<img alt="V3 diagnostics export" src="data:{mime};base64,{payload3}" style="max-width:100%">'))
    else:display(HTML(f'<a download="{pth.name}" href="data:{mime};base64,{payload3}">{title}</a>'))
