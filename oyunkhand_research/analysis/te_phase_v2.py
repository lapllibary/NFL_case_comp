"""TE movement development V2. No 2025 data or outcome-based window selection.
Inputs: te_v2_inputs/{tracking.csv,player_features.csv,routes.csv}.
Dependencies: numpy,pandas. Outputs: te_phase_v2_outputs. Run locally or Colab.
"""
from pathlib import Path
import os,json,hashlib
import numpy as np
import pandas as pd
D=Path(os.environ.get('TE_V2_INPUT','/content/te_v2_inputs'))
O=Path(os.environ.get('TE_V2_OUTPUT','/content/te_phase_v2_outputs'));O.mkdir(parents=True,exist_ok=True)
SEED=2041; ALPHAS=[.1,1.,10.,100.]
VARIANTS=['Training-player mean','Context only','Combine baseline','Original aggregate','Original separate drills','First sustained bend 0.6s','First sustained bend 0.8s','Two-penalty Combine','Two-penalty first bend']
GRIDS={v:[(c,p) for c in [.1,1.] for p in [10.,100.,1000.,10000.]] for v in ['Two-penalty Combine','Two-penalty first bend']}
def grid_for(variant):return GRIDS.get(variant,ALPHAS)
SPEC=dict(stage='Development only; all 2023-2024 data have been explored previously',
    no_2025=True,primary_comparison='same 15 players; nested leave-one-player-out; pooled development diagnosis, not forward-time validation',
    variants=VARIANTS,alphas=ALPHAS,two_penalty_grids=GRIDS,
    phase_rule='First contiguous same-sign episode with absolute chord angle >=15 degrees for >=3 candidate centers; select largest absolute angle inside that episode.',
    windows_seconds=[.6,.8],progress=[.15,.85],min_chord_yards=.3,min_approach_speed=2.,
    min_frame_speed=0.,max_frame_speed=15.,dt=.1,dt_tolerance=.015,
    sample_policy='Same original 15 eligible development players; missing phase measurements imputed within training fold plus missing indicator.',
    features='6 conventional Combine features and original route/alignment/coverage/game context; tracking split by drill for new versions',
    uncertainty='paired player bootstrap conditional on nested CV predictions; overlapping folds and training uncertainty not captured',
    interpretation='Operational first bend, not film-validated anatomical/route-break ground truth; conditional NFL context model, not pure pre-draft forecast',
    selection='No winner claimed as externally validated; no automatic rerun on 2025',seed=SEED)
(O/'protocol.json').write_text(json.dumps(SPEC,indent=2))
t=pd.read_csv(D/'tracking.csv',dtype={'event_id':'string'});pf=pd.read_csv(D/'player_features.csv');routes=pd.read_csv(D/'routes.csv')
assert set(t.draft_year)<=set([2023,2024]);assert set(pf.draft_year)<=set([2023,2024]);assert set(routes.draft_year)=={2023,2024}
assert routes.nfl_id.nunique()==15
SEQ=['nfl_id','draft_year','event_id','drill_type','drill_name','attempt']
t['timestamp']=pd.to_datetime(t.time,utc=True);assert not t.duplicated(SEQ+['timestamp']).any()
phase=[];candidate_rows=[]
for labels,g in t.groupby(SEQ,sort=True):
 g=g.sort_values('timestamp');xy=g[['x','y']].to_numpy(float);ns=g.timestamp.astype('int64').to_numpy();dt=np.diff(ns)/1e9
 tt=(ns-ns[0])/1e9;speed=np.r_[np.nan,np.linalg.norm(np.diff(xy,axis=0),axis=1)/dt]
 good=np.r_[False,(abs(dt-.1)<=.015)&(speed[1:]>=0)&(speed[1:]<=15)]
 length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];progress=length/length[-1] if length[-1]>0 else np.zeros(len(xy))
 for half in [3,4]:
  candidates=[]
  for j in range(8,len(xy)-8):
   if not .15<=progress[j]<=.85 or not good[j-7:j+9].all():continue
   incoming=xy[j]-xy[j-half];outgoing=xy[j+half]-xy[j]
   if min(np.linalg.norm(incoming),np.linalg.norm(outgoing))<.3:continue
   angle=np.degrees(np.arctan2(incoming[0]*outgoing[1]-incoming[1]*outgoing[0],np.dot(incoming,outgoing)))
   approach=float(np.median(speed[j-6:j-2]));through=float(min(np.mean(speed[k-1:k+2]) for k in range(j-2,j+3)))
   if approach<2 or abs(angle)<15:continue
   v=dict(j=j,seconds=float(tt[j]),angle=float(angle),retention=through/approach,approach=approach,through=through)
   candidates.append(v);candidate_rows.append(dict(zip(SEQ,labels))|dict(half=half,**v))
  episodes=[]
  for c in candidates:
   if not episodes or c['j']!=episodes[-1][-1]['j']+1 or np.sign(c['angle'])!=np.sign(episodes[-1][-1]['angle']):episodes.append([])
   episodes[-1].append(c)
  eligible=[ep for ep in episodes if len(ep)>=3]
  row=dict(zip(SEQ,labels))|dict(half_window_frames=half,episodes=len(eligible),status='valid' if eligible else 'no_sustained_episode')
  if eligible:
   c=max(eligible[0],key=lambda x:abs(x['angle']));row.update(c);row['episode_start_seconds']=eligible[0][0]['seconds'];row['episode_end_seconds']=eligible[0][-1]['seconds']
  phase.append(row)
phase=pd.DataFrame(phase);phase.to_csv(O/'attempt_phase_metrics.csv',index=False)
pd.DataFrame(candidate_rows).to_csv(O/'all_phase_candidates.csv',index=False)
DRILLS=['HOOK_ROUTE_RIGHT','IN_ROUTE_LEFT'];BASE=['forty','ten_yd_split','three_cone','short_shuttle','combine_height','combine_weight'];PN=['down','yards_to_go','cushion','pass_length'];PC=['route_ran','lined_up_position','team_coverage_man_zone']
original=['retention_'+x for x in DRILLS]
for h in [3,4]:
 q=phase[phase.half_window_frames.eq(h)&phase.status.eq('valid')].groupby(['nfl_id','drill_name']).retention.median().unstack().reindex(columns=DRILLS)
 q.columns=[f'phase{h}_{x}' for x in DRILLS];pf=pf.merge(q,on='nfl_id',how='left',validate='one_to_one')
pf.to_csv(O/'development_features.csv',index=False);F=pf.set_index('nfl_id')
PRED_CACHE={};CACHE_META={}
def fit_grid(train_ids,test_ids,variant):
 train_ids=tuple(sorted(train_ids));test_ids=tuple(sorted(test_ids));key=(train_ids,test_ids,variant)
 if key in PRED_CACHE:return PRED_CACHE[key]
 assert set(train_ids).isdisjoint(test_ids)
 train=routes[routes.nfl_id.isin(train_ids)];test=routes[routes.nfl_id.isin(test_ids)]
 if variant=='Training-player mean':
  value=float(train.groupby('nfl_id').separation_at_pass_forward.mean().mean());pred={alpha:np.full(len(test),value) for alpha in ALPHAS};PRED_CACHE[key]=pred;CACHE_META[key]=dict(parameters=1);return pred
 cols_base=[] if variant=='Context only' else BASE
 a=F.loc[list(train_ids),cols_base].astype(float).copy();b=F.loc[list(test_ids),cols_base].astype(float).copy()
 if variant=='Original aggregate':
  z=F.loc[list(train_ids),original];med=z.median();sc=(z.quantile(.75)-z.quantile(.25))/1.349;sc=sc.replace(0,np.nan).fillna(1.)
  a['movement']=((F.loc[list(train_ids),original]-med)/sc).mean(axis=1);b['movement']=((F.loc[list(test_ids),original]-med)/sc).mean(axis=1)
 elif variant in ['Original separate drills','First sustained bend 0.6s','First sustained bend 0.8s','Two-penalty first bend']:
  cols=original if variant=='Original separate drills' else [f"phase{3 if variant.endswith('0.6s') or variant=='Two-penalty first bend' else 4}_{x}" for x in DRILLS]
  for col in cols:a[col]=F.loc[list(train_ids),col];b[col]=F.loc[list(test_ids),col]
 m=a.median().fillna(0.);center=a.fillna(m).mean();sd=a.fillna(m).std(ddof=0).replace(0,1.).fillna(1.)
 def ptransform(v):return np.column_stack([((v.fillna(m)-center)/sd).to_numpy(),v.isna().astype(float).to_numpy()])
 aa=pd.DataFrame(ptransform(a),index=train_ids);bb=pd.DataFrame(ptransform(b),index=test_ids)
 w=1/train.groupby('nfl_id').nfl_id.transform('size').to_numpy(float)
 assert np.allclose(pd.Series(w,index=train.nfl_id).groupby(level=0).sum(),1)
 q=train[PN].astype(float);r=test[PN].astype(float);qm=q.median().fillna(0);qc=np.average(q.fillna(qm),weights=w,axis=0);qs=np.sqrt(np.average((q.fillna(qm).to_numpy()-qc)**2,weights=w,axis=0));qs[qs==0]=1
 def context(v):return np.column_stack([(v.fillna(qm).to_numpy()-qc)/qs,v.isna().astype(float).to_numpy()])
 xt=[np.ones((len(train),1)),aa.loc[train.nfl_id].to_numpy(),context(q)];xv=[np.ones((len(test),1)),bb.loc[test.nfl_id].to_numpy(),context(r)]
 category_levels={}
 for col in PC:
  levels=sorted(train[col].fillna('MISSING').astype(str).unique());category_levels[col]=levels
  for level in levels[1:]:
   xt.append(train[col].fillna('MISSING').astype(str).eq(level).to_numpy(float)[:,None]);xv.append(test[col].fillna('MISSING').astype(str).eq(level).to_numpy(float)[:,None])
 X=np.column_stack(xt);V=np.column_stack(xv);y=train.separation_at_pass_forward.to_numpy(float)
 gram=X.T@(w[:,None]*X);rhs=X.T@(w*y);pen=np.eye(X.shape[1]);pen[0,0]=0
 pred={};betas={}
 for alpha in grid_for(variant):
  if isinstance(alpha,tuple):
   ca,pa=alpha;penalty=pen*ca;np.fill_diagonal(penalty[1:1+2*a.shape[1],1:1+2*a.shape[1]],pa)
  else:penalty=alpha*pen
  beta=np.linalg.solve(gram+penalty,rhs);betas[str(alpha)]=beta.tolist();pred[alpha]=np.maximum(V@beta,0.)
 PRED_CACHE[key]=pred;CACHE_META[key]=dict(parameters=X.shape[1],player_columns=list(a.columns),player_median=m.to_dict(),player_center=center.to_dict(),player_sd=sd.to_dict(),context_numeric=PN,context_median=qm.to_dict(),context_center=dict(zip(PN,qc)),context_sd=dict(zip(PN,qs)),category_levels=category_levels,coefficients=betas);return pred

def score_table(test_ids,pred):
 test=routes[routes.nfl_id.isin(test_ids)];q=test[['nfl_id','draft_year','separation_at_pass_forward']].copy();q['predicted']=pred
 q=q.groupby(['nfl_id','draft_year'],as_index=False).agg(observed=('separation_at_pass_forward','mean'),predicted=('predicted','mean'),routes=('nfl_id','size'))
 q['absolute_error']=abs(q.observed-q.predicted);return q

def tune(ids,variant):
 errs={alpha:[] for alpha in grid_for(variant)}
 for pid in sorted(ids):
  p=fit_grid(set(ids)-{pid},[pid],variant)
  for alpha in grid_for(variant):errs[alpha].append(score_table([pid],p[alpha]).absolute_error.iloc[0])
 means={alpha:float(np.mean(v)) for alpha,v in errs.items()}
 return min(grid_for(variant),key=lambda a:means[a]),means
# Reproduce the original temporal development comparison before changing features.
train23=sorted(routes.loc[routes.draft_year.eq(2023),'nfl_id'].unique());test24=sorted(routes.loc[routes.draft_year.eq(2024),'nfl_id'].unique())
reference={'Combine baseline':(.1,.48590400976667236),'Original aggregate':(100.,.359974825619093)}
for variant,(alpha,expected) in reference.items():
 result=score_table(test24,fit_grid(train23,test24,variant)[alpha]);assert abs(result.absolute_error.mean()-expected)<1e-8,(variant,result.absolute_error.mean(),expected)
ids=sorted(routes.nfl_id.unique());nested=[];temporal=[];tuning=[]
for variant in VARIANTS:
 for pid in ids:
  tr=set(ids)-{pid};alpha,inner=tune(tr,variant);pred=fit_grid(tr,[pid],variant)[alpha]
  q=score_table([pid],pred);q['model']=variant;q['alpha']=str(alpha);nested.append(q)
  for aa,err in inner.items():tuning.append(dict(model=variant,outer_player=int(pid),alpha=aa,inner_MAE=err))
 alpha,inner=tune(set(train23),variant);q=score_table(test24,fit_grid(train23,test24,variant)[alpha]);q['model']=variant;q['alpha']=str(alpha);temporal.append(q)
 print('Completed:',variant,flush=True)
nested=pd.concat(nested,ignore_index=True);temporal=pd.concat(temporal,ignore_index=True)
nested.to_csv(O/'nested_player_predictions.csv',index=False);temporal.to_csv(O/'temporal_development_predictions.csv',index=False);pd.DataFrame(tuning).to_csv(O/'inner_alpha_CV.csv',index=False)
def summarize(p,split):
 return pd.DataFrame([dict(split=split,model=m,players=len(q),MAE=float(q.absolute_error.mean()),RMSE=float(np.sqrt(np.mean((q.predicted-q.observed)**2)))) for m,q in p.groupby('model',sort=False)])
metrics=pd.concat([summarize(nested,'nested_LOPO_2023_24'),summarize(temporal,'temporal_2023_to_2024')],ignore_index=True)
metrics.to_csv(O/'model_comparison.csv',index=False)
unc=[];rng=np.random.default_rng(SEED)
for split,p in [('nested_LOPO_2023_24',nested),('temporal_2023_to_2024',temporal)]:
 err=p.pivot(index='nfl_id',columns='model',values='absolute_error')
 for m in [v for v in VARIANTS if v!='Combine baseline']:
  delta=(err[m]-err['Combine baseline']).to_numpy();boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(axis=1);lo,hi=np.quantile(boot,[.025,.975]);unc.append(dict(split=split,model=m,delta_MAE=float(delta.mean()),CI_low=float(lo),CI_high=float(hi)))
unc=pd.DataFrame(unc);unc.to_csv(O/'paired_player_uncertainty.csv',index=False)
# Common-alpha diagnostic isolates differences from separately selected penalties.
common=[]
for m in VARIANTS:
 if m in GRIDS:continue
 q=score_table(test24,fit_grid(train23,test24,m)[100.]);common.append(dict(model=m,alpha=100.,temporal_MAE=float(q.absolute_error.mean())))
pd.DataFrame(common).to_csv(O/'common_alpha_diagnostic.csv',index=False)
# Held-out outcome permutation must not affect fitted transforms or predictions.
check_pid=ids[0];key=(tuple(sorted(set(ids)-{check_pid})),(check_pid,),'First sustained bend 0.6s')
p1=fit_grid(set(ids)-{check_pid},[check_pid],key[2])[100.].copy();saved=routes.loc[routes.nfl_id.eq(check_pid),'separation_at_pass_forward'].copy();routes.loc[routes.nfl_id.eq(check_pid),'separation_at_pass_forward']=rng.permutation(saved)
PRED_CACHE.pop(key);p2=fit_grid(set(ids)-{check_pid},[check_pid],key[2])[100.];assert np.array_equal(p1,p2)
routes.loc[routes.nfl_id.eq(check_pid),'separation_at_pass_forward']=saved
assert len(nested)==15*len(VARIANTS);assert not nested.duplicated(['model','nfl_id']).any()
(O/'validation_checks.txt').write_text('PASS: no 2025 inputs; original temporal scores reproduced; unique player/play and phase keys; disjoint inner/outer folds; equal player training weights; training-only transforms; permuted held-out outcomes leave predictions identical; no automatic 2025 evaluation.\n')
print(metrics.to_string(index=False));print(unc.to_string(index=False))
# Fit development candidates on all 15 players; these coefficients are not a new validation.
from te_phase_v2_predict import predict_saved_model
import copy
finals=[]
for variant in ['Context only','Two-penalty Combine','Two-penalty first bend']:
 alpha,cv=tune(set(ids),variant);fit_grid(ids,[],variant)
 key=(tuple(ids),tuple(),variant);meta=copy.deepcopy(CACHE_META[key]);betas=meta.pop('coefficients')
 model=dict(model=variant,stage='development fit; no new-cohort validation',training_players=[int(v) for v in ids],training_cohorts=[2023,2024],alpha=alpha,alpha_selection='leave-one-player-out within all 15 development players',transforms=meta,coefficients=betas[str(alpha)],feature_rule=SPEC['phase_rule'] if variant.endswith('first bend') else None)
 slug=variant.lower().replace(' ','_').replace('-','_');path=O/('model_'+slug+'.json');path.write_text(json.dumps(model,indent=2,allow_nan=False));finals.append(dict(model=variant,alpha=str(alpha),file=path.name))
 # Serialize a held-out fit and verify standalone inference reproduces CV predictions.
 pid=ids[0];tr=set(ids)-{pid};aa,_=tune(tr,variant);pred=fit_grid(tr,[pid],variant)[aa];kk=(tuple(sorted(tr)),(pid,),variant)
 mm=copy.deepcopy(CACHE_META[kk]);bb=mm.pop('coefficients');check=O/'_inference_check.json';check.write_text(json.dumps(dict(transforms=mm,coefficients=bb[str(aa)]),allow_nan=False))
 pred2,_=predict_saved_model(check,routes[routes.nfl_id.eq(pid)],pf);assert np.allclose(pred,pred2,rtol=1e-12,atol=1e-12);check.unlink()
pd.DataFrame(finals).to_csv(O/'final_development_models.csv',index=False)
with (O/'validation_checks.txt').open('a') as f:f.write('PASS: serialized model + standalone inference matches held-out CV route predictions for all three development candidates.\n')
print('Saved 3 development models with training-only transforms; independent inference verified.')
