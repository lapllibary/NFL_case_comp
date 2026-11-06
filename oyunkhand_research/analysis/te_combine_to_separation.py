# 6A. TE Combine -> NFL separation: урьдчилан тогтоосон протокол.
from pathlib import Path
import os,json,hashlib
import numpy as np
import pandas as pd
TE_DATA=Path(os.environ.get('NFL_DATA_DIR','/content'))
TE_OUT=Path(os.environ.get('TE_OUTPUT_DIR','/content/te_separation_outputs'))
TE_OUT.mkdir(parents=True,exist_ok=True)
TE_SPEC=dict(position='combine_position=TE AND nfl_position=TE',
    drills=['HOOK_ROUTE_RIGHT','IN_ROUTE_LEFT'],
    outcome='separation_at_pass_forward',outcome_units='yards',
    outcome_definition='Distance to nearest defender at forward-pass release; supplied by NFL, not recomputed from prospect-only tracking.',
    nfl_window='rookie regular season only: season == draft_year and season_type == REG',
    passes=['C','I','IN'],minimum_routes=20,minimum_drills=2,
    dt=.1,dt_tolerance=.015,speed_min=0.,speed_max=15.,heading_min_displacement_yards=.3,
    heading_half_window_frames=3,min_turn_degrees=20.,minimum_approach_speed=2.,
    progress_min=.15,progress_max=.85,
    primary_tracking_feature='equal-drill standardized median speed retention at largest eligible 0.6s heading-change window',
    zero_missing_policy='retain audit flags; no interpolation across gaps',
    development='train 2023 -> validate 2024; alpha selected by player-held-out CV within 2023',
    final='train 2023-2024 -> evaluate 2025 once, no 2025 hyperparameter selection',
    regularization_grid=[.1,1.,10.,100.],
    primary_error='MAE across players of predicted versus observed rookie mean separation',
    weighting='each training player total weight 1 regardless of route count',seed=2040)
(TE_OUT/'analysis_protocol.json').write_text(json.dumps(TE_SPEC,indent=2))
te_players_all=pd.read_csv(TE_DATA/'players.csv')
te_combine_all=pd.read_csv(TE_DATA/'combine_results.csv')
te_track_all=pd.read_csv(TE_DATA/'combine_tracking.csv',dtype={'event_id':'string'})
te_ids_combine=set(te_combine_all.loc[te_combine_all.combine_position.eq('TE'),'nfl_id'])
te_ids=set(te_players_all.loc[te_players_all.nfl_position.eq('TE'),'nfl_id']) & te_ids_combine
te_identity_audit=te_players_all.loc[te_players_all.nfl_id.isin(te_ids_combine),['nfl_id','display_name','draft_year','nfl_position']].copy()
te_identity_audit['included_TE']=te_identity_audit.nfl_id.isin(te_ids)
te_identity_audit.to_csv(TE_OUT/'position_audit.csv',index=False)
te_tracking=te_track_all.loc[te_track_all.nfl_id.isin(te_ids)&te_track_all.entity_type.eq('PLAYER')].copy()
te_drill_coverage=te_tracking.groupby(['drill_type','drill_name']).agg(players=('nfl_id','nunique'),frames=('nfl_id','size')).reset_index()
te_drill_coverage.to_csv(TE_OUT/'drill_coverage.csv',index=False)
print('Combine-position TE:',len(te_ids_combine),'Both position labels TE:',len(te_ids))
display(te_drill_coverage) if 'display' in globals() else print(te_drill_coverage.to_string(index=False))


# 6B. TE-ийн өөрийн сорилоос шууд тайлбарлагдах хөдөлгөөний хэмжүүр гаргах.
# Uses the same coordinate-derived turning/QC principle as WR; not a claim that WR V3 is validated for TEs.
TE_SEQ=['nfl_id','draft_year','event_id','drill_type','drill_name','attempt']
mt=te_tracking.loc[te_tracking.drill_name.isin(TE_SPEC['drills'])].copy()
mt['timestamp']=pd.to_datetime(mt.time,utc=True,errors='coerce')
key=TE_SEQ+['timestamp']
if mt[key].isna().any().any():raise ValueError('TE sequence key/time missing')
mt=mt.drop_duplicates().sort_values(key)
if mt.duplicated(key).any():raise ValueError('Conflicting TE tracking duplicate')
metrics=[];attempt_audit=[]
for labels,g in mt.groupby(TE_SEQ,sort=False,observed=True):
    g=g.sort_values('timestamp');n=len(g)
    row=dict(zip(TE_SEQ,labels));row['frames']=n
    if n<17:
        row['status']='too_short';attempt_audit.append(row);continue
    xy=g[['x','y']].to_numpy(float);ts=g.timestamp.astype('int64').to_numpy()/1e9
    dt=np.diff(ts);disp=np.diff(xy,axis=0)
    speed=np.r_[np.nan,np.linalg.norm(disp,axis=1)/dt]
    good=np.r_[False,(np.abs(dt-.1)<=.015)&(speed[1:]>=0)&(speed[1:]<=15)]
    path=np.r_[0,np.cumsum(np.linalg.norm(disp,axis=1))]
    progress=path/path[-1] if path[-1]>0 else np.full(n,np.nan)
    candidates=[]
    for j in range(8,n-8):
        if not (.15<=progress[j]<=.85):continue
        if not good[j-7:j+9].all():continue
        incoming=xy[j]-xy[j-3];outgoing=xy[j+3]-xy[j]
        if min(np.linalg.norm(incoming),np.linalg.norm(outgoing))<.3:continue
        angle=np.arctan2(incoming[0]*outgoing[1]-incoming[1]*outgoing[0],np.dot(incoming,outgoing))
        approach=float(np.median(speed[j-6:j-2]));post=float(np.median(speed[j+3:j+7]))
        through=float(min(np.mean(speed[k-1:k+2]) for k in range(j-2,j+3)))
        if approach<2 or abs(np.degrees(angle))<20:continue
        candidates.append((abs(angle),j,float(angle),approach,post,through))
    if not candidates:
        row['status']='no_eligible_turn_window';attempt_audit.append(row);continue
    _,j,angle,approach,post,through=max(candidates,key=lambda v:v[0])
    row.update(status='valid',turn_angle_deg=np.degrees(angle),turn_progress=progress[j],
               approach_speed=approach,through_speed=through,post_speed=post,
               speed_retention=through/approach,braking_proxy=(approach-through)/.4,
               reacceleration_proxy=(post-through)/.5,
               peak_time=str(g.timestamp.iloc[j]))
    metrics.append(row.copy());attempt_audit.append(row)
te_attempt_audit=pd.DataFrame(attempt_audit);te_attempts=pd.DataFrame(metrics)
if te_attempts.empty:raise ValueError('No eligible TE movement windows')
te_attempt_audit.to_csv(TE_OUT/'attempt_QC_audit.csv',index=False)
te_attempts.to_csv(TE_OUT/'attempt_movement_metrics.csv',index=False)
# One vote per drill, first median across attempts; no outcome-dependent feature weights.
te_drill_features=te_attempts.groupby(['nfl_id','draft_year','drill_name']).agg(
    retention=('speed_retention','median'),braking=('braking_proxy','median'),
    reacceleration=('reacceleration_proxy','median'),turn_degrees=('turn_angle_deg',lambda s:s.abs().median()),
    attempts=('nfl_id','size')).reset_index()
te_drill_features.to_csv(TE_OUT/'player_drill_features.csv',index=False)
wide=te_drill_features.pivot(index='nfl_id',columns='drill_name',values='retention').reindex(columns=TE_SPEC['drills'])
wide.columns=['retention_'+d for d in wide.columns]
wide['valid_drills']=wide.notna().sum(axis=1)
te_player_features=te_combine_all.loc[te_combine_all.nfl_id.isin(te_ids)].merge(
    te_players_all[['nfl_id','display_name','nfl_position']],on='nfl_id',validate='one_to_one').merge(
    wide.reset_index(),on='nfl_id',how='left',validate='one_to_one')
te_player_features['valid_drills']=te_player_features.valid_drills.fillna(0).astype(int)
te_player_features.to_csv(TE_OUT/'player_combine_features.csv',index=False)
print('Tracking attempts:',len(te_attempt_audit),'valid:',len(te_attempts))
print('Players with >=2 valid drills:',int(te_player_features.valid_drills.ge(2).sum()))
print(te_attempt_audit.status.value_counts().to_string())
print(te_drill_features.groupby('drill_name').agg(players=('nfl_id','nunique'),attempts=('attempts','sum')).to_string())


# 6C. NFL rookie separation outcome ба зөв холболт.
ppte=pd.read_csv(TE_DATA/'player_play.csv',low_memory=False)
gmte=pd.read_csv(TE_DATA/'games.csv')
ppte=ppte.loc[ppte.nfl_id.isin(te_ids)].merge(
    te_players_all[['nfl_id','draft_year']],on='nfl_id',validate='many_to_one').merge(
    gmte[['game_id','season','season_type','week']],on='game_id',validate='many_to_one')
if ppte.duplicated(['nfl_id','game_id','play_id']).any():raise ValueError('Repeated player/play key')
rookie=ppte.loc[ppte.season.eq(ppte.draft_year)&ppte.season_type.eq('REG')].copy()
rookie['included_route']=rookie.pass_result.isin(TE_SPEC['passes']) & rookie.play_nullified_by_penalty.eq('N') & rookie.route_ran.notna() & rookie.separation_at_pass_forward.notna()
coverage=rookie.groupby('draft_year').agg(players=('nfl_id','nunique'),charted_rows=('nfl_id','size'),separation_present=('separation_at_pass_forward','count'),eligible_routes=('included_route','sum')).reset_index()
coverage.to_csv(TE_OUT/'NFL_coverage.csv',index=False)
route_data=rookie.loc[rookie.included_route].copy()
if (route_data.separation_at_pass_forward<0).any():raise ValueError('Negative separation')
counts=route_data.groupby('nfl_id').size().rename('separation_routes')
player_outcomes=route_data.groupby('nfl_id').agg(observed_mean_separation=('separation_at_pass_forward','mean'),separation_routes=('nfl_id','size'),games=('game_id','nunique')).reset_index()
eligibility=te_player_features.merge(player_outcomes,on='nfl_id',how='left',validate='one_to_one')
eligibility['analysis_included']=eligibility.valid_drills.ge(2)&eligibility.separation_routes.ge(20)
eligibility.to_csv(TE_OUT/'player_eligibility.csv',index=False)
analysis_ids=set(eligibility.loc[eligibility.analysis_included,'nfl_id'])
route_data=route_data.loc[route_data.nfl_id.isin(analysis_ids)].copy()
cols=['nfl_id','game_id','play_id','draft_year','route_ran','lined_up_position','team_coverage_man_zone','down','yards_to_go','cushion','pass_length','separation_at_pass_forward','target']
route_data=route_data[cols]
route_data.to_csv(TE_OUT/'eligible_rookie_routes.csv',index=False)
print('Coverage counts only; no 2025 model tuning:');print(coverage.to_string(index=False))
print('Eligible player counts by cohort:');print(eligibility.loc[eligibility.analysis_included].groupby('draft_year').size().to_string())


# 6D. Conventional Combine baseline vs one added movement feature.
# Player-level cross-validation, training-only imputation/scaling, equal player weights.
BASE_NUM=['forty','ten_yd_split','three_cone','short_shuttle','combine_height','combine_weight']
PLAY_NUM=['down','yards_to_go','cushion','pass_length']
PLAY_CAT=['route_ran','lined_up_position','team_coverage_man_zone']
RET_COLS=['retention_'+d for d in TE_SPEC['drills']]
FEATURE_LOOKUP=te_player_features.set_index('nfl_id')

def fit_predict_te(train,test,alpha,expanded):
    train_ids=train.nfl_id.unique();test_ids=test.nfl_id.unique()
    if not set(train_ids).isdisjoint(test_ids):raise ValueError('Player leakage')
    pf=FEATURE_LOOKUP.loc[train_ids].copy()
    # Median/IQR references for drill normalization fit only on training players.
    med=pf[RET_COLS].median();scale=(pf[RET_COLS].quantile(.75)-pf[RET_COLS].quantile(.25))/1.349
    scale=scale.replace(0,np.nan).fillna(1.)
    def movement(ids):return ((FEATURE_LOOKUP.loc[ids,RET_COLS]-med)/scale).mean(axis=1)
    def player_block(ids):
        block=FEATURE_LOOKUP.loc[ids,BASE_NUM].astype(float).copy()
        if expanded:block['movement_retention_score']=movement(ids)
        return block
    pb=player_block(train_ids);pt=player_block(test_ids)
    m=pb.median().fillna(0);center=pb.fillna(m).mean();sd=pb.fillna(m).std(ddof=0).replace(0,1).fillna(1)
    def pb_transform(b):return np.column_stack([((b.fillna(m)-center)/sd).to_numpy(),b.isna().astype(float).to_numpy()])
    ptrain=pd.DataFrame(pb_transform(pb),index=train_ids);ptest=pd.DataFrame(pb_transform(pt),index=test_ids)
    weights=1/train.groupby('nfl_id').nfl_id.transform('size').to_numpy(float)
    q=train[PLAY_NUM].astype(float);r=test[PLAY_NUM].astype(float)
    qm=q.median().fillna(0);qc=np.average(q.fillna(qm),weights=weights,axis=0)
    qs=np.sqrt(np.average((q.fillna(qm).to_numpy()-qc)**2,weights=weights,axis=0));qs[qs==0]=1
    def play_num(b):return np.column_stack([(b.fillna(qm).to_numpy()-qc)/qs,b.isna().astype(float).to_numpy()])
    xt=[np.ones((len(train),1)),ptrain.loc[train.nfl_id].to_numpy(),play_num(q)]
    xv=[np.ones((len(test),1)),ptest.loc[test.nfl_id].to_numpy(),play_num(r)]
    for col in PLAY_CAT:
        levels=sorted(train[col].fillna('MISSING').astype(str).unique())
        # Drop one training reference category; unseen categories map to all-zero.
        for level in levels[1:]:
            xt.append(train[col].fillna('MISSING').astype(str).eq(level).to_numpy()[:,None].astype(float))
            xv.append(test[col].fillna('MISSING').astype(str).eq(level).to_numpy()[:,None].astype(float))
    X=np.column_stack(xt);V=np.column_stack(xv);y=train.separation_at_pass_forward.to_numpy(float)
    penalty=np.eye(X.shape[1])*alpha;penalty[0,0]=0.
    beta=np.linalg.solve(X.T@(weights[:,None]*X)+penalty,X.T@(weights*y))
    return np.maximum(V@beta,0.),dict(alpha=alpha,expanded=expanded,parameters=X.shape[1],movement_beta_standardized=float(beta[1+len(BASE_NUM)]) if expanded else None)

def player_prediction_table(test,pred):
    t=test[['nfl_id','draft_year','separation_at_pass_forward']].copy();t['prediction']=pred
    t=t.groupby(['nfl_id','draft_year'],as_index=False).agg(observed=('separation_at_pass_forward','mean'),predicted=('prediction','mean'),routes=('nfl_id','size'))
    t['absolute_error']=np.abs(t.observed-t.predicted)
    return t

def select_alpha_te(train,expanded):
    if train.nfl_id.nunique()<4:raise ValueError('Too few training players')
    records=[]
    for alpha in TE_SPEC['regularization_grid']:
        err=[]
        for pid in train.nfl_id.unique():
            a=train.loc[train.nfl_id.ne(pid)];b=train.loc[train.nfl_id.eq(pid)]
            pred,_=fit_predict_te(a,b,alpha,expanded)
            err.append(float(player_prediction_table(b,pred).absolute_error.iloc[0]))
        records.append(dict(alpha=alpha,LOPO_player_MAE=float(np.mean(err)),expanded=expanded))
    cv=pd.DataFrame(records)
    best=float(cv.loc[cv.LOPO_player_MAE.idxmin(),'alpha'])
    return best,cv

te_predictions=[];te_scores=[];te_tuning=[];te_models=[]
# Temporal development validation first; final fitting does not select on 2025.
for split,years,target_year in [('development',[2023],2024),('heldout_2025',[2023,2024],2025)]:
    train=route_data.loc[route_data.draft_year.isin(years)].copy()
    test=route_data.loc[route_data.draft_year.eq(target_year)].copy()
    if train.empty or test.empty:raise ValueError('Empty time split')
    chosen={}
    for expanded in [False,True]:
        alpha,cv=select_alpha_te(train,expanded);chosen[str(expanded)]=alpha
        cv['split']=split;te_tuning.append(cv)
    # Persist all choices before calling the final test prediction.
    (TE_OUT/('locked_'+split+'.json')).write_text(json.dumps(dict(train_years=years,test_year=target_year,alphas=chosen,spec=TE_SPEC),indent=2))
    for expanded in [False,True]:
        pred,model=fit_predict_te(train,test,chosen[str(expanded)],expanded)
        t=player_prediction_table(test,pred);t['model']='Combine + movement' if expanded else 'Combine baseline';t['split']=split
        te_predictions.append(t)
        score=dict(split=split,model=t.model.iloc[0],players=len(t),routes=int(t.routes.sum()),MAE_yards=float(t.absolute_error.mean()),RMSE_yards=float(np.sqrt(np.mean((t.observed-t.predicted)**2))),alpha=model['alpha'])
        te_scores.append(score);model.update(split=split,model=score['model']);te_models.append(model)
te_scores=pd.DataFrame(te_scores);te_predictions=pd.concat(te_predictions,ignore_index=True)
pd.concat(te_tuning).to_csv(TE_OUT/'training_only_alpha_CV.csv',index=False)
te_scores.to_csv(TE_OUT/'temporal_model_comparison.csv',index=False)
te_predictions.merge(te_players_all[['nfl_id','display_name']],on='nfl_id',validate='many_to_one').to_csv(TE_OUT/'heldout_player_predictions.csv',index=False)
(TE_OUT/'model_fit_metadata.json').write_text(json.dumps(te_models,indent=2))
uncertainty=[];rngte=np.random.default_rng(2040)
for split in ['development','heldout_2025']:
    t=te_predictions.loc[te_predictions.split.eq(split)]
    e=t.pivot(index='nfl_id',columns='model',values='absolute_error')
    delta=(e['Combine + movement']-e['Combine baseline']).to_numpy()
    boot=np.mean(rngte.choice(delta,size=(5000,len(delta)),replace=True),axis=1)
    uncertainty.append(dict(split=split,players=len(delta),delta_MAE_augmented_minus_baseline=float(delta.mean()),bootstrap_95_low=float(np.quantile(boot,.025)),bootstrap_95_high=float(np.quantile(boot,.975))))
te_uncertainty=pd.DataFrame(uncertainty);te_uncertainty.to_csv(TE_OUT/'paired_player_bootstrap.csv',index=False)
print(te_scores.to_string(index=False));print(te_uncertainty.to_string(index=False))
print('Negative delta favors added tracking. Bootstrap describes paired test-player error differences conditional on fitted models, not full model uncertainty.')


# 6E. График, энгийн дүгнэлт, нөөц.
import matplotlib.pyplot as plt
figte,axte=plt.subplots(1,3,figsize=(16,4.5))
for label,color in [('Combine baseline','gray'),('Combine + movement','teal')]:
    s=te_scores.loc[te_scores.model.eq(label)]
    x=np.arange(len(s));offset=-.17 if label=='Combine baseline' else .17
    axte[0].bar(x+offset,s.MAE_yards,width=.32,label=label,color=color)
axte[0].set_xticks([0,1],['2023 -> 2024','2023-24 -> 2025']);axte[0].set(title='Time-separated player-mean prediction',ylabel='MAE (yards)');axte[0].legend(fontsize=8)
last=te_predictions.loc[te_predictions.split.eq('heldout_2025')]
for label,mark,color in [('Combine baseline','x','gray'),('Combine + movement','o','teal')]:
    q=last.loc[last.model.eq(label)];axte[1].scatter(q.observed,q.predicted,label=label,marker=mark,c=color)
lo=float(min(last.observed.min(),last.predicted.min()))-.1;hi=float(max(last.observed.max(),last.predicted.max()))+.1
axte[1].plot([lo,hi],[lo,hi],'k--',linewidth=1);axte[1].set(title='2025: one point per player/model',xlabel='Observed mean separation (yards)',ylabel='Predicted mean separation (yards)');axte[1].legend(fontsize=8)
err=last.pivot(index='nfl_id',columns='model',values='absolute_error')
diff=err['Combine + movement']-err['Combine baseline']
axte[2].bar(np.arange(len(diff)),diff,color=np.where(diff<0,'teal','darkorange'));axte[2].axhline(0,color='k',linewidth=1)
axte[2].set(title='2025 paired prediction-error change',xlabel='Held-out player (arbitrary order)',ylabel='Movement - baseline absolute error (yards)')
plt.tight_layout();figte.savefig(TE_OUT/'TE_combine_to_separation.png',dpi=170,bbox_inches='tight');plt.show()
final_u=te_uncertainty.loc[te_uncertainty.split.eq('heldout_2025')].iloc[0]
status_te=dict(analysis='TE Combine tracking linked to rookie NFL separation at pass release',
 outcome='provided nearest-defender distance at pass release, yards',
 holdout_used_only_for_final_evaluation=True,
 test_players=int(final_u.players),delta_MAE=float(final_u.delta_MAE_augmented_minus_baseline),
 bootstrap_CI=[float(final_u.bootstrap_95_low),float(final_u.bootstrap_95_high)],
 evidence='improved prediction on this cohort with CI below zero' if final_u.bootstrap_95_high<0 else 'additional predictive value NOT established',
 limitations=['small player sample','observational association, not causation','largest-turn window heuristic not film-validated','only players with observed eligible rookie routes','conditional on recorded route/alignment/coverage/context','not an NFL ranking or a validated TE version of the WR Bayesian model'])
(TE_OUT/'conclusion.json').write_text(json.dumps(status_te,indent=2))
print(json.dumps(status_te,indent=2))
import shutil
te_archive=Path(shutil.make_archive(str(TE_OUT.parent/'TE_combine_to_separation_results'),'zip',root_dir=TE_OUT))
print('Result archive:',te_archive)


# 6F. Хэмжилтийн QA зураг, дүгнэлт, үр дүнг экспортлох.
from IPython.display import display,HTML
import base64
figqa,axsqa=plt.subplots(2,2,figsize=(11,8))
for col,d in enumerate(TE_SPEC['drills']):
    sample=te_attempts.loc[(te_attempts.draft_year==2023)&te_attempts.drill_name.eq(d)].sort_values(['nfl_id','event_id','attempt']).iloc[0]
    mask=np.ones(len(mt),dtype=bool)
    for k in TE_SEQ:mask &= mt[k].eq(sample[k]).to_numpy()
    gqa=mt.loc[mask].sort_values('timestamp');xy=gqa[['x','y']].to_numpy();tt=(gqa.timestamp-gqa.timestamp.iloc[0]).dt.total_seconds().to_numpy()
    j=int(np.flatnonzero(gqa.timestamp.eq(pd.Timestamp(sample.peak_time)).to_numpy())[0])
    axsqa[0,col].plot(xy[:,0],xy[:,1],'.-',color='steelblue',markersize=3)
    axsqa[0,col].scatter(*xy[0],c='green',label='Start');axsqa[0,col].scatter(*xy[j],c='darkorange',s=60,label='Selected turn window')
    axsqa[0,col].set(title=d,xlabel='x (yards)',ylabel='y (yards)');axsqa[0,col].axis('equal');axsqa[0,col].legend(fontsize=7)
    ss=np.r_[np.nan,np.linalg.norm(np.diff(xy,axis=0),axis=1)/np.diff(tt)]
    axsqa[1,col].plot(tt,ss,color='steelblue');axsqa[1,col].axvline(tt[j],color='darkorange')
    axsqa[1,col].axvspan(tt[j-6],tt[j-3],color='gray',alpha=.2,label='Approach samples')
    axsqa[1,col].axvspan(tt[j-3],tt[j+3],color='orange',alpha=.12,label='Heading-change window')
    axsqa[1,col].set(title='Deterministic first valid 2023 example; retention = %.2f'%sample.speed_retention,xlabel='Seconds from drill start',ylabel='Coordinate speed (yards/s)');axsqa[1,col].legend(fontsize=7)
plt.tight_layout();figqa.savefig(TE_OUT/'TE_turn_window_QA.png',dpi=150,bbox_inches='tight');plt.show()
dev_s=te_scores.loc[te_scores.split.eq('development')];fin_s=te_scores.loc[te_scores.split.eq('heldout_2025')]
te_note=(
'# TE Combine → NFL separation: анхны холбоосын шинжилгээ\n\n'
'## Юу хийсэн бэ?\n'
'WR дээр ашигласан координатаас өнцөг гаргах, дараалал ба хугацааг шалгах зарчмыг TE-ийн Hook болон In сорилд тохируулсан. WR V3-ийн коэффициент, тоглогчийн чансааг TE рүү шууд шилжүүлээгүй. Энэ бол тайлбарлахад хялбар нэг tracking хэмжүүрийн анхны шалгалт; TE дээр дахин сургасан Байесын V3 гэж үзэхгүй.\n\n'
'Хэмжүүр: хамгийн их хүчинтэй 0.6 секундын чиглэлийн өөрчлөлтийн цонхны үеийн хурд / эргэлтийн өмнөх хурд. Оролдлого бүрээс гаргаад сорил бүрд median, хоёр сорилд ижил жин өгсөн. Тухайн fold-ийн сургалтын тоглогчдоор л сорилын стандартчиллыг хийсэн. Тоормослох үед бодит зогсолтыг арилгахгүйн тулд 0 хурдыг зөвшөөрч, харин чиглэлийн хоёр шилжилт тус бүр 0.3 yard-аас их байх шаардлага тавьсан. Window нь film-аар баталсан route break биш.\n\n'
'## NFL хэмжүүр\n'
'player_play.csv дахь separation_at_pass_forward: бөмбөг урагш шидэгдэх мөчид хамгийн ойрын хамгаалагч хүртэлх зай, yard. Энэ нь бөмбөг барих мөчийн зай биш. Target болсон эсэхээс үл хамааран шидэлттэй, хүчинтэй шошготой route-уудыг ашигласан. Зөвхөн rookie regular season; нэг хүнд хамгийн багадаа 20 хүчинтэй route, хоёр Combine сорил хэрэгтэй.\n\n'
'Combine-д TE гэсэн 42 хүнээс NFL position нь FB болсон нэг хүнийг үндсэн шинжилгээнээс хасч, 41 TE-ийг аудитласан. 36 нь хоёр сорилын хүчинтэй хэмжүүртэй. Эцсийн сургалт/шалгалтын хамрах хүрээг player_eligibility.csv-д мөр бүрээр хадгалсан.\n\n'
'## Харьцуулалт\n'
'Суурь загвар: forty, ten_yd_split, three_cone, short_shuttle, өндөр, жин + route, lineup position, coverage болон down/yardage/cushion/pass depth нөхцөл. Өргөтгөсөн загварт tracking-ийн speed retention хэмжүүрийг нэмсэн. Тоглогч бүр нийт ижил жинтэй; олон route нэг хүнийг олон бие даасан хүн мэт тооцоогүй. Хоёр загварт regularization strength-ийг зөвхөн сургалтын тоглогчийг бүхэлд нь хассан CV-ээр сонгосон.\n\n'
'2023 оны 5 хүнээр сургаж, 2024 оны 10 хүн дээр хөгжүүлэлтийн шалгалт хийсэн. Дараа нь 2023–2024 оны 15 хүнээр сургаж, 2025 оны 7 хүнийг эцсийн шалгалтад хэрэглэсэн. 2025 оныг alpha сонголтод ашиглаагүй. Одоо энэ cohort-ийн үр дүнг харсан тул дараагийн хувилбаруудын хувьд шинэ untouched test гэж нэрлэхгүй.\n\n'
+te_scores.to_string(index=False)+'\n\n'+te_uncertainty.to_string(index=False)+'\n\n'
'Үр дүн: 2024 оны point estimate сайжирсан боловч player-bootstrap интервал нь 0-ийг агуулсан. Development-ийн хоёр загвар өөр alpha сонгосон учраас түүний ялгааг tracking feature-ийн дан нөлөө гэж тайлбарлахгүй.\n\n'
'2025 онд хөдөлгөөний хэмжүүр нэмэхэд ердийн Combine загвараас илүү сайн болсон нь батлагдсангүй. Энэ нь хэмжүүр хэрэггүйг баталсан дүгнэлт биш; 7 хүний түүвэр, чанарын шүүлт, ажиглагдсан тоглох боломж, model shrinkage-д нөхцөлдсөн анхны үр дүн юм. Bootstrap нь fitted model-уудын test-player алдааны ялгааны интервал бөгөөд бүтэн model-fitting uncertainty биш.\n\n'
'Загвар нь ирээдүйн route/alignment/context-ийг мэдэхгүй pre-draft бие даасан forecast биш: ажиглагдсан NFL нөхцөлд хэмжүүр нэмэлт тайлбар өгч байгаа эсэхийг шалгасан. Шалтгаан-үр дагавар, blocking success, тоглогчийн чансаа гэж дүгнэхгүй.\n\n'
'## Дараагийн судалгаа\n'
'Хэмжүүрийн turn window-ийг дүрс бичлэгтэй шалгах, илүү олон cohort/TE нэмэх, өмнө тогтоосон sensitivity протоколоор өөр цонх ба coverage босгыг судлах. 2025 дээр хамгийн сайн харагдах тохиргоог хайж түүнийгээ шинэ баталгаа гэж тайлбарлахгүй.\n\n'
'Албан ёсны өгөгдлийн толь: https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data\n'
)
(TE_OUT/'README_MN.md').write_text(te_note)
# Repackage after QA and readable notes are added.
te_archive=Path(shutil.make_archive(str(TE_OUT.parent/'TE_combine_to_separation_results'),'zip',root_dir=TE_OUT))
print('TE analysis complete. Final holdout:',status_te['evidence'])
for image_path,title in [(TE_OUT/'TE_combine_to_separation.png','TE separation model comparison'),(TE_OUT/'TE_turn_window_QA.png','TE turn window QA')]:
    encoded=base64.b64encode(image_path.read_bytes()).decode('ascii')
    display(HTML(f'<img alt="{title}" src="data:image/png;base64,{encoded}" style="max-width:100%">'))
encoded=base64.b64encode(te_archive.read_bytes()).decode('ascii')
display(HTML(f'<a download="{te_archive.name}" href="data:application/zip;base64,{encoded}">Download TE analysis results</a>'))
print('ZIP size MiB:',round(te_archive.stat().st_size/1024**2,2))


# 6G. Багийн TE асуултад холбосон энгийн дүгнэлт
display(HTML("""<h3>TE Combine хөдөлгөөн → NFL separation</h3><p>Hook, In сорилын эргэлтийн үед хурдаа хадгалсан харьцааг NFL-ийн эхний улиралд бөмбөг шидэх мөчийн хамгаалагч хүртэлх зайтай холбов.</p><p>2023–2024 оны 15 тоглогчоор сургаж, 2025 оны 7 тоглогчийн 1,506 route дээр шалгасан. Route, alignment, coverage болон тоглолтын нөхцөлийг тооцсон.</p><p><b>Дүгнэлт: энэ хэмжүүр ердийн Combine мэдээллээс нэмэлт таамаглах ашиг өгсөн нь батлагдсангүй.</b> Энэ нь бүх хөдөлгөөний хэмжүүр хэрэггүй гэсэн баталгаа биш; түүвэр жижиг, хэмжилтийн window нь film-аар батлагдаагүй.</p>"""))
report_te=te_scores.loc[te_scores.split.eq("heldout_2025"),["model","MAE_yards","players","routes"]].copy()
report_te.columns=["Загвар","Дундаж зөрүү (yard)","Тоглогч","Route"]
display(report_te.round(4))
print("2025 ялгаа (movement - baseline): +0.00133 yard; paired-player bootstrap 95%: [-0.00247, 0.00551].")
print("Энэ TE шинжилгээ нь WR V3-ийн коэффициентийг хуулсан биш; өнцөг/хурд хэмжих зарчмыг TE-д тохируулсан эхний холбоосын шалгалт юм.")
encoded_te=base64.b64encode((TE_OUT/"TE_combine_to_separation.png").read_bytes()).decode("ascii")
display(HTML(f'<img alt="TE final results" src="data:image/png;base64,{encoded_te}" style="max-width:100%">'))
