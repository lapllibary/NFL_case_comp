"""Research extension: operational TE proxies, pressure catch rates, draft-team outcomes,
pre-draft historical references and provisional physical-similarity screens.
No verified archetype, causal effect or validated 32-team fit score is claimed.
External pressure data attribution: FTN Data via nflverse, CC-BY-SA 4.0.
"""
from pathlib import Path
import os,json,math,hashlib
import numpy as np
import pandas as pd
D=Path(os.environ.get('NFL_DATA_DIR','/content/nfl-big-data-bowl-2027'))
E=Path(os.environ.get('TE_EXTERNAL_DIR','/content/te_external'))
O=Path(os.environ.get('TE_SCOUT_OUT','/content/te_scouting_extension'));O.mkdir(parents=True,exist_ok=True)
SPEC={'version':'TE scouting operational research v1','threshold_reference_cohorts':[2023,2024],
 'archetypes':'multi-label provisional proxies; all human_validated and archetype_validated false',
 'off_script':'scripted drill proxy only: route approach and post speed >= reference 75th percentiles; cannot measure off-script decisions',
 'inline':'mass >= reference 75th percentile and bench >= reference median; cannot measure leverage/block integrity',
 'hard_brake':'at least 2 valid selected route attempts; at least one turn >=30deg with derived smoothed acceleration <=-5 m/s^2',
 'apex':'wingspan and vertical >= reference 75th percentiles; cannot measure actual catch radius',
 'pressure':'FTN via nflverse was_pressure; never substitute blocker pressure_allowed or qb_hit',
 'H1_window':'rookie regular season, original drafting franchise; matched non-nullified target plays; unknown pressure excluded and counted',
 'pair_minimum':'at least 10 pressured and 10 clean-pocket targets for primary descriptive pair-rate comparison',
 'catch':'Kaggle target true and pass_result=C versus C/I/IN eligible targets',
 'champion_reference':'2019-2022 Super Bowl winners, REG TE production; fixed before earliest 2023 cohort',
 'fit':'prior-season roster TE target-weighted height/weight; descriptive similarity shortlist only, not validated fit',
 'fit_candidates':'3 per team per draft cohort, no post-draft performance used in similarity; historical outcomes displayed separately',
 'selection':'same existing cohorts already explored; new analysis is retrospective exploratory, not new heldout validation'}
(O/'protocol.json').write_text(json.dumps(SPEC,indent=2))
ABBR={'LA':'LAR','LVR':'LV','OAK':'LV','GNB':'GB','KAN':'KC','NWE':'NE','NOR':'NO','SFO':'SF','TAM':'TB','JAC':'JAX','SD':'LAC','STL':'LAR'}
def team(s):return s.replace(ABBR) if isinstance(s,pd.Series) else ABBR.get(s,s)
def save(df,name):df.to_csv(O/(name+'.csv'),index=False)
def flag(v):return v.astype('string').str.lower().map({'true':True,'false':False,'1':True,'0':False})
players=pd.read_csv(D/'players.csv');combine=pd.read_csv(D/'combine_results.csv');tracking=pd.read_csv(D/'combine_tracking.csv',dtype={'event_id':'string'})
base=combine[combine.combine_position.eq('TE')].merge(players,on=['nfl_id','draft_year'],validate='one_to_one');save(base[['nfl_id','display_name','draft_year','nfl_position']],'TE_position_audit')
base=base[base.nfl_position.eq('TE')].copy();assert len(base)==41
# Stable numeric NFL id -> GSIS link from year-specific external rosters.
rosters=[]
for year in range(2019,2026):
 r=pd.read_csv(E/f'roster_{year}.csv');r['team']=team(r.team);rosters.append(r)
R=pd.concat(rosters,ignore_index=True)
links=R[R.gsis_it_id.notna()][['gsis_it_id','gsis_id']].drop_duplicates();links.gsis_it_id=links.gsis_it_id.astype(int)
links=links[links.gsis_it_id.isin(base.nfl_id)&links.gsis_id.notna()]
if links.gsis_it_id.duplicated().any():raise ValueError('Ambiguous NFL->GSIS link')
base=base.merge(links.rename(columns={'gsis_it_id':'nfl_id'}),on='nfl_id',how='left',validate='one_to_one')
draft=pd.read_csv(E/'draft_picks.csv');draft['draft_team']=team(draft.team)
C=draft[['season','pick','draft_team','gsis_id','pfr_player_name']].rename(columns={'season':'draft_year','pick':'draft_overall_pick','gsis_id':'draft_gsis_id'})
assert not C.duplicated(['draft_year','draft_overall_pick']).any()
base=base.merge(C,on=['draft_year','draft_overall_pick'],how='left',validate='many_to_one')
base['draft_status']=np.where(base.draft_overall_pick.isna(),'undrafted_original_signing_team_unresolved',np.where(base.draft_team.notna(),'drafted_verified','draft_crosswalk_unresolved'))
check=base.draft_status.eq('drafted_verified');assert base.loc[check,'gsis_id'].notna().all();assert base.loc[check,'gsis_id'].eq(base.loc[check,'draft_gsis_id']).all()
save(base[['nfl_id','display_name','draft_year','draft_overall_pick','gsis_id','draft_team','pfr_player_name','draft_status']],'draft_team_crosswalk')
# Tracking-derived first sustained bend. Coordinates yards, time seconds, derivatives converted to m/s².
T=tracking[tracking.nfl_id.isin(base.nfl_id)&tracking.entity_type.eq('PLAYER')&tracking.drill_name.isin(['HOOK_ROUTE_RIGHT','IN_ROUTE_LEFT'])].copy()
T['timestamp']=pd.to_datetime(T.time,utc=True);KEY=['nfl_id','draft_year','event_id','drill_type','drill_name','attempt'];assert not T.duplicated(KEY+['timestamp']).any()
rows=[]
for key,g in T.groupby(KEY,sort=True):
 g=g.sort_values('timestamp');xy=g[['x','y']].to_numpy(float);ns=g.timestamp.astype('int64').to_numpy();dt=np.diff(ns)/1e9;tt=(ns-ns[0])/1e9;n=len(g)
 s=np.r_[np.nan,np.linalg.norm(np.diff(xy,axis=0),axis=1)/dt];good=np.r_[False,(abs(dt-.1)<=.015)&(s[1:]>=0)&(s[1:]<=15)]
 path=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];progress=path/path[-1] if path[-1]>0 else path
 smooth=pd.Series(s).rolling(3,center=True,min_periods=3).mean().to_numpy();acc=np.r_[np.nan,np.diff(smooth)/dt]*.9144
 candidates=[]
 for j in range(8,n-8):
  if not .15<=progress[j]<=.85 or not good[j-7:j+9].all():continue
  v=xy[j]-xy[j-3];w=xy[j+3]-xy[j]
  if min(np.linalg.norm(v),np.linalg.norm(w))<.3:continue
  angle=np.degrees(np.arctan2(v[0]*w[1]-v[1]*w[0],v@w));approach=float(np.median(s[j-6:j-2]))
  if approach<2 or abs(angle)<15:continue
  candidates.append(dict(j=j,angle=angle,approach=approach,through=float(min(np.mean(s[k-1:k+2]) for k in range(j-2,j+3))),post=float(np.median(s[j+3:j+7])),brake_m_s2=float(np.nanmin(acc[j-4:j+5]))))
 episodes=[]
 for c in candidates:
  if not episodes or c['j']!=episodes[-1][-1]['j']+1 or np.sign(c['angle'])!=np.sign(episodes[-1][-1]['angle']):episodes.append([])
  episodes[-1].append(c)
 eligible=[ep for ep in episodes if len(ep)>=3];r=dict(zip(KEY,key));r['status']='valid' if eligible else 'no_sustained_episode'
 if eligible:
  c=max(eligible[0],key=lambda x:abs(x['angle']));r.update(c);r['seconds']=float(tt[c['j']]);r['retention']=c['through']/c['approach'];r['brake_turn_evidence']=abs(c['angle'])>=30 and c['brake_m_s2']<=-5
 rows.append(r)
attempts=pd.DataFrame(rows);save(attempts,'archetype_attempt_features')
v=attempts[attempts.status.eq('valid')]
f=v.groupby('nfl_id').agg(route_attempts=('nfl_id','size'),route_drills=('drill_name','nunique'),approach_speed=('approach','median'),post_speed=('post','median'),speed_retention=('retention','median'),brake_turn_attempts=('brake_turn_evidence','sum'),min_brake_m_s2=('brake_m_s2','min')).reset_index()
base=base.merge(f,on='nfl_id',how='left',validate='one_to_one');ref=base[base.draft_year.isin([2023,2024])]
thresholds={c:float(ref[c].quantile(q)) for c,q in [('approach_speed',.75),('post_speed',.75),('combine_weight',.75),('bench_reps',.5),('wing_span',.75),('vertical',.75)]}
(O/'archetype_thresholds.json').write_text(json.dumps(thresholds,indent=2))
labelrows=[]
for _,r in base.iterrows():
 conditions=[('Off-Script Seam Buster','scripted_route_speed_proxy',['approach_speed','post_speed','route_attempts'],r.approach_speed>=thresholds['approach_speed'] and r.post_speed>=thresholds['post_speed'] and r.route_attempts>=2,'Scripted drill cannot establish off-script decision making.'),
 ('Inline Leverage Anchor','size_strength_proxy',['combine_weight','bench_reps'],r.combine_weight>=thresholds['combine_weight'] and r.bench_reps>=thresholds['bench_reps'],'Body mass and bench do not establish leverage, force or block integrity.'),
 ('Hard-Brake Separator','braking_turn_proxy',['route_attempts','brake_turn_attempts'],r.route_attempts>=2 and r.brake_turn_attempts>=1,'Threshold and movement phase are operational; NFL separation translation remains unproven.'),
 ('Apex Catch Radius','reach_jump_proxy',['wing_span','vertical'],r.wing_span>=thresholds['wing_span'] and r.vertical>=thresholds['vertical'],'Wingspan and jump are proxies, not measured catch radius.')]
 for archetype,proxy,needed,passed,limit in conditions:
  status='insufficient_measurements' if any(pd.isna(r[c]) for c in needed) else ('provisional_candidate' if passed else 'below_proxy_rule')
  labelrows.append(dict(nfl_id=r.nfl_id,display_name=r.display_name,draft_year=r.draft_year,archetype=archetype,proxy=proxy,status=status,human_validated=False,archetype_validated=False,limit=limit))
labels=pd.DataFrame(labelrows);save(labels,'TE_archetype_proxies');save(base,'TE_feature_table');save(labels.groupby(['archetype','status']).size().rename('players').reset_index(),'archetype_coverage')
readiness=[]
for archetype,g in labels[labels.draft_year.isin([2023,2024])].groupby('archetype'):
 observed=g.status.ne('insufficient_measurements');rate=float(g.loc[observed,'status'].eq('provisional_candidate').mean()) if observed.any() else np.nan
 readiness.append(dict(archetype=archetype,reference_players=len(g),measured_players=int(observed.sum()),reference_candidate_rate=rate,classification_ready=False,decision='reject_as_discriminative_classifier' if rate>=.9 else 'proxy_only_requires_construct_and_film_validation'))
save(pd.DataFrame(readiness),'archetype_readiness')
# Match complete passing data and true QB-pressure labels by game/play; preserve missingness.
P=[];U=[]
for year in [2023,2024,2025]:
 cols=['old_game_id','game_id','play_id','season','season_type','posteam','passer_player_id','passer_player_name','receiver_player_id','complete_pass','pass_attempt','qb_spike','qb_hit','first_down_pass']
 p=pd.read_csv(E/f'play_by_play_{year}.csv.gz',usecols=cols,low_memory=False);p['old_game_id']=pd.to_numeric(p.old_game_id,errors='coerce').astype('Int64');p['posteam']=team(p.posteam);P.append(p)
 u=pd.read_csv(E/f'pbp_participation_{year}.csv',usecols=['old_game_id','nflverse_game_id','play_id','was_pressure']);u['old_game_id']=pd.to_numeric(u.old_game_id,errors='coerce').astype('Int64');u['was_pressure']=flag(u.was_pressure);U.append(u)
pbp=pd.concat(P,ignore_index=True);pressure=pd.concat(U,ignore_index=True)
assert not pbp.duplicated(['old_game_id','play_id']).any();assert not pressure.duplicated(['old_game_id','play_id']).any()
pbp=pbp.merge(pressure[['old_game_id','play_id','was_pressure']],on=['old_game_id','play_id'],how='left',validate='one_to_one')
q=pd.read_csv(D/'player_play.csv',low_memory=False);q=q[q.nfl_id.isin(base.nfl_id)].copy();assert not q.duplicated(['game_id','play_id','nfl_id']).any()
games=pd.read_csv(D/'games.csv');q=q.merge(games[['game_id','season','season_type']],on='game_id',validate='many_to_one')
q=q.merge(base[['nfl_id','display_name','draft_year','draft_team','draft_status','gsis_id']],on='nfl_id',validate='many_to_one');q['team_abbr']=team(q.team_abbr)
lookup=pbp.drop(columns=['season','season_type']).rename(columns={'old_game_id':'game_id','game_id':'nflverse_game_id'})
q=q.merge(lookup,on=['game_id','play_id'],how='left',validate='many_to_one',indicator='pbp_match')
q['original_franchise']=q.draft_status.eq('drafted_verified')&q.team_abbr.eq(q.draft_team)
q['rookie']=q.season.eq(q.draft_year)&q.season_type.eq('REG')
q['eligible_target']=flag(q.target).eq(True)&q.pass_result.isin(['C','I','IN'])&q.play_nullified_by_penalty.eq('N')
q['reception']=q.pass_result.eq('C');q['receiver_conflict']=q.receiver_player_id.notna()&q.receiver_player_id.ne(q.gsis_id)
all_targets=q[q.rookie&q.eligible_target].copy();
assert all_targets.pbp_match.eq('both').all()
assert all_targets.team_abbr.eq(all_targets.posteam).all()
assert all_targets.reception.eq(all_targets.complete_pass.eq(1)).all()
save(all_targets[['nfl_id','display_name','draft_year','game_id','play_id','team_abbr','draft_team','passer_player_id','passer_player_name','pass_result','was_pressure','original_franchise','receiver_conflict','pbp_match']],'H1_target_link_audit')
primary=all_targets[all_targets.original_franchise&~all_targets.receiver_conflict&all_targets.pbp_match.eq('both')&all_targets.passer_player_id.notna()].copy()
def wilson(k,n):
 if not n:return (np.nan,np.nan)
 z=1.95996398454;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return mid-rad,mid+rad
pairs=[]
for key,g in primary.groupby(['nfl_id','display_name','draft_year','draft_team','passer_player_id','passer_player_name'],dropna=False):
 r=dict(zip(['nfl_id','display_name','draft_year','draft_team','qb_gsis_id','qb_name'],key));r['unknown_pressure_targets']=int(g.was_pressure.isna().sum())
 for val,name in [(True,'pressured'),(False,'clean')]:
  s=g[g.was_pressure.eq(val)];n=len(s);k=int(s.reception.sum());lo,hi=wilson(k,n);r.update({name+'_targets':n,name+'_receptions':k,name+'_catch_rate':k/n if n else np.nan,name+'_CI_low':lo,name+'_CI_high':hi})
 r['pressure_resilience']=r['pressured_catch_rate']-r['clean_catch_rate'];r['primary_pair_eligible']=r['pressured_targets']>=10 and r['clean_targets']>=10;pairs.append(r)
pairs=pd.DataFrame(pairs);save(pairs,'QB_TE_pressure_catch_rates')
coverage=all_targets.groupby('draft_year').agg(targets=('nfl_id','size'),players=('nfl_id','nunique'),pressure_known=('was_pressure','count'),receiver_conflicts=('receiver_conflict','sum')).reset_index();save(coverage,'H1_coverage')
# First-franchise outcomes by season: no trades or cumulative career conflation.
first=q[q.original_franchise&q.season_type.eq('REG')&q.play_nullified_by_penalty.eq('N')].copy();outcomes=[]
for key,g in first.groupby(['nfl_id','display_name','draft_year','draft_team','season']):
 tar=g[g.eligible_target];rt=g[g.route_ran.notna()&g.pass_result.isin(['C','I','IN'])];r=dict(zip(['nfl_id','display_name','draft_year','draft_team','season'],key));r.update(years_since_draft=int(key[-1]-key[2]),charted_games=g.game_id.nunique(),charted_player_plays=len(g),charted_routes=len(rt),targets=len(tar),receptions=int(tar.reception.sum()),receiving_yards=float(tar.rec_yards.sum()),yards_after_catch=float(tar.yards_after_catch.sum()),mean_separation=float(rt.separation_at_pass_forward.mean()),separation_routes=int(rt.separation_at_pass_forward.notna().sum()),catch_rate=float(tar.reception.mean()) if len(tar) else np.nan,targets_per_charted_route=len(tar)/len(rt) if len(rt) else np.nan);outcomes.append(r)
outcomes=pd.DataFrame(outcomes);save(outcomes,'first_drafting_team_season_outcomes')
ledger=[]
for _,r in base[base.draft_status.eq('drafted_verified')].iterrows():
 for year in range(int(r.draft_year),2026):
  present=((outcomes.nfl_id==r.nfl_id)&(outcomes.season==year)).any()
  ledger.append(dict(nfl_id=r.nfl_id,display_name=r.display_name,draft_year=r.draft_year,draft_team=r.draft_team,season=year,observed_original_team_rows=bool(present),status='observed' if present else 'no_charted_original_team_rows_not_zero_performance'))
save(pd.DataFrame(ledger),'first_team_observation_ledger')
# Source-derived champions before 2023, and TE room production from 2019-2022 REG.
sched=pd.read_csv(E/'games.csv');sb=sched[sched.game_type.eq('SB')&sched.season.between(2019,2022)].copy();sb['champion']=team(pd.Series(np.where(sb.home_score>sb.away_score,sb.home_team,sb.away_team),index=sb.index));save(sb[['season','game_id','champion','home_team','away_team','home_score','away_score']],'Super_Bowl_reference_seasons')
assert len(sb)==4 and sb.season.max()<2023
stats=[]
for y in range(2019,2025):
 x=pd.read_csv(E/f'player_stats_{y}.csv');x=x[x.season_type.eq('REG')&x.position.eq('TE')].copy();x['team']=team(x.recent_team);stats.append(x)
S=pd.concat(stats,ignore_index=True)
prod=S.groupby(['season','team','player_id','player_display_name'],as_index=False).agg(games_with_stats=('week','nunique'),targets=('targets','sum'),receptions=('receptions','sum'),receiving_yards=('receiving_yards','sum'),receiving_tds=('receiving_tds','sum'))
phys=R[R.position.eq('TE')][['season','team','gsis_id','height','weight']].drop_duplicates()
if phys.duplicated(['season','team','gsis_id']).any():raise ValueError('Conflicting roster body measurements')
prod=prod.merge(phys,left_on=['season','team','player_id'],right_on=['season','team','gsis_id'],how='left',validate='one_to_one')
champ=prod.merge(sb[['season','champion']],left_on=['season','team'],right_on=['season','champion'],validate='many_to_one');save(champ,'Super_Bowl_TE_room_reference')
# Team profiles use the previous season only. No future NFL prospect outcome enters screening.
profiles=[]
for (y,tm),g in prod[prod.season.between(2022,2024)].groupby(['season','team']):
 g=g[g.targets.gt(0)&g.height.notna()&g.weight.notna()]
 if not len(g):continue
 profiles.append(dict(draft_year=y+1,team=tm,reference_season=y,TEs=len(g),TE_targets=int(g.targets.sum()),profile_height=float(np.average(g.height,weights=g.targets)),profile_weight=float(np.average(g.weight,weights=g.targets))))
profiles=pd.DataFrame(profiles);assert profiles.reference_season.lt(profiles.draft_year).all();save(profiles,'pre_draft_team_TE_profiles')
std_h=max(float(R[R.position.eq('TE')&R.season.between(2019,2022)].height.std()),1.);std_w=max(float(R[R.position.eq('TE')&R.season.between(2019,2022)].weight.std()),10.)
known=champ[champ.targets.gt(0)&champ.height.notna()&champ.weight.notna()];ch=float(np.average(known.height,weights=known.targets));cw=float(np.average(known.weight,weights=known.targets))
label_strings=labels[labels.status.eq('provisional_candidate')].groupby('nfl_id').archetype.agg('; '.join)
short=[]
for _,pr in profiles.iterrows():
 candidates=base[base.draft_year.eq(pr.draft_year)&base.combine_height.notna()&base.combine_weight.notna()].copy()
 candidates['physical_distance']=np.hypot((candidates.combine_height-pr.profile_height)/std_h,(candidates.combine_weight-pr.profile_weight)/std_w)
 for rank,(_,c) in enumerate(candidates.sort_values(['physical_distance','nfl_id']).head(3).iterrows(),1):
  short.append(dict(draft_year=int(pr.draft_year),team=pr.team,similarity_order=rank,nfl_id=int(c.nfl_id),prospect=c.display_name,profile_season=int(pr.reference_season),height=c.combine_height,weight=c.combine_weight,physical_distance=float(c.physical_distance),champion_reference_distance=float(np.hypot((c.combine_height-ch)/std_h,(c.combine_weight-cw)/std_w)),archetype_proxy_candidates=label_strings.get(c.nfl_id,''),actual_draft_team=c.draft_team,validated_fit=False,evidence_status='INSUFFICIENT_FOR_SCOUTING_RECOMMENDATION',use='provisional two-trait similarity screen for human review; not a validated overall team-fit ranking'))
short=pd.DataFrame(short);save(short,'all_32_teams_provisional_shortlists')
coverage32=profiles.groupby('draft_year').team.nunique();assert coverage32.eq(32).all();assert len(short)==32*3*3;assert not short.validated_fit.any()
status=dict(TE_players=len(base),verified_drafted_players=int(check.sum()),undrafted_or_unresolved=int((~check).sum()),GSIS_matched_players=int(base.gsis_id.notna().sum()),archetype_status=labels.groupby('status').size().to_dict(),H1_pairs=len(pairs),H1_primary_pairs=int(pairs.primary_pair_eligible.sum()),H1_pressure_known_targets=int(primary.was_pressure.notna().sum()),H1_unknown_pressure_targets=int(primary.was_pressure.isna().sum()),H1_primary_players=int(primary.nfl_id.nunique()),first_team_player_seasons=len(outcomes),champion_seasons=len(sb),champion_TE_player_seasons=len(champ),teams_per_cohort={str(k):int(v) for k,v in coverage32.items()},shortlist_rows=len(short),archetypes_validated=False,fit_validated=False,submission_uploaded=False)
(O/'status.json').write_text(json.dumps(status,indent=2,default=int))
(O/'validation_checks.txt').write_text('PASS: 41 reconciled TEs; unique roster NFL->GSIS mapping; draft pick/year/GSIS agreement; unique game/play/player and external pressure keys; first-franchise filter; no future seasons in team or champion profiles; fixed 2023-24 proxy thresholds; unknown pressure retained separately; 32 teams per cohort and 288 screen rows; no validated archetype or fit claim.\n')
print(json.dumps(status,indent=2,default=int));print(labels.groupby(['archetype','status']).size().to_string());print(pairs[pairs.primary_pair_eligible][['display_name','draft_year','qb_name','pressured_targets','pressured_catch_rate','clean_targets','clean_catch_rate']].to_string(index=False))
