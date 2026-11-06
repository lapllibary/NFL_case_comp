"""Inference for saved TE development models. Does not read outcomes."""
import json
import numpy as np
import pandas as pd

def predict_saved_model(model_path,route_frame,player_features):
    with open(model_path) as f:m=json.load(f)
    t=m['transforms'];cols=t['player_columns'];F=player_features.set_index('nfl_id')
    if F.index.duplicated().any():raise ValueError('Duplicate player features')
    if not set(route_frame.nfl_id).issubset(F.index):raise ValueError('Player features missing')
    p=F.loc[route_frame.nfl_id,cols].astype(float)
    def series(k):return pd.Series(t[k],dtype=float)
    P=np.column_stack([((p.fillna(series('player_median'))-series('player_center'))/series('player_sd')).to_numpy(),p.isna().to_numpy(float)])
    q=route_frame[t['context_numeric']].astype(float)
    Q=np.column_stack([((q.fillna(series('context_median'))-series('context_center'))/series('context_sd')).to_numpy(),q.isna().to_numpy(float)])
    blocks=[np.ones((len(q),1)),P,Q];unseen={}
    for col,levels in t['category_levels'].items():
        values=route_frame[col].fillna('MISSING').astype(str);unseen[col]=int((~values.isin(levels)).sum())
        for level in levels[1:]:blocks.append(values.eq(level).to_numpy(float)[:,None])
    X=np.column_stack(blocks);beta=np.array(m['coefficients'])
    if X.shape[1]!=len(beta):raise ValueError('Model/feature dimension mismatch')
    return np.maximum(X@beta,0.),{'unseen_category_rows':unseen,'development_only':True}
