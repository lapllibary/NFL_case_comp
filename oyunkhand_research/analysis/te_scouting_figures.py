import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
P=Path(os.environ['TE_SCOUT_OUT'])
q=pd.read_csv(P/'QB_TE_pressure_catch_rates.csv');q=q[q.primary_pair_eligible].reset_index(drop=True)
fig,ax=plt.subplots(figsize=(10,6));y=np.arange(len(q))
for offset,prefix,color,label in [(-.13,'pressured','#d45d43','Pressured'),(.13,'clean','#286a9b','Clean')]:
 rate=q[prefix+'_catch_rate'];lo=q[prefix+'_CI_low'];hi=q[prefix+'_CI_high']
 ax.errorbar(rate,y+offset,xerr=[rate-lo,hi-rate],fmt='o',capsize=4,color=color,label=label)
 for i,r in q.iterrows():ax.text(1.02,i+offset,f"{r[prefix+'_receptions']}/{r[prefix+'_targets']}",va='center',fontsize=9)
ax.set_yticks(y,[f"{r.display_name} / {r.qb_name} ({r.draft_year})" for _,r in q.iterrows()]);ax.set_xlim(0,1);ax.set_xlabel('Catch rate with Wilson 95% interval');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(loc='lower left');ax.set_title('Five QB–TE pairs: descriptive rates, small samples');fig.tight_layout();fig.savefig(P/'pressure_catch_rates.png',dpi=170,bbox_inches='tight');plt.close(fig)
c=pd.read_csv(P/'archetype_coverage.csv').pivot(index='archetype',columns='status',values='players').fillna(0)
fig,ax=plt.subplots(figsize=(10,5));left=np.zeros(len(c))
for key,color in [('provisional_candidate','#337ea5'),('below_proxy_rule','#b8c5ce'),('insufficient_measurements','#e5b45a')]:
 ax.barh(c.index,c[key],left=left,label=key.replace('_',' '),color=color);left+=c[key].to_numpy()
ax.set_xlabel('Players (41 total per proxy)');ax.set_title('Archetype proxies — none validated; Hard-Brake rule rejected');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3);fig.tight_layout();fig.savefig(P/'archetype_coverage.png',dpi=170,bbox_inches='tight');plt.close(fig)
display(__import__('IPython').display.Image(filename=str(P/'pressure_catch_rates.png')))
display(__import__('IPython').display.Image(filename=str(P/'archetype_coverage.png')))
