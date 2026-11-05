"""Do mot hanh vi moi lam Top-20 doi bao nhieu, khong ghi vao Neo4j. So o muc 4.5.6 cua de an. Can Neo4j va bien moi truong cua bt-shop."""
import sys,time; sys.path.insert(0,'.')
import numpy as np,pandas as pd
from pathlib import Path
from src.serving.service import DemoService
from src.serving.store import GraphStore
svc=DemoService.from_artifacts('active','bt_dkgrec_l05',Path('experiments/serving/active_bt_dkgrec_l05_2020_20261104-141829/e0_best.npy'))
st=GraphStore()
cat=pd.read_parquet('data/app/catalog.parquet').set_index('item_id')
ids=[i for i in cat.index if i in svc._item_row]
pop=cat.loc[ids].assign(p=lambda d:d.n_view+3*d.n_cart+5*d.n_txn).sort_values('p',ascending=False)
# candidates: top 1500 by activity, plus top 15 of each department
cand=list(pop.index[:40])+[i for _,g in pop.groupby('department') for i in g.index[:8]]
cand=list(dict.fromkeys(cand)); print(len(cand))
for v in (962057,481437,437888):
    h=st.history(v)
    base=svc.recommend(v,h,20); b=set(base.item_id.tolist()); bl=base.item_id.tolist()
    t=svc.next_timestamp(h); seen=set(h.item_id.tolist())
    out=[]
    t0=time.time()
    for it in cand:
        if it in seen: continue
        for beh in ('view','addtocart'):
            row=pd.DataFrame([{'item_id':it,'behavior':beh,'timestamp':t}])
            hh=pd.concat([h[['item_id','behavior','timestamp']],row],ignore_index=True)
            a=svc.recommend(v,hh,20).item_id.tolist()
            out.append((it,beh,len(set(a)-b-{it}),sum(1 for x,y in zip(bl,a) if x!=y)))
    o=pd.DataFrame(out,columns=['item','beh','new','moved'])
    print('\nV',v,'events',len(h),'time',round(time.time()-t0,1),'base depts',cat.loc[[i for i in bl if i in cat.index]].department.value_counts().to_dict())
    for beh,g in o.groupby('beh'):
        print(beh,'median new',g.new.median(),'mean',round(g.new.mean(),2),'share>=5',round((g.new>=5).mean(),3),'max',g.new.max(),'moved med',g.moved.median())
    tp=o[o.beh=='addtocart'].sort_values('new',ascending=False).head(6)
    for r in tp.itertuples():
        c=cat.loc[r.item]; print('  ',r.item,r.new,'moved',r.moved,c['name'],'|',c.department,'view-new',int(o[(o.item==r.item)&(o.beh=='view')].new.iloc[0]))
    print('  1998 in seen',1998 in seen, o[o.item==1998].to_dict('records'), cat.loc[1998,'name'] if 1998 in cat.index else None)
