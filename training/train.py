import argparse, json, math, os, random, sys
from model import NadosRanker, TARGETS


def load_rows(path):
    rows=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    if not rows: raise ValueError('EMPTY_DATASET')
    return rows


def bce(y,p):
    p=max(1e-7,min(1-1e-7,p))
    return -(y*math.log(p)+(1-y)*math.log(1-p))


def evaluate(model, rows):
    losses=[]
    for r in rows:
        p=model.predict(r['features'])
        losses.extend(bce(float(r['targets'].get(k,0)),p[k]) for k in TARGETS)
    return sum(losses)/len(losses)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=250);ap.add_argument('--lr',type=float,default=0.03);ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); random.seed(a.seed)
    rows=load_rows(a.data); split=max(1,int(len(rows)*0.8)); random.shuffle(rows); train,valid=rows[:split],rows[split:]
    model=NadosRanker(hidden=16,seed=a.seed).fit(train,epochs=a.epochs,lr=a.lr)
    artifact={'name':'Nados v1.1','version':'1.1.0','task':'social_engagement_ranking','features':model.state_dict()['features'],'targets':TARGETS,'metrics':{'train_bce':evaluate(model,train),'validation_bce':evaluate(model,valid) if valid else None,'train_examples':len(train),'validation_examples':len(valid)},'model':model.state_dict()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True)
    with open(a.out,'w',encoding='utf-8') as f: json.dump(artifact,f,ensure_ascii=False,separators=(',',':'))
    print(json.dumps(artifact['metrics'],ensure_ascii=False))

if __name__=='__main__': main()
