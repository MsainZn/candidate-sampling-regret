"""Table 8 and the decision numbers: worst-case regret in points and relative to the best payoff
under each law, the break-even weight with its seed-level interval, and the value of information.

    python analysis/b_analysis.py   # reads results/aggregated, writes results/derived/b_analysis.json
"""
import csv, statistics as st, math, json
ARMS=["uniform","popularity","inbatch","inbatch_logq"]; LAWS=["uniform","sqrt_pop","popularity"]; AB={"uniform":"U","popularity":"P","inbatch":"B","inbatch_logq":"Q"}
UL,PL=("uniform","inbatch_logq"),("popularity","inbatch")
rows=[r for r in csv.DictReader(open("results/aggregated/all_e2_crosscert.csv")) if r["split"]=="random" and r["train_law"] in ARMS and r["ndcg20"]]
DS=["amazon_books","hotelrec","yelp","google_connecticut","goodreads_all","netflix","ml20m"]
SEEDS=["0","1","2","3"]; t3=3.182446
OUT={}
def worst(v,d,rel):
    best={L:max(v[(d,a,L)] for a in ARMS) for L in LAWS}
    return {a:max((best[L]-v[(d,a,L)])/(best[L] if rel else 1) for L in LAWS)*(100 if rel else 1) for a in ARMS}
def breakeven(v,d,rel):
    bu=max(v[(d,a,"uniform")] for a in ARMS) if rel else 1; bp=max(v[(d,a,"popularity")] for a in ARMS) if rel else 1
    def gap(w):
        e=lambda a:(1-w)*v[(d,a,"uniform")]/bu + w*v[(d,a,"popularity")]/bp
        return max(e(a) for a in UL)-max(e(a) for a in PL)
    return next(i/100000 for i in range(100001) if gap(i/100000)<=0)
def ci(x): return st.mean(x), t3*st.stdev(x)/2
for metric in ["ndcg20","hr10"]:
    vs={(r["dataset"],r["train_law"],r["cert_ref"],r["seed"]):100*float(r[metric]) for r in rows}
    v={(d,a,L):st.mean(vs[(d,a,L,s)] for s in SEEDS) for d in DS for a in ARMS for L in LAWS}
    print("=====",metric)
    for d in DS:
        wa=worst(v,d,False); wr=worst(v,d,True)
        best={L:max(v[(d,a,L)] for a in ARMS) for L in LAWS}
        ev={a:st.mean(v[(d,a,L)] for L in LAWS) for a in ARMS}
        evpi=st.mean(best.values())-max(ev.values())
        # seed-level: group differences of worst-case (abs and rel), w† per seed
        sv=lambda s:{(d,a,L):vs[(d,a,L,s)] for a in ARMS for L in LAWS}
        gd_abs=[]; gd_rel=[]; wb=[]; wbr=[]
        for s in SEEDS:
            x=sv(s); A=worst(x,d,False); R=worst(x,d,True)
            gd_abs.append(min(A[a] for a in PL)-min(A[a] for a in UL))
            gd_rel.append(min(R[a] for a in PL)-min(R[a] for a in UL))
            wb.append(breakeven(x,d,False)); wbr.append(breakeven(x,d,True))
        ma,ha=ci(gd_abs); mr,hr=ci(gd_rel); mw,hw=ci(wb); mwr,hwr=ci(wbr)
        print(f"{d:19s} REL U/P/B/Q " + " ".join(f"{wr[a]:5.1f}" for a in ARMS) + f" pick {AB[min(wr,key=wr.get)]} | grpdiff(P/B-U/Q) abs {ma:6.2f}±{ha:.2f} rel {mr:6.2f}±{hr:.2f} | w† {breakeven(v,d,False):.3f} seed {mw:.3f}±{hw:.3f} | w†rel {breakeven(v,d,True):.3f} seed {mwr:.3f}±{hwr:.3f} | EVPI {evpi:.2f}")
        OUT[f"{metric}|{d}"]=dict(rel=wr, abs=wa, evpi=evpi, grp_abs=(ma,ha), grp_rel=(mr,hr), wdag=breakeven(v,d,False), wdag_seed=(mw,hw), wdag_rel=breakeven(v,d,True), wdag_rel_seed=(mwr,hwr))
import os
os.makedirs("results/derived", exist_ok=True)
json.dump(OUT, open("results/derived/b_analysis.json","w"), indent=1)
