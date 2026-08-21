import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
from voices import separate, top_voice
beat=0.303
MEL=[72,74,75,77,75,74,72,70,72,74,75,74,72,70,69,70]
ALTO=[65,65,67,67,68,68,67,65,65,67,68,68,67,65,64,65]
def build(overlap=0.0, partials=False, spurious=0):
    notes=[];truth=[]
    def add(s,e,p,a,t): notes.append((s,e,p,a)); truth.append(t)
    for i,p in enumerate(MEL):  add(i*beat,i*beat+0.28+overlap,p,0.6,'mel')
    for i,p in enumerate(ALTO): add(i*beat,i*beat+0.28+overlap,p,0.75,'alto')
    for i in range(0,len(MEL),2):
        b=41+(i//2%2)*7
        add(i*beat,i*beat+2*beat,b,0.95,'bass')
        if partials:
            for off,amp in ((12,.8),(19,.55),(24,.7)):
                add(i*beat,i*beat+2*beat,b+off,amp,'partial')
    if spurious:
        import random; random.seed(0)
        for k in range(spurious):
            s=random.uniform(0,len(MEL)*beat)
            add(s,s+0.15,random.randint(60,90),0.4,'junk')
    order=sorted(range(len(notes)),key=lambda k:(notes[k][0],notes[k][2]))
    return [notes[k] for k in order],[truth[k] for k in order]
ok=True
def run(tag, expect, **kw):
    global ok
    notes,truth=build(**kw)
    top=set(top_voice(notes))
    hit=sum(1 for i in top if truth[i]=='mel')
    from collections import Counter
    good = hit>=expect
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} {tag:30s} tune {hit:2d}/16 (need {expect}) "
          f"| {dict(Counter(truth[i] for i in top))}")

run("clean three voices",        16)
run("legato 60 ms",              16, overlap=0.06)
run("harmonic partials",         10, partials=True)
run("partials + legato",         10, partials=True, overlap=0.06)
run("partials + legato + junk",   5, partials=True, overlap=0.06, spurious=20)
print("\nALL GOOD" if ok else "\nSOMETHING FAILED")
