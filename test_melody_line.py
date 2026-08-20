import numpy as np, itertools, io, contextlib
src=open('backend_api.py').read()
def grab(n):
    i=src.index("def "+n+"("); j=src.index("\ndef ", i+1); return src[i:j]
from dataclasses import dataclass
@dataclass
class Note:
    start: float
    end: float
    pitch_midi: int
    hand: str = "RH"
    velocity: int = 80
ns={'np':np,'Note':Note,'List':list}
for f in ['_suppress_harmonic_partials','_select_line','_weighted_median','_collapse_octave_duplicates','_quantize_to_beat_grid','_two_means_split_midi']:
    exec(grab(f),ns)
SUP,SEL,COL,QNT=ns['_suppress_harmonic_partials'],ns['_select_line'],ns['_collapse_octave_duplicates'],ns['_quantize_to_beat_grid']
def quiet(f,*a,**k):
    with contextlib.redirect_stdout(io.StringIO()): return f(*a,**k)
ok=True
def check(name,cond,info=""):
    global ok
    print(("PASS " if cond else "FAIL ")+name+("  "+info if info else "")); ok&=bool(cond)

# 1. staircase scene (reproduction of the real 34-38 s failure)
beat=0.1515
mel=[65,68,70,72,73,72,70,68,70,73,75,77,75,73,72,70,68,66,68,70,72,70,68,65]
c=[];truth=set()
for i,p in enumerate(mel):
    s=2*i*beat; c.append((s,s+0.26,p,0.55)); truth.add((round(s,3),p))
for k in range(2*len(mel)):
    b=[46,49,48,46][(k//8)%4]; s=k*beat
    for off,a in ((0,.95),(12,.85),(24,.78),(19,.55)): c.append((s,s+0.14,b+off,a))
up=[e for e in quiet(SUP,c) if e[2]>=62]
got=[p for _s,_e,p,_a in SEL(up)]
hit=sum(1 for s,e,p,a in SEL(up) if (round(s,3),p) in truth)
run=max(len(list(g)) for _,g in itertools.groupby(got))
check("melody survives a louder harmonic ostinato", hit>=18 and run<=3,
      f"{hit}/{len(mel)} melody notes, longest run {run}, {len(set(got))} distinct")

# 2. a genuinely repeated-note phrase must NOT be flattened away
rep=[70]*8+[65]*4
c2=[(i*0.303,i*0.303+0.24,p,0.8) for i,p in enumerate(rep)]
o=[p for _s,_e,p,_a in SEL(c2)]
check("repeated notes still kept", o==rep, f"{len(o)}/12 -> {o}")

# 3. suppression must not fire on a lone melody with no bass under it
solo=[(i*0.3,i*0.3+0.25,p,0.7) for i,p in enumerate([65,67,69,70,72,70,69,67])]
out=quiet(SUP,solo)
check("no false suppression on a bare melody", all(a[3]==b[3] for a,b in zip(solo,out)))

# 4. quantiser idempotency
notes=[Note(0.076+i*0.151,0.076+i*0.151+0.14,60+ (i%5)) for i in range(30)]
q1,ph1=quiet(QNT,notes,99.0,4); q2,ph2=quiet(QNT,q1,99.0,4)
check("quantiser is idempotent",[ (round(n.start,6),n.pitch_midi) for n in q1]==[(round(n.start,6),n.pitch_midi) for n in q2])

# 5. octave collapse leaves a single-octave melody untouched
mn=[Note(i*0.3,i*0.3+0.25,p) for i,p in enumerate([65,68,70,72,73,72,70,68])]
check("octave collapse is a no-op on one octave",[n.pitch_midi for n in quiet(COL,mn,70.0)]==[n.pitch_midi for n in mn])
print("\nALL GOOD" if ok else "\nSOMETHING FAILED")
