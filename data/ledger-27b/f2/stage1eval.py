#!/usr/bin/env python3
# Limits fixed before results (scout3-pick.md): pooled short tg >=.97x C, tg@52k >=16.28, tg@100k >=13.96, pp4k >=.97x, tg2agg >=-.8%, clock >=.97x. Exit 3 = F short tg under .93x, skip long runs.
import json,sys
R=[json.loads(l) for l in open('/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113/fp32mv-stage1-20261109.jsonl')]
avg=lambda s,a,k:(lambda v:sum(v)/len(v) if v else float('nan'))([r[k] for r in R if r['sec']==s and r['arm']==a and r.get(k) is not None])
for k,s in (('pooled','short'),('pp4k','short'),('tg1','short'),('tg2agg','short'),('MHz','short'),('W','short'),('maxT','short'),('tg','52k'),('tg','100k')):
    c,f=avg(s,'C',k),avg(s,'F',k); print(f'{s}.{k}: C {c:.2f} F {f:.2f} ratio {f/c if c else 0:.3f}')
print('accept',[(r['arm'],r['accept']) for r in R if r['sec']=='short'])
print('long cached',[(r['sec'],r['arm'],r['rep'],r['prompt_n'],r.get('cached'),r['tg']) for r in R if r['sec'] in('52k','100k')])
sys.exit(3 if avg('short','F','pooled')<.93*avg('short','C','pooled') else 0)
