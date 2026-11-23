#!/usr/bin/python3 -I
# card6: split an nvprof --print-gpu-trace --csv (us) into bursts (>GAP ms idle) and print per-burst top kernels.
import csv, re, sys, collections
f = sys.argv[1]; GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0; TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 22
rows = []; hdr = None
for r in csv.reader(l for l in open(f, errors="replace") if not l.startswith("==")):
    if hdr is None:
        if "Start" in r and "Name" in r: hdr = r; iS, iD, iN = r.index("Start"), r.index("Duration"), r.index("Name")
        continue
    try: rows.append((float(r[iS]), float(r[iD] or 0), r[iN]))
    except (ValueError, IndexError): pass
rows.sort()
def short(n):
    n = n.replace("void ", "")
    m = re.match(r"([A-Za-z_0-9:]+)(<[^()]*>)?", n)
    s = m.group(1) + (m.group(2) or "") if m else n
    return s[:90]
bursts = []; cur = [rows[0]]
for a, b in zip(rows, rows[1:]):
    if b[0] - (a[0] + a[1]) > GAP * 1000: bursts.append(cur); cur = []
    cur.append(b)
bursts.append(cur)
for k, bu in enumerate(bursts):
    span = bu[-1][0] + bu[-1][1] - bu[0][0]; busy = sum(x[1] for x in bu)
    print(f"burst {k}: n={len(bu)} span={span/1e3:.1f} ms busy={busy/1e3:.1f} ms ({100*busy/max(span,1):.1f}%)")
for k in [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else []:
    bu = bursts[k]; span = bu[-1][0] + bu[-1][1] - bu[0][0]; busy = sum(x[1] for x in bu)
    agg = collections.defaultdict(lambda: [0.0, 0])
    for s, d, n in bu: agg[short(n)][0] += d; agg[short(n)][1] += 1
    print(f"\n== burst {k} span {span/1e3:.1f} ms busy {busy/1e3:.1f} ms; top kernels (ms, %busy, calls, avg us)")
    for n, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:TOP]:
        print(f"{t/1e3:9.1f} {100*t/busy:5.1f}% {c:7d} {t/c:8.1f}  {n}")
    # idle gaps histogram
    gaps = [b[0] - (a[0] + a[1]) for a, b in zip(bu, bu[1:])]
    big = [g for g in gaps if g > 50]
    print(f"gaps: total {sum(g for g in gaps if g>0)/1e3:.1f} ms; >50us: {len(big)} totalling {sum(big)/1e3:.1f} ms; >1ms: {sum(1 for g in gaps if g>1000)} totalling {sum(g for g in gaps if g>1000)/1e3:.1f} ms")
