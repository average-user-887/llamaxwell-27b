#!/bin/bash
# 7 Nov night chain 1: clock sweep (987 vs 1152, same-session, QUICK_LONG rebase) -> pick clock -> tg at long ctx. Prod left stopped (NO_RESTORE).
set -u; O=/media/avg-usr/Storage/m6000-llama-cpp/opt27b-20261113; G=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; TS=$(date +%Y%m%d-%H%M)
cd $O; exec 9>$O/../gpu-bench.lock; flock 9; touch $O/NO_RESTORE
trap 'kill $SP 2>/dev/null; docker rm -f llamacpp-m6000-27b-exp >/dev/null 2>&1' EXIT
docker stop -t 60 llamacpp-m6000-27b >/dev/null 2>&1
for clk in 987 1152; do
  sudo -n nvidia-smi -i $G -ac 3305,$clk >/dev/null || { echo "clock set failed $clk"; exit 3; }
  nvidia-smi -i $G --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active -l 2 --format=csv > temps-clk$clk-$TS.csv & SP=$!
  QUICK_LONG=1 python3 quick.py 23 2>&1 | tail -4; kill $SP
  cp state.json state-clk$clk-$TS.json
  echo "clk$clk peak_temp $(awk -F', ' 'NR>1{t=$2+0; if(t>m)m=t} END{print m}' temps-clk$clk-$TS.csv)"
done
python3 - $TS <<'PY' | tee clock-pick-$TS.txt
import json, sys; ts = sys.argv[1]
a, b = (json.load(open(f"state-clk{c}-{ts}.json"))["base_m"] for c in (987, 1152))
pk = {c: max(int(l.split(", ")[1]) for l in list(open(f"temps-clk{c}-{ts}.csv"))[1:] if l.split(", ")[1].strip().isdigit()) for c in (987, 1152)}
d = {k: b[k]/a[k]-1 for k in ("pp4k", "tg1", "tg2agg", "pp23k") if a.get(k) and b.get(k)}
ok = d.get("pp23k", 0) >= 0.006 and min(d.values()) > -0.006 and pk[1152] <= 77
print(json.dumps({"pick": 1152 if ok else 987, "delta": {k: round(v*100, 2) for k, v in d.items()}, "peak": pk, "a987": {k: a.get(k) for k in d}, "b1152": {k: b.get(k) for k in d}}))
PY
PICK=$(python3 -c "import json;print(json.load(open('clock-pick-$TS.txt'))['pick'])")
sudo -n nvidia-smi -i $G -ac 3305,$PICK >/dev/null; cp state-clk$PICK-$TS.json state.json; echo "clock set $PICK"
echo "== tg long ctx @$PICK"; python3 tg_long.py 2600 4900 2>&1 | tee tg-long-$TS.txt
echo CHAIN1 DONE
