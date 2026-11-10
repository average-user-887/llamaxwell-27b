#!/usr/bin/env python3
"""Stage 0 log replay: re-processed prefill in llama-server logs (read-only).

Streams `docker logs <container> 2>&1` through a pipe; keeps only per-task records.
Per slot, a task is: "new prompt ... task.n_tokens = P" -> optional
"restored context checkpoint ... n_past = R" -> first "cached n_tokens = C" (common
prefix actually kept; this is the n_past used) -> "prompt eval time = ms / N tokens"
-> "release: ... stop processing: n_tokens = E" (end of task on slot).
Prior task on the same slot = last released task before this task's new prompt.

Re-processed tokens = min(max(0, E_prior - n_past), N_eval), split into
older-prompt part min(max(0, min(E,P_prior)-n_past), reproc) and reply part (rest).
Time = reproc_tokens * (prompt_eval_ms / N_eval) for that same eval.
"""
import re
import subprocess
import sys
from collections import defaultdict

CONTAINERS = ["llamacpp-m6000", "llamacpp-m6000-RETIRED-card16-20261027"]
# (name, n_past_max, prior_prompt_min): unrelated-conversation flag
THRESHOLDS = [("A", 512, 2048), ("B", 2048, 4096)]
BUCKETS = [(4096, "<4k"), (16384, "4k-16k"), (32768, "16k-32k"), (65536, "32k-64k"), (10**9, ">=64k")]

RE_SLOT = re.compile(r"\bid\s+(\d+) \| task (-?\d+) \| (.*)$")
RE_NEWPROMPT = re.compile(r"new prompt, .*task\.n_tokens = (\d+)")
RE_RESTORE = re.compile(r"restored context checkpoint .*n_past = (\d+)")
RE_CACHED = re.compile(r"cached n_tokens = (\d+)")
RE_EVAL = re.compile(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
RE_RELEASE = re.compile(r"stop processing: n_tokens = (\d+)")


def parse_container(name):
    """Return list of task records and format-coverage counters."""
    cov = defaultdict(int)
    proc = subprocess.Popen(["docker", "logs", name], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    cur = {}          # slot -> current record
    last_rel = {}     # slot -> last finished record (end E may be None)
    recs = []
    for line in proc.stdout:
        m = RE_SLOT.search(line)
        if not m:
            continue
        slot, task, msg = int(m.group(1)), int(m.group(2)), m.group(3)
        if task < 0:
            continue
        c = cur.get(slot)
        mm = RE_NEWPROMPT.search(msg)
        if mm:
            cov["new_prompt"] += 1
            prior = last_rel.get(slot)
            if c is not None and c.get("done") is None:
                c["done"] = True
                recs.append(c)
            c = {"container": name, "slot": slot, "task": task, "P": int(mm.group(1)),
                 "restore": None, "cached": None, "eval_ms": 0.0, "eval_tok": 0,
                 "prior_task": None, "prior_P": None, "prior_E": None}
            if prior is not None:
                c["prior_task"] = prior["task"]
                c["prior_P"] = prior["P"]
                c["prior_E"] = prior["E"]
            cur[slot] = c
            continue
        if c is None or c["task"] != task:
            continue
        mm = RE_RESTORE.search(msg)
        if mm:
            cov["restored"] += 1
            c["restore"] = int(mm.group(1))
            continue
        mm = RE_CACHED.search(msg)
        if mm:
            cov["cached"] += 1
            if c["cached"] is None:
                c["cached"] = int(mm.group(1))
            continue
        mm = RE_EVAL.search(msg)
        if mm:
            cov["prompt_eval"] += 1
            c["eval_ms"] = float(mm.group(1))
            c["eval_tok"] = int(mm.group(2))
            continue
        mm = RE_RELEASE.search(msg)
        if mm:
            cov["release"] += 1
            c["E"] = int(mm.group(1))
            last_rel[slot] = c
            continue
    for slot, c in cur.items():
        if c.get("done") is None:
            recs.append(c)
    proc.stdout.close()
    rc = proc.wait()
    cov["docker_rc"] = rc
    return recs, cov


def analyse(recs, thr):
    """thr = (name, n_past_max, prior_P_min) or None for no exclusion."""
    out = defaultdict(float)
    depth = defaultdict(int)
    for r in recs:
        n_past = r["cached"] if r["cached"] is not None else 0
        ntok, ms = r["eval_tok"], r["eval_ms"]
        out["tasks"] += 1
        out["prefill_ms_all"] += ms
        out["eval_tok_all"] += ntok
        if r["restore"] is not None:
            out["restores"] += 1
            for lim, lab in BUCKETS:
                if n_past < lim:
                    depth[lab] += 1
                    break
        elif n_past == 0:
            out["full_reprocess"] += 1
        else:
            out["in_slot_reuse"] += 1
        if r["prior_task"] is None:
            out["unattr_no_prior"] += 1
            unattr = True
        elif r["prior_E"] is None:
            out["unattr_no_prior_end"] += 1
            unattr = True
        else:
            unattr = False
        E = r["prior_E"]
        if not unattr and thr is not None and n_past < thr[1] and r["prior_P"] >= thr[2]:
            out["unattr_unrelated"] += 1
            unattr = True
        raw = 0
        older = reply = 0
        if r["prior_task"] is not None and E is not None:
            raw = min(max(0, E - n_past), ntok)
            older_raw = max(0, min(E, r["prior_P"]) - n_past)
            older = min(older_raw, raw)
            reply = raw - older
        ms_per_tok = ms / ntok if ntok > 0 else 0.0
        reproc_ms = raw * ms_per_tok
        older_ms = older * ms_per_tok
        out["reproc_tok_all"] += raw
        out["reproc_ms_all"] += reproc_ms
        if not unattr:
            out["attr_tasks"] += 1
            out["prefill_ms_attr"] += ms
            out["reproc_tok_attr"] += raw
            out["reproc_reply_tok_attr"] += reply
            out["reproc_older_tok_attr"] += older
            out["reproc_ms_attr"] += reproc_ms
            out["reproc_reply_ms_attr"] += reply * ms_per_tok
            out["reproc_older_ms_attr"] += older_ms
            if older > 0:
                out["older_evals_attr"] += 1
            if reply > 0:
                out["reply_evals_attr"] += 1
    return out, depth


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def block(title, o):
    L = []
    L.append(title)
    L.append(f"  tasks={int(o['tasks'])} (attributable {int(o['attr_tasks'])})  restores={int(o['restores'])} "
             f"in_slot_reuse={int(o['in_slot_reuse'])} full_reprocess(cached 0, no restore)={int(o['full_reprocess'])}")
    L.append(f"  prefill minutes: all={o['prefill_ms_all']/60000:.2f}  attributable={o['prefill_ms_attr']/60000:.2f}")
    L.append(f"  re-processed tokens: all={int(o['reproc_tok_all'])}  attributable={int(o['reproc_tok_attr'])} "
             f"(reply part {int(o['reproc_reply_tok_attr'])}, older-prompt part {int(o['reproc_older_tok_attr'])})")
    L.append(f"  re-processed minutes: all={o['reproc_ms_all']/60000:.2f}  attributable={o['reproc_ms_attr']/60000:.2f} "
             f"(reply {o['reproc_reply_ms_attr']/60000:.2f}, older-prompt {o['reproc_older_ms_attr']/60000:.2f})")
    L.append(f"  SHARE of prefill minutes: all-tasks={pct(o['reproc_ms_all'], o['prefill_ms_all'])}  "
             f"attributable (primary)={pct(o['reproc_ms_attr'], o['prefill_ms_attr'])}")
    L.append(f"  reply vs older-prompt split of re-processed tokens (attributable): "
             f"reply={pct(o['reproc_reply_tok_attr'], o['reproc_tok_attr'])} older={pct(o['reproc_older_tok_attr'], o['reproc_tok_attr'])}")
    L.append(f"  evals with mismatch BEFORE prior reply start (older-prompt part>0, attributable): "
             f"count={int(o['older_evals_attr'])} minutes={o['reproc_older_ms_attr']/60000:.2f} "
             f"({pct(o['reproc_older_ms_attr'], o['reproc_ms_attr'])} of attributable re-processed minutes)")
    L.append(f"  unattributable: no_prior={int(o['unattr_no_prior'])} no_prior_end={int(o['unattr_no_prior_end'])} "
             f"unrelated={int(o['unattr_unrelated'])}")
    return L


def main():
    all_recs = []
    covs = {}
    for name in CONTAINERS:
        recs, cov = parse_container(name)
        covs[name] = cov
        all_recs.append((name, recs))
    lines = []
    lines.append("Stage 0 log replay (read-only docker logs; no GPU, no container changes)")
    lines.append("Conversion: re-processed ms = re-processed tokens x (that eval's prompt_eval_ms / its N_eval tokens).")
    lines.append("n_past = first 'cached n_tokens' line of the task (restore n_past equals it in samples).")
    lines.append("Prior end E = 'stop processing n_tokens' of previous task on same slot; prior prompt P = its 'task.n_tokens'.")
    lines.append("Re-processed = min(max(0, E - n_past), N_eval). Older-prompt part = portion below prior prompt P.")
    lines.append("Unrelated-conversation flag: n_past < X AND prior P >= Y.  A: X=512, Y=2048.  B: X=2048, Y=4096.")
    lines.append("")
    pooled = []
    for name, recs in all_recs:
        c = covs[name]
        lines.append(f"== {name}: records={len(recs)} new_prompt={c['new_prompt']} restored_lines={c['restored']} "
                     f"cached_lines={c['cached']} prompt_eval_lines={c['prompt_eval']} release_lines={c['release']} "
                     f"docker_rc={c['docker_rc']}")
        pooled.extend(recs)
    lines.append(f"== POOLED: records={len(pooled)}")
    lines.append("")
    # Per-container and pooled outputs.
    for label, recs in [(n, r) for n, r in all_recs] + [("POOLED", pooled)]:
        for tname, n_max, p_min in THRESHOLDS:
            o, depth = analyse(recs, (tname, n_max, p_min))
            lines += block(f"[{label}] threshold {tname} (n_past<{n_max} & prior_P>={p_min})", o)
        o0, _ = analyse(recs, None)
        lines.append(f"  [{label}] no unrelated exclusion (raw): share all-tasks = {pct(o0['reproc_ms_all'], o0['prefill_ms_all'])}")
        o, depth = analyse(recs, THRESHOLDS[0])
        lines.append(f"  [{label}] restore n_past depth (all restores): " +
                     ", ".join(f"{lab}={depth.get(lab,0)}" for _, lab in BUCKETS))
        lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
