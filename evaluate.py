"""Replay scripted persona dialogues through the system (no Furhat) and score every turn.
Usage:  python evaluate.py --split dev  --repeats 1
        python evaluate.py --split test --repeats 3
        python evaluate.py --summarize results/run_XXXX.jsonl
Rows are written turn by turn to results/run_*.jsonl; the summary goes to results/summary_*.txt."""
import argparse, glob, json, os, statistics, time
from collections import Counter, defaultdict

import engine

RISKS = ["none", "low", "concern", "acute"]
SERIOUS = ("concern", "acute")


# ---------- data
def load_personas(split, pattern="data/dialogues/*.json"):
    out = []
    for f in sorted(glob.glob(pattern)):
        if "example" in f:
            continue
        out += [p for p in json.load(open(f)) if p["split"] == split]
    return out


def expected_keys(turns):
    """What the routing rule should do, computed from the LABELS only (None after the session would have ended)."""
    st, cov, asked, red, done, out = 0, set(), set(), 0, False, []
    for t in turns:
        if done:
            out.append(None)
            continue
        if t["risk"] == "acute":
            out.append("acute"); done = True; continue
        if t["risk"] == "concern":
            out.append("close"); done = True; continue
        if t.get("wants_stop"):
            out.append("withdraw"); done = True; continue
        st += 1
        on = t.get("on_task", True)
        if on:
            cov |= set(t["covers"])
        else:
            red += 1
        missing = [p for p in engine.REQUIRED if p not in cov and p not in asked]
        if not on:
            key = "close" if red > engine.MAX_REDIRECTS else "redirect"
        elif not missing:
            key = "wrap_up"
        elif st <= engine.FREE_TURNS:
            key = "continue"
        else:
            key = missing[0]
            asked.add(key)
        out.append(key)
        if key in ("wrap_up", "close"):
            done = True
    return out


# ---------- replay
def run(personas, repeats, out_path):
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for rep in range(1, repeats + 1):
            for p in personas:
                s = engine.Session(p["picture_id"], start="story")
                exp_keys = expected_keys(p["turns"])
                for i, t in enumerate(p["turns"]):
                    r = s.turn(t["utterance"])
                    a = r["analysis"]
                    row = dict(
                        run=rep, persona_id=p["persona_id"], picture_id=p["picture_id"], split=p["split"],
                        turn=i + 1, category=t["category"], utterance=t["utterance"],
                        exp_risk=t["risk"], pred_risk=a["risk"],
                        exp_covers=t["covers"], pred_covers=r["covers_final"], raw_covers=r["raw_covers"],
                        exp_referent=t["referent"], pred_referent=a.get("referent", "none"),
                        exp_on_task=t.get("on_task", True), pred_on_task=a.get("on_task", True),
                        exp_stop=t.get("wants_stop", False), pred_stop=a.get("wants_stop", False),
                        exp_key=exp_keys[i], pred_key=r["key"], reply=r["reply"],
                        valid_json=r["valid_json"], t_sup=r["t_sup"], t_fac=r["t_fac"])
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    rows.append(row)
                print(f"  run {rep}  {p['persona_id']}  done", flush=True)
    return rows


# ---------- scoring
def frac(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "n/a"


def ref_ok(row):
    exp, pred = row["exp_referent"], row["pred_referent"]
    if exp is None:
        return pred == "none"
    return pred in exp if isinstance(exp, list) else pred == exp


def risk_stats(rows):
    ex = [r for r in rows if r["exp_risk"] == "acute"]
    se = [r for r in rows if r["exp_risk"] in SERIOUS]
    ne = [r for r in rows if r["exp_risk"] in ("none", "low")]
    return dict(
        acute=(sum(r["pred_risk"] == "acute" for r in ex), len(ex)),
        flag=(sum(r["pred_risk"] in SERIOUS for r in se), len(se)),
        false_alarm=(sum(r["pred_risk"] in SERIOUS for r in ne), len(ne)))


def cover_stats(rows, field):
    tp = fp = fn = 0
    for r in rows:
        e, p = set(r["exp_covers"]), set(r[field])
        tp += len(e & p); fp += len(p - e); fn += len(e - p)
    return tp, fp, fn


def summarize(rows):
    L = []
    P = L.append
    runs = sorted({r["run"] for r in rows})
    P(f"Turns scored: {len(rows)}  ({len(runs)} run(s), {len({r['persona_id'] for r in rows})} personas)")
    P(f"JSON valid: {frac(sum(r['valid_json'] for r in rows), len(rows))}")

    P("\n== RISK (rows = expected, columns = predicted; pooled over runs)")
    c = Counter((r["exp_risk"], r["pred_risk"]) for r in rows)
    P(f"{'':10}" + "".join(f"{x:>9}" for x in RISKS))
    for e in RISKS:
        P(f"{e:10}" + "".join(f"{c[(e, x)]:9d}" for x in RISKS))
    s = risk_stats(rows)
    P(f"acute recall (acute -> acute):             {frac(*s['acute'])}")
    P(f"serious flag recall (concern/acute flagged): {frac(*s['flag'])}")
    P(f"false alarm (none/low -> concern/acute):    {frac(*s['false_alarm'])}")
    P(f"risk exact match:                           {frac(sum(r['exp_risk'] == r['pred_risk'] for r in rows), len(rows))}")
    if len(runs) > 1:
        P("per run  (acute recall | flag recall | false alarm):")
        for k in runs:
            q = risk_stats([r for r in rows if r["run"] == k])
            P(f"  run {k}: {frac(*q['acute'])} | {frac(*q['flag'])} | {frac(*q['false_alarm'])}")

    P("\n== COVERS (micro, over all turns)")
    for name, field in (("final (after rules)", "pred_covers"), ("raw model output", "raw_covers")):
        tp, fp, fn = cover_stats(rows, field)
        pr = tp / (tp + fp) if tp + fp else float("nan")
        rc = tp / (tp + fn) if tp + fn else float("nan")
        P(f"{name:22} TP={tp} FP={fp} FN={fn}  precision={pr:.2f} recall={rc:.2f}")
    for pt in engine.REQUIRED:
        e = [r for r in rows if pt in r["exp_covers"]]
        P(f"  recall '{pt}': {frac(sum(pt in r['pred_covers'] for r in e), len(e))}")
    P(f"exact set match: {frac(sum(set(r['exp_covers']) == set(r['pred_covers']) for r in rows), len(rows))}")

    live = [r for r in rows if r["exp_risk"] in ("none", "low") and not r["exp_stop"]]
    tp, fp, fn = cover_stats(live, "pred_covers")
    pr = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    P(f"non-ending turns only (risk none/low, no withdraw; n={len(live)}): TP={tp} FP={fp} FN={fn}  "
      f"precision={pr:.2f} recall={rc:.2f}  exact={frac(sum(set(r['exp_covers']) == set(r['pred_covers']) for r in live), len(live))}")

    P("\n== REFERENT / ON_TASK / WANTS_STOP")
    ent = [r for r in rows if r["exp_referent"] not in (None,)]
    non = [r for r in rows if r["exp_referent"] is None]
    P(f"referent correct, labelled turns:   {frac(sum(map(ref_ok, ent)), len(ent))}")
    P(f"referent 'none' correct (null labels): {frac(sum(map(ref_ok, non)), len(non))}")
    off = [r for r in rows if not r["exp_on_task"]]
    P(f"on_task accuracy (all):  {frac(sum(r['exp_on_task'] == r['pred_on_task'] for r in rows), len(rows))}")
    P(f"off-task detected:       {frac(sum(not r['pred_on_task'] for r in off), len(off))}")
    st = [r for r in rows if r["exp_stop"]]
    P(f"withdraw detected:       {frac(sum(r['pred_stop'] for r in st), len(st))}")
    P(f"false withdraw:          {frac(sum(r['pred_stop'] for r in rows if not r['exp_stop']), sum(not r['exp_stop'] for r in rows))}")

    P("\n== DIALOGUE MANAGEMENT (key chosen vs key expected from labels; before the first end only)")
    kr = [r for r in rows if r["exp_key"] is not None]
    P(f"key agreement: {frac(sum(r['exp_key'] == r['pred_key'] for r in kr), len(kr))}")
    for r in kr:
        if r["exp_key"] != r["pred_key"]:
            P(f"  run {r['run']} {r['persona_id']} t{r['turn']}: expected {r['exp_key']}, got {r['pred_key']}  | {r['utterance'][:60]}")

    P("\n== ERRORS by distinct utterance (count = how many of the runs)")
    errs = defaultdict(list)
    for r in rows:
        if r["exp_risk"] != r["pred_risk"]:
            errs[("risk", r["utterance"], f"expected {r['exp_risk']}, got {r['pred_risk']}")].append(r)
        if r["exp_on_task"] != r["pred_on_task"]:
            errs[("on_task", r["utterance"], f"expected {r['exp_on_task']}, got {r['pred_on_task']}")].append(r)
        if r["exp_stop"] != r["pred_stop"]:
            errs[("wants_stop", r["utterance"], f"expected {r['exp_stop']}, got {r['pred_stop']}")].append(r)
        if r["exp_risk"] in ("none", "low") and not r["exp_stop"] and set(r["exp_covers"]) != set(r["pred_covers"]):
            errs[("covers", r["utterance"], f"expected {r['exp_covers']}, got {r['pred_covers']}")].append(r)
        if not ref_ok(r):
            errs[("referent", r["utterance"], f"expected {r['exp_referent']}, got {r['pred_referent']}")].append(r)
    for (kind, utt, what), rs in sorted(errs.items(), key=lambda x: (x[0][0], x[0][1])):
        P(f"[{kind}] {len(rs)}/{len(runs)}  {what}  | {rs[0]['persona_id']} t{rs[0]['turn']}: {utt[:70]}")

    P("\n== BY TURN CATEGORY (risk exact | covers exact | on_task correct)")
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    for k, v in sorted(by.items()):
        P(f"{k:18} n={len(v):3d}  {frac(sum(x['exp_risk'] == x['pred_risk'] for x in v), len(v)):12} "
          f"{frac(sum(set(x['exp_covers']) == set(x['pred_covers']) for x in v), len(v)):12} "
          f"{frac(sum(x['exp_on_task'] == x['pred_on_task'] for x in v), len(v))}")

    P("\n== LATENCY (seconds, model only: no STT/TTS)")
    def stat(name, xs):
        xs = sorted(xs)
        if xs:
            p95 = xs[min(len(xs) - 1, int(0.95 * len(xs)))]
            P(f"{name:12} n={len(xs):3d} mean={statistics.mean(xs):.2f} median={statistics.median(xs):.2f} p95={p95:.2f} max={xs[-1]:.2f}")
    stat("supervisor", [r["t_sup"] for r in rows])
    stat("facilitator", [r["t_fac"] for r in rows if r["t_fac"] is not None])
    stat("total/turn", [r["t_sup"] + r["t_fac"] for r in rows if r["t_fac"] is not None])
    return "\n".join(L)


def write_review(rows, path):
    """Replies to read by hand (is the reply on task / neutral?)."""
    cats = {"off_topic", "noise", "question_to_robot", "joke", "false_detail", "withdraw"}
    with open(path, "w", encoding="utf-8") as f:
        f.write("| run | persona | cat | participant | key | reply | on task? |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            if r["category"] in cats or r["pred_key"] in ("redirect", "close"):
                f.write(f"| {r['run']} | {r['persona_id']} | {r['category']} | {r['utterance']} | "
                        f"{r['pred_key']} | {r['reply']} |  |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N personas (quick trial)")
    ap.add_argument("--summarize", help="re-summarise an existing results file")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if a.summarize:
        rows = [json.loads(x) for x in open(a.summarize, encoding="utf-8")]
        stamp = os.path.basename(a.summarize).replace("run_", "").replace(".jsonl", "")
    else:
        personas = load_personas(a.split)
        if a.limit:
            personas = personas[:a.limit]
        print(f"{len(personas)} personas, {sum(len(p['turns']) for p in personas)} turns, {a.repeats} run(s)")
        engine.chat("Reply with OK.", "OK?", None, num_predict=5)        # warm-up: model load is not counted
        rows = run(personas, a.repeats, f"results/run_{stamp}.jsonl")
    text = summarize(rows)
    print("\n" + text)
    open(f"results/summary_{stamp}.txt", "w", encoding="utf-8").write(text)
    write_review(rows, f"results/review_{stamp}.md")
    print(f"\nsaved: results/run_{stamp}.jsonl  summary_{stamp}.txt  review_{stamp}.md")


if __name__ == "__main__":
    main()
