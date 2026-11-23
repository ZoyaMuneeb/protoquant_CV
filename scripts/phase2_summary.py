"""Collect Phase 2 results into results/phase2_results.{md,json}, with the fixed decision rules.

Comparison (docs/phase0_protocol.md, section 6): a method beats the reference at a given K if the 95 %
paired-bootstrap interval of its test-accuracy difference lies above zero AND the difference has the same
sign in all three seeds. "No meaningful loss" = lower bound above -1 point. Reference: B1 for one-shot
masking, B2 (= B1 retrained) for retrained models.

Bootstrap: for each test image, correctness is averaged over the seeds for each method; the per-image
difference (method - reference) is resampled over images, 10,000 times; the interval is the 2.5 / 97.5
percentiles of the mean difference (in points). Same images, paired predictions for all methods.
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

N_BOOT = 10_000
ONE_SHOT = ("B1", "random", "P1", "P1contrast", "P3", "P3nomin")


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.is_file() else {}


def bootstrap(diff: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    means = diff[idx].mean(1) * 100
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(lo: float, seed_diffs: list[float]) -> str:
    if lo > 0 and all(d > 0 for d in seed_diffs):
        return "**better**"
    if lo > -1:
        return "no meaningful loss"
    return "loss not excluded"


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    nd = 0 if all(isinstance(v, int) for v in vals) else 2  # counts vs accuracies
    if len(vals) == 1:
        return f"{vals[0]:,.{nd}f}"
    return f"{statistics.mean(vals):,.{nd}f} ± {statistics.stdev(vals):,.{nd}f}"


def cells(c: dict | None) -> str:
    """Δ, CI, per-seed Δ and verdict cells for a comparison (or the reference row)."""
    if not c:
        return "ref |  |  | "
    lo, hi = c["ci95"]
    seeds = ", ".join(f"{d:+.2f}" for d in c["seed_diffs"])
    return f"{c['diff_mean']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {seeds} | {c['verdict']}"


def compare(correct: dict, keys: list[str], ref: str, seeds: list[int]) -> dict:
    """correct[(seed, key)] = bool array over test images (same order)."""
    out = {}
    for key in keys:
        have = [s for s in seeds if (s, key) in correct and (s, ref) in correct]
        if not have or key == ref:
            continue
        m = np.mean([correct[(s, key)] for s in have], axis=0)
        r = np.mean([correct[(s, ref)] for s in have], axis=0)
        lo, hi = bootstrap(m - r)
        sd = [float((correct[(s, key)].mean() - correct[(s, ref)].mean()) * 100) for s in have]
        out[key] = {"diff_mean": float((m - r).mean() * 100), "ci95": [lo, hi], "seed_diffs": sd,
                    "seeds": have, "verdict": verdict(lo, sd)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2_root", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    root = Path(args.phase2_root)

    rows, correct, retrain_rows, iter_rows, budget_rows, deletion, ref_ids = {}, {}, {}, {}, {}, {}, None
    seeds = sorted(int(d.name.split("_s")[1]) for d in root.glob("strict_s*") if d.is_dir())
    for s in seeds:
        d = root / f"strict_s{s}"
        # Phase 2 one-shot masks, then the Phase 4 one-shot ablation (P3 without the minimum)
        for res_name, npz_name in (("results.json", "test_preds.npz"), ("ablation.json", "ablation_test_preds.npz"),
                                   ("codeprune.json", "codeprune_test_preds.npz"),
                                   ("codeprune_matched.json", "codeprune_matched_test_preds.npz"),
                                   ("repair.json", "repair_test_preds.npz"),
                                   ("repair_cesweep.json", "repair_cesweep_test_preds.npz"),
                                   ("repair_cachedrt.json", "repair_cachedrt_test_preds.npz")):
            res = load(d / "masking" / res_name)
            for r in res.get("models", []):
                key = "dense" if r["method"] == "dense" else f"{r['method']}_K{r['K']}"
                rows[(s, key)] = r
            npz = d / "masking" / npz_name
            if npz.is_file():
                z = np.load(npz)
                ids = z["img_ids"]
                ref_ids = ids if ref_ids is None else ref_ids
                if not np.array_equal(ids, ref_ids):
                    raise SystemExit(f"test image order differs in {npz}")
                for key in z.files:
                    if key not in ("img_ids", "labels"):
                        correct[(s, key)] = z[key] == z["labels"]
        dl = load(d / "masking" / "deletion.json")
        if dl:
            deletion[s] = dl["spearman"]
        for rd in sorted((d / "retrain").glob("*_K*")):
            rj, et, ev = load(rd / "retrain.json"), load(rd / "eval_test.json"), load(rd / "eval_val.json")
            if not et:
                continue
            key = "R_" + rd.name
            retrain_rows[(s, key)] = {**rj, "test_top1": et["top1"], "val_top1": ev.get("top1"),
                                      "nonzero_edges": et.get("nonzero_edges"), "codes": et.get("codebook_entries")}
            z = np.load(rd / "eval_test.npz")
            ref_ids = z["img_ids"] if ref_ids is None else ref_ids
            if not np.array_equal(z["img_ids"], ref_ids):
                raise SystemExit(f"test image order differs in {rd}")
            correct[(s, key)] = z["preds"] == z["labels"]
        for rd in sorted((d / "iterative").glob("*_K*")):
            fin = rd / "final"
            it, rj, et = load(rd / "iterative.json"), load(fin / "retrain.json"), load(fin / "eval_test.json")
            if not et:
                continue
            key = "IT_" + rd.name
            iter_rows[(s, key)] = {"rounds": it.get("rounds"), "total_epochs": it.get("round_training_epochs", 0) + rj.get("epochs", 0),
                                   "best_val_top1": rj.get("best_val_top1"), "test_top1": et["top1"],
                                   "codes": et.get("codebook_entries"), "nonzero_edges": et.get("nonzero_edges")}
            z = np.load(fin / "eval_test.npz")
            ref_ids = z["img_ids"] if ref_ids is None else ref_ids
            if not np.array_equal(z["img_ids"], ref_ids):
                raise SystemExit(f"test image order differs in {fin}")
            correct[(s, key)] = z["preds"] == z["labels"]
        for rd in sorted((d / "budget").glob("*_K*_E*")):   # epoch-matched one-shot control (protocol 10)
            rj, et = load(rd / "retrain.json"), load(rd / "eval_test.json")
            if not et:
                continue
            key = "BUD_" + rd.name
            budget_rows[(s, key)] = {"total_epochs": rj.get("epochs"), "best_val_top1": rj.get("best_val_top1"),
                                     "test_top1": et["top1"], "codes": et.get("codebook_entries")}
            z = np.load(rd / "eval_test.npz")
            ref_ids = z["img_ids"] if ref_ids is None else ref_ids
            if not np.array_equal(z["img_ids"], ref_ids):
                raise SystemExit(f"test image order differs in {rd}")
            correct[(s, key)] = z["preds"] == z["labels"]

    ks =sorted({r["K"] for (s, k), r in rows.items() if r.get("K")})
    md = ["# Phase 2 results", "",
          f"Seeds: {', '.join(map(str, seeds))}. Test top-1 is mean ± sd over seeds. Δ = method − reference "
          "(points), with the 95 % paired-bootstrap interval over test images (10,000 resamples, correctness "
          "averaged over seeds). Verdict rules fixed in Phase 0: better = interval above 0 and same sign in every "
          "seed; no meaningful loss = lower bound above −1; otherwise a loss of 1 point or more is not excluded.", ""]
    dense = [rows[(s, "dense")]["test_top1"] for s in seeds if (s, "dense") in rows]
    if dense:
        md += [f"Dense head (no pruning): test {ms(dense)} %, "
               f"{statistics.mean(rows[(s, 'dense')]['nonzero_edges'] for s in seeds if (s, 'dense') in rows):,.0f} "
               "nonzero edges.", ""]
    summary = {"seeds": seeds, "one_shot": {}, "retrained": {}, "deletion": deletion}

    md += ["## One-shot masking (no retraining)", ""]
    for k in ks:
        keys = [f"{m}_K{k}" for m in ONE_SHOT]
        cmp_ = compare(correct, keys, f"B1_K{k}", seeds)
        md += [f"### K = {k} edges per class (N = {k * 200:,})", "",
               "| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |",
               "|---|---|---|---|---|---|---|---|---|"]
        for key in keys:
            rs = [rows[(s, key)] for s in seeds if (s, key) in rows]
            if not rs:
                continue
            c = cmp_.get(key)
            extra = (f" (classes with no edge: {ms([r.get('classes_with_no_edge') for r in rs])})" if key.startswith("P3nomin")
                     else f" (min-1 classes: {ms([r.get('classes_at_minimum') for r in rs])})" if key.startswith("P3") else "")
            md.append(
                f"| {key.rsplit('_K', 1)[0]}{extra} | {ms([r['val_top1'] for r in rs])} | {ms([r['test_top1'] for r in rs])} | "
                f"{cells(c)} | "
                f"{ms([r['nonzero_edges'] for r in rs])} | {ms([r['active_codes'] for r in rs])} |")
            summary["one_shot"][key] = {"rows": rs, "vs_B1": c}
        md.append("")

    rkeys = sorted({k for (s, k) in retrain_rows})
    if rkeys:
        md += ["## With retraining (codes no class uses are removed, then 30 epochs, mask held fixed)", "",
               "B2 = B1 mask retrained (the paper's Pruned & Tuned). Methods are compared with B2 at the same K.", "",
               "| Model | Codes | Val before → best | Test top-1 | Δ vs B2 | 95 % CI | Seeds (Δ) | Verdict |",
               "|---|---|---|---|---|---|---|---|"]
        for k in ks:
            ref = f"R_B1_K{k}"
            group = [r for r in rkeys if r.endswith(f"_K{k}")]
            cmp_ = compare(correct, group, ref, seeds)
            for key in sorted(group, key=lambda x: (x != ref, x)):
                rs = [retrain_rows[(s, key)] for s in seeds if (s, key) in retrain_rows]
                c = cmp_.get(key)
                name = key[2:].replace("B1_", "B2_")
                md.append(
                    f"| {name} | {ms([r['codes'] for r in rs])} | "
                    f"{ms([r['val_before_retraining'] for r in rs])} → {ms([r['best_val_top1'] for r in rs])} | "
                    f"{ms([r['test_top1'] for r in rs])} | {cells(c)} |")
                summary["retrained"][key] = {"rows": rs, "vs_B2": c}
        md.append("")

    ikeys = sorted({k for (s, k) in iter_rows})
    bkeys = sorted({k for (s, k) in budget_rows})
    if ikeys or bkeys:   # the epoch-matched control can exist without iterative runs (improved-backbone rerun)
        summary["iterative"] = {}
        md += ["## Iterative pruning and longer retraining (Phase 4)", "",
               "Iterative: 20 % of each class's edges removed per round, 5 epochs of retraining and rescoring between "
               "rounds, full codebook during the rounds; then unused codes removed and 30 epochs (the B2 recipe). "
               "Control: the one-shot B1 mask retrained with the B2 recipe for as many epochs in total. "
               "Same rules as above.", "",
               "| Model | Rounds | Total epochs | Codes | Best val | Test top-1 |", "|---|---|---|---|---|---|"]
        for key in ikeys:
            rs = [iter_rows[(s, key)] for s in seeds if (s, key) in iter_rows]
            md.append(f"| {key[3:]} iterative | {ms([r['rounds'] for r in rs])} | {ms([r['total_epochs'] for r in rs])} | "
                      f"{ms([r['codes'] for r in rs])} | {ms([r['best_val_top1'] for r in rs])} | "
                      f"{ms([r['test_top1'] for r in rs])} |")
            summary["iterative"][key] = {"rows": rs}
        for key in bkeys:
            rs = [budget_rows[(s, key)] for s in seeds if (s, key) in budget_rows]
            name = key[4:].split("_E")[0]
            name = (name.replace("B1_", "B2_") + " one-shot") if name.startswith("B1_") else (name + " code-level")
            md.append(f"| {name}, {ms([r['total_epochs'] for r in rs])} epochs{' (control)' if 'B2_' in name else ''} | - | "
                      f"{ms([r['total_epochs'] for r in rs])} | {ms([r['codes'] for r in rs])} | "
                      f"{ms([r['best_val_top1'] for r in rs])} | {ms([r['test_top1'] for r in rs])} |")
            summary["iterative"][key] = {"rows": rs}
        md += ["", "| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |", "|---|---|---|---|---|---|"]
        pairs = []
        for key in ikeys:
            score, k = key[3:].rsplit("_K", 1)
            ref = f"R_{score}_K{k}"
            pairs.append((key, ref, f"iterative {score} vs one-shot {'B2' if score == 'B1' else score} (retrained), K = {k}",
                          "does gradual pruning help?"))
        for key in ikeys:
            score, k = key[3:].rsplit("_K", 1)
            if score != "B1" and f"IT_B1_K{k}" in ikeys:
                pairs.append((key, f"IT_B1_K{k}", f"iterative {score} vs iterative B1, K = {k}",
                              "does the score matter under repeated pruning?"))
        for bkey in bkeys:
            base, e = bkey[4:].rsplit("_E", 1)
            score, k = base.rsplit("_K", 1)
            for ikey in ikeys:   # every iterative model at this K vs the epoch-matched one-shot B2
                iscore, ik = ikey[3:].rsplit("_K", 1)
                if ik == k:
                    pairs.append((ikey, bkey, f"iterative {iscore} vs one-shot B2 at {e} epochs, K = {k}",
                                  "gradual pruning, at equal training?" if iscore == score
                                  else "does iterative P1 beat the simplest strong baseline?"))
            label = "one-shot B2" if score == "B1" else f"code-level {score}"
            if score == "B1":
                pairs.append((bkey, f"R_{base}", f"one-shot B2 at {e} vs 30 epochs, K = {k}",
                              "does longer training alone help?"))
            else:   # code-level pruning vs the edge-level B2 control at the same training budget
                pairs.append((bkey, f"BUD_B1_K{k}_E{e}", f"{label} vs one-shot B2, both {e} epochs, K = {k}",
                              "fewer codes at no loss?"))
            if score.startswith("CB1-eqCP1M"):   # code-matched control: CP1 vs CB1 at equal active codes
                tgt = "CP1-M" + score.split("CP1M", 1)[1]
                pairs.append((f"BUD_{tgt}_K{k}_E{e}", bkey,
                              f"code-level {tgt} vs CB1 with as many codes, both {e} epochs, K = {k}",
                              "does the code score matter after retraining?"))
            pairs.append((bkey, "dense", f"{label} at {e} epochs (K = {k}) vs dense head",
                          "is dense accuracy kept?"))
        for key in ikeys:
            score, k = key[3:].rsplit("_K", 1)
            pairs.append((key, "dense", f"iterative {score} (K = {k}) vs dense head", "is dense accuracy kept?"))
        for key, ref, label, q in pairs:
            c = compare(correct, [key], ref, seeds).get(key)
            if c:
                md.append(f"| {label} | {q} | {cells(c)} |")
                summary["iterative"].setdefault(key, {})[f"vs_{ref}"] = c
        md.append("")

    ckeys = sorted({k for (s_, k) in rows if k.startswith(("CB1-", "CP1-")) and "+" not in k},   # "+": repaired
                   key=lambda x: (x.split("-")[0], -int(x.split("M")[-1].split("_K")[0]), x))
    if ckeys:
        md += ["## Code-level pruning (one-shot, section 12)", "",
               "The M best codes by summed score (CB1: magnitude, CP1: activation-aware), then the top K edges of "
               "each class among them. Δ vs edge-level B1 at the same K (which keeps the codes its edges touch).", "",
               "| Model | Codes kept (active) | Edges | Classes < K | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        summary["code_level"] = {}
        for key in ckeys:
            rs = [rows[(s_, key)] for s_ in seeds if (s_, key) in rows]
            k = key.rsplit("_K", 1)[1]
            c = compare(correct, [key], f"B1_K{k}", seeds).get(key)
            md.append(f"| {key} | {ms([r['active_codes'] for r in rs])} | {ms([r['nonzero_edges'] for r in rs])} | "
                      f"{ms([r['classes_below_k'] for r in rs])} | {ms([r['val_top1'] for r in rs])} | "
                      f"{ms([r['test_top1'] for r in rs])} | {cells(c)} |")
            summary["code_level"][key] = {"rows": rs, "vs_B1": c}
        md.append("")
        cmp_rows = []
        for key in ckeys:
            if key.startswith("CP1-"):
                ref = "CB1-" + key[4:]
                c = compare(correct, [key], ref, seeds).get(key)
                if c:
                    cmp_rows.append(f"| {key} vs {ref} | does the activation-aware code score help? | {cells(c)} |")
                    summary["code_level"][key]["vs_CB1"] = c
            elif key.startswith("CB1-eqCP1M"):
                tgt = "CP1-M" + key.split("CP1M", 1)[1]
                c = compare(correct, [tgt], key, seeds).get(tgt)
                if c:
                    cmp_rows.append(f"| {tgt} vs {key} | does the code score help at equal active codes? | {cells(c)} |")
                    summary["code_level"].setdefault(tgt, {})[f"vs_{key}"] = c
        if cmp_rows:
            md += ["| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |", "|---|---|---|---|---|---|",
                   *cmp_rows, ""]

    rep_bases = []   # closed-form weight repair (section 13): "<method>+<variant>_K<K>" from repair.json
    for (s_, k) in rows:
        base = rows[(s_, k)].get("base_key")
        if base and "+" in k and base not in rep_bases:
            rep_bases.append(base)
    if rep_bases:
        summary["repair"] = {}
        md += ["## Closed-form weight repair (no retraining, section 13)", "",
               "The surviving weights of a pruned head re-fitted on the cached training features, mask and w ≥ 0 kept. "
               "LS: per-class non-negative least squares against the dense model's logits. CE: the same weights "
               "re-fitted to the labels (projected Adam). phys: codes no class uses are removed first, so the softmax "
               "renormalises over the kept codes (as in retraining). CEsel: CE refit with the step count (300 / 1,000 / "
               "3,000 / 10,000) chosen per model on validation (post hoc); the chosen counts are in brackets. rtH / rtCH: "
               "the retraining recipe (145 epochs, best epoch on validation) run on the cached features, head only / "
               "codes + head (post hoc); best epochs in brackets. Δ vs the "
               "same mask one-shot (no repair, every code in the softmax).", "",
               "| Model | Codes in softmax | Edges | Val top-1 | Test top-1 | Δ vs one-shot | 95 % CI | Seeds (Δ) | Verdict | Fit (s) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        variants = ("phys", "LS", "CE", "phys+LS", "phys+CE", "CEsel", "phys+CEsel", "phys+rtH", "phys+rtCH")
        for base in rep_bases:
            m, k = base.rsplit("_K", 1)
            keys = [f"{m}+{v}_K{k}" for v in variants]
            cmp_ = compare(correct, keys, base, seeds)
            rs0 = [rows[(s_, base)] for s_ in seeds if (s_, base) in rows]
            if rs0:
                md.append(f"| {m} one-shot | {ms([r.get('active_codes') for r in rs0])} (all in softmax) | "
                          f"{ms([r['nonzero_edges'] for r in rs0])} | {ms([r['val_top1'] for r in rs0])} | "
                          f"{ms([r['test_top1'] for r in rs0])} | ref |  |  |  |  |")
            for key in keys:
                rs = [rows[(s_, key)] for s_ in seeds if (s_, key) in rows]
                if not rs:
                    continue
                c = cmp_.get(key)
                steps = [r["ce_steps_selected"] for r in rs if "ce_steps_selected" in r]
                ep = [r["best_epoch"] for r in rs if "best_epoch" in r]
                label = key.rsplit('_K', 1)[0] + (f" ({', '.join(map(str, steps))} steps)" if steps else "") + \
                    (f" (best epoch {', '.join(map(str, ep))})" if ep else "")
                md.append(f"| {label} | {ms([r['codebook_rows'] for r in rs])} | "
                          f"{ms([r['nonzero_edges'] for r in rs])} | {ms([r['val_top1'] for r in rs])} | "
                          f"{ms([r['test_top1'] for r in rs])} | {cells(c)} | {ms([r['repair_seconds'] for r in rs])} |")
                summary["repair"][key] = {"rows": rs, "vs_one_shot": c}
        md.append("")
        pairs = []
        retrained = sorted({k for (s_, k) in budget_rows} | {k for (s_, k) in retrain_rows})
        for base in rep_bases:
            m, k = base.rsplit("_K", 1)
            for v in ("phys+LS", "phys+CE", "phys+CEsel", "phys+rtH", "phys+rtCH"):
                key = f"{m}+{v}_K{k}"
                for ref in retrained:   # the same mask retrained (30 or 145 epochs, codes physically removed)
                    if ref == f"R_{base}" or ref.startswith(f"BUD_{base}_E"):
                        e = ref.rsplit("_E", 1)[1] if ref.startswith("BUD_") else "30"
                        pairs.append((key, ref, f"{m} {v} vs {m} retrained {e} epochs, K = {k}",
                                      "can repair replace retraining?"))
                pairs.append((key, "dense", f"{m} {v} (K = {k}) vs dense head", "is dense accuracy kept?"))
            pairs.append((f"{m}+phys+LS_K{k}", f"{m}+phys+CE_K{k}", f"{m}: LS (dense logits) vs CE (labels), physical",
                          "which repair target is better?"))
            pairs.append((f"{m}+phys+CEsel_K{k}", f"{m}+phys+CE_K{k}", f"{m}: CE, steps chosen on val vs 300 steps, physical",
                          "do more CE steps help?"))
            pairs.append((f"{m}+phys+rtCH_K{k}", f"{m}+phys+rtH_K{k}",
                          f"{m}: codes + head vs head only, retraining recipe on cached features",
                          "does adapting the codes help?"))
        for base in rep_bases:   # does the pruning score still matter once the weights are repaired?
            m, k = base.rsplit("_K", 1)
            ref = "B1" if m == "P1" else ("CB1-eqCP1M" + m[5:]) if m.startswith("CP1-M") else None
            if ref and f"{ref}_K{k}" in rep_bases:
                for v in ("LS", "CEsel", "rtH", "rtCH"):
                    pairs.append((f"{m}+phys+{v}_K{k}", f"{ref}+phys+{v}_K{k}", f"{m} vs {ref}, both physical + {v} repair",
                                  "does the score matter after repair?"))
        rep_rows = []
        for key, ref, label, q in pairs:
            c = compare(correct, [key], ref, seeds).get(key)
            if c:
                rep_rows.append(f"| {label} | {q} | {cells(c)} |")
                summary["repair"].setdefault(key, {})[f"vs_{ref}"] = c
        if rep_rows:
            md += ["| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |", "|---|---|---|---|---|---|",
                   *rep_rows, ""]

    abl = []   # Phase 4 ablations that need no training (methodology, Phase 4)
    for k in ks:
        abl.append((f"P1contrast_K{k}", f"P1_K{k}", f"P1-contrast vs P1, one-shot, K = {k}",
                    "does contrasting against other classes help?"))
    for k in sorted({int(k.rsplit("_K", 1)[1]) for (s_, k) in retrain_rows}):
        abl.append((f"R_P1contrast_K{k}", f"R_P1_K{k}", f"P1-contrast vs P1, retrained, K = {k}",
                    "does contrasting against other classes help?"))
    for k in ks:
        abl.append((f"P3nomin_K{k}", f"P3_K{k}", f"P3 without vs with the one-edge minimum, K = {k}",
                    "does the minimum help P3?"))
    abl_rows = []
    for key, ref, label, q in abl:
        c = compare(correct, [key], ref, seeds).get(key)
        if c:
            abl_rows.append(f"| {label} | {q} | {cells(c)} |")
            summary.setdefault("ablations", {})[f"{key}_vs_{ref}"] = c
    if abl_rows:
        md += ["## Ablations (Phase 4)", "", "Same rules as above; the reference is the second method named.", "",
               "| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |", "|---|---|---|---|---|---|",
               *abl_rows, ""]

    if deletion:
        md += ["## Deletion-effect control", "",
               "100 class–code edges (a third each from low, medium and high P1 scores) removed one at a time "
               "from the dense head; damage = change in validation cross-entropy. Spearman ρ between damage and "
               "each score (higher = the score predicts real damage better).", "",
               "| Seed | Magnitude (B1) | P1 | P1-contrast |", "|---|---|---|---|"]
        for s, c in sorted(deletion.items()):
            md.append(f"| {s} | {c['B1']['spearman_rho']:.3f} | {c['P1']['spearman_rho']:.3f} | "
                      f"{c['P1contrast']['spearman_rho']:.3f} |")
        md.append(f"| mean | {ms([c['B1']['spearman_rho'] for c in deletion.values()])} | "
                  f"{ms([c['P1']['spearman_rho'] for c in deletion.values()])} | "
                  f"{ms([c['P1contrast']['spearman_rho'] for c in deletion.values()])} |")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md) + "\n")
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))
    print("\n".join(md))
    print(f"\nwrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
