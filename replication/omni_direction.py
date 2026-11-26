"""The refusal direction on the omni model, per language (and form): is the direction shared? (RQ1)

arditi.py for Qwen2.5-Omni-7B's thinker. Same recipe as on the text models (Arditi et al.; "Nanda's direction"):
difference of means of the residual stream between harmful and harmless prompts at each layer and each of the
five template tokens, then Arditi's selection (the candidate that lowers refusal most when removed everywhere,
among those that pass his filters), then generation with the direction removed at every layer and token. The
score is the log odds of the first answer token being a refusal token; the refusal tokens are the first tokens of
that language's own refusal openings (from omni.py phrases), "I" when there are none.
The forward pass goes through the thinker, with omni.py's batching (prompts of one length per batch, no padding).

Subcommands:
  run      one language set and form: candidates, selection, the direction (outputs/arditi/<model>_omni_<form>_<tag>.pt
           and .json), then answers on the evaluation split with nothing changed, with the set's own direction
           removed, and with each --also direction removed (e.g. the English one: does a direction found in one
           language switch refusal off in another?). Answers go to outputs/evaluate/<model>_arditi_omni_<form>_<tag>.json.
  compare  no GPU: cosine similarity between the sets' directions, at each set's own selected place and at the
           English set's place in every set.
  align    Appendix G: how far each refusal neuron's write vector points along the direction, and its rank in its layer.
Usage:
  python replication/omni_direction.py run --tag poly_en --lang en --harmful poly_en_harmful_train ...
  python replication/omni_direction.py compare --tags poly_en poly_de poly_ar
Needs the refomni environment. Logs carry metrics only; answers are never printed.
"""
import argparse
import json
import math
import time

import torch

import omni
from arditi import intervention, kl_score, refusal_score, rejection_reasons, unit_direction
from neuron_check import ROOT, is_refusal, load

N_POST = len(omni.SCORE_TOKENS)


def make_batches(model, proc, form, prompts, bs):
    """[(indices, batch on the model's device)], prompts of one length per batch."""
    return [(idx, omni.unpadded(model, proc, form, prompts, idx)) for idx in omni.batches(proc, form, prompts, bs)]


@torch.no_grad()
def last_logits(model, batch):
    return model(**batch, use_cache=False).logits[:, -1].float().cpu()


@torch.no_grad()
def scores(model, batches, token_ids, n):
    """Refusal score per prompt, in the prompts' original order."""
    out = torch.zeros(n, dtype=torch.float64)
    for idx, batch in batches:
        out[idx] = refusal_score(last_logits(model, batch), token_ids)
    return out


@torch.no_grad()
def mean_activations(model, batches):
    blocks = model.model.layers
    n = sum(len(idx) for idx, _ in batches)
    means = torch.zeros(N_POST, len(blocks), model.config.text_config.hidden_size, dtype=torch.float64)
    handles = []

    def collector(layer):
        def hook(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            means[:, layer].add_(h[:, -N_POST:].detach().to(device="cpu", dtype=torch.float64).sum(dim=0) / n)
        return hook

    try:
        for i, block in enumerate(blocks):
            handles.append(block.register_forward_pre_hook(collector(i), with_kwargs=True))
        for _, batch in batches:
            model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if not torch.isfinite(means).all():
        raise FloatingPointError("Nonfinite residual means; try --dtype float32")
    return means


@torch.no_grad()
def select_direction(model, candidates, harmful, harmless, token_ids, n_h, n_s):
    """arditi.select_direction with the thinker's forward: lowest bypass score among candidates that pass."""
    n_pos, n_layers, _ = candidates.shape
    base_logits = [last_logits(model, batch) for _, batch in harmless]
    baseline = {"harmful": float(scores(model, harmful, token_ids, n_h).mean()),
                "harmless": float(torch.cat([refusal_score(x, token_ids) for x in base_logits]).mean())}
    records, started = [], time.time()
    for pos in range(-n_pos, 0):
        for layer in range(n_layers):
            direction = candidates[pos, layer]
            record = {"pos": pos, "post_token_index": pos + n_pos, "layer": layer}
            if direction.norm() == 0:
                record.update(bypass_score=None, induce_score=None, kl_score=None,
                              rejection_reasons=["zero_direction"], passes_filter=False)
            else:
                with intervention(model, direction):
                    bypass = float(scores(model, harmful, token_ids, n_h).mean())
                    kl = float(torch.cat([kl_score(base, last_logits(model, batch))
                                          for base, (_, batch) in zip(base_logits, harmless)]).mean())
                with intervention(model, direction, layer=layer):
                    induce = float(scores(model, harmless, token_ids, n_s).mean())
                reasons = rejection_reasons(bypass, induce, kl, layer, n_layers)
                record.update(bypass_score=bypass if math.isfinite(bypass) else None,
                              induce_score=induce if math.isfinite(induce) else None,
                              kl_score=kl if math.isfinite(kl) else None,
                              rejection_reasons=reasons, passes_filter=not reasons)
            records.append(record)
            if len(records) % 20 == 0 or len(records) == n_pos * n_layers:
                print(f"selection: {len(records)}/{n_pos * n_layers} candidates ({(time.time() - started) / 60:.1f} min)", flush=True)
    valid = sorted((r for r in records if r["passes_filter"]), key=lambda r: r["bypass_score"])
    for r in valid[:5]:
        print(f"  candidate token [{r['post_token_index']}] layer {r['layer']:2d}: bypass {r['bypass_score']:+.3f}  "
              f"induce {r['induce_score']:+.3f}  KL {r['kl_score']:.4f}", flush=True)
    if valid:
        return valid[0], records, baseline
    # Fallback, a deviation from Arditi's selection (German, 26 November: all 140 candidates fail the induce check,
    # so adding the direction to harmless prompts never makes one of that language's own refusal tokens the likely
    # first token, although removing it lowers the refusal score as elsewhere). Keep the KL and layer filters, drop
    # the induce check, and say so in the saved record.
    relaxed = sorted((r for r in records if r["rejection_reasons"] == ["induce_below_0"]), key=lambda r: r["bypass_score"])
    if not relaxed:
        return None, records, baseline
    chosen = dict(relaxed[0], fallback="induce check dropped: no candidate passed all of Arditi's filters")
    print(f"  NO candidate passes all filters; induce check dropped. token [{chosen['post_token_index']}] layer "
          f"{chosen['layer']}: bypass {chosen['bypass_score']:+.3f}  induce {chosen['induce_score']:+.3f}  "
          f"KL {chosen['kl_score']:.4f}", flush=True)
    return chosen, records, baseline


def refusal_token_ids(proc, phrases_json, key):
    """First tokens of the set's own refusal openings; 'I' when it has none."""
    tok = proc.tokenizer
    ids = []
    if phrases_json and (ROOT / phrases_json).exists():
        entry = json.loads((ROOT / phrases_json).read_text())["splits"].get(key)
        if entry:
            ids = [tok(o, add_special_tokens=False).input_ids[0] for o, _ in entry["top"] if o]
    ids = sorted(set(ids)) or [tok("I", add_special_tokens=False).input_ids[0]]
    return ids


def cmd_run(args):
    t0 = time.time()
    omni.LANG, omni.SYSTEM = args.lang, args.system
    model, proc = omni.load_omni(args.model, args.dtype)
    model.requires_grad_(False)
    short = args.model.split("/")[-1]
    stem = f"{short}_omni_{args.form}_{args.tag}"
    token_ids = refusal_token_ids(proc, args.phrases_json, args.harmful)
    print(f"{args.tag} ({args.form}): refusal tokens {[proc.tokenizer.decode([i]) for i in token_ids]}", flush=True)
    splits = {"harmful_train": load(args.harmful)[:args.n_train or None], "harmless_train": load(args.harmless)[:args.n_train or None],
              "harmful_val": load(args.harmful_val)[:args.n_val or None],
              "harmless_val": load(args.harmless_val)[:args.n_val or None]}
    kept, filtering = {}, {}
    for name, prompts in splits.items():
        sc = scores(model, make_batches(model, proc, args.form, prompts, args.bs), token_ids, len(prompts))
        keep = (sc > 0) if name.startswith("harmful") else (sc < 0)
        idx = keep.nonzero().flatten().tolist()
        filtering[name] = {"before": len(prompts), "after": len(idx), "mean_refusal_score": float(sc.mean())}
        print(f"{name}: kept {len(idx)}/{len(prompts)} (mean refusal score {sc.mean().item():+.2f})", flush=True)
        if not idx:
            raise SystemExit(f"{args.tag}: filtering emptied {name}; no direction (the model rarely refuses here?)")
        kept[name] = [prompts[i] for i in idx]
    b = {name: make_batches(model, proc, args.form, prompts, args.bs) for name, prompts in kept.items()}
    candidates = mean_activations(model, b["harmful_train"]) - mean_activations(model, b["harmless_train"])
    print(f"candidate directions {list(candidates.shape)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    selected, records, baseline = select_direction(model, candidates, b["harmful_val"], b["harmless_val"], token_ids,
                                                   len(kept["harmful_val"]), len(kept["harmless_val"]))
    out_dir = ROOT / "outputs" / "arditi"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"args": {k: v for k, v in vars(args).items() if k != "fn"}, "model": args.model, "form": args.form,
            "refusal_token_ids": token_ids, "post_tokens": omni.SCORE_TOKENS, "filtering": filtering,
            "baseline_val_refusal_scores": baseline, "candidates": records, "selected": selected}
    tensors = {"candidate_directions": candidates, "post_tokens": omni.SCORE_TOKENS, "selected": selected}
    if selected is not None:
        direction = candidates[selected["pos"], selected["layer"]].clone()
        tensors.update(direction=direction, unit_direction=unit_direction(direction))
        print(f"selected: token [{selected['post_token_index']}] layer {selected['layer']}  bypass "
              f"{selected['bypass_score']:+.3f} (unchanged {baseline['harmful']:+.3f})", flush=True)
    torch.save(tensors, out_dir / f"{stem}.pt")
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved outputs/arditi/{stem}.pt and .json ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if selected is None:
        raise SystemExit(f"{args.tag}: every candidate failed Arditi's filters; no direction")
    if args.no_eval:
        return
    del b
    raw = load(args.eval_split)[:args.n or None]
    runs = [("none", None, None), ("own", direction, selected["layer"])]
    for item in args.also:
        name, path = item.split("=", 1)
        other = torch.load(ROOT / path, map_location="cpu")
        if other.get("direction") is None:
            print(f"--also {name}: no direction in {path}; skipped")
            continue
        runs.append((f"from_{name}", other["direction"], other["selected"]["layer"]))
    eval_dir = ROOT / "outputs" / "evaluate"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"{short}_arditi_omni_{args.form}_{args.tag}.json"
    result = {"args": meta["args"], "model": args.model, "form": args.form, "prompts": raw, "selected": selected, "runs": []}
    for mode, vec, layer in runs:
        with intervention(model, vec):  # None = nothing changed; a vector = removed at every layer and token
            texts = omni.generate(model, proc, [], args.form, raw, args.bs, args.max_new)
        rr = sum(map(is_refusal, texts)) / len(texts)
        result["runs"].append({"layer": layer, "neuron": None, "pin": None, "mode": mode, "k": None,
                               "refusal_rate_stringmatch": rr, "responses": texts})
        eval_path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{mode}: string-match refusal {rr:.2f}, garbled {omni.garbled(texts):.2f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"saved outputs/evaluate/{eval_path.name} ({(time.time() - t0) / 60:.1f} min)")


def cmd_compare(args):
    short = args.model.split("/")[-1]
    data = {}
    for tag in args.tags:
        f = ROOT / "outputs" / "arditi" / f"{short}_omni_{args.form}_{tag}.pt"
        if f.exists():
            data[tag] = torch.load(f, map_location="cpu")
    have = [t for t in args.tags if t in data and data[t].get("direction") is not None]
    print(f"form {args.form}: directions for {have}; none for {[t for t in args.tags if t not in have]}")
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=0))
    for t in have:
        s = data[t]["selected"]
        print(f"  {t}: selected token [{s['post_token_index']}] layer {s['layer']}  bypass {s['bypass_score']:+.3f}")
    ref = args.ref if args.ref in have else (have[0] if have else None)
    if ref is None:
        return
    rs = data[ref]["selected"]
    print(f"\ncosine with the {ref} direction: each set's own selected direction | each set's candidate at {ref}'s place "
          f"(token [{rs['post_token_index']}], layer {rs['layer']})")
    for t in args.tags:
        if t not in data:
            continue
        own = f"{cos(data[t]['direction'], data[ref]['direction']):+.3f}" if t in have else "   —  "
        same = cos(data[t]["candidate_directions"][rs["pos"], rs["layer"]], data[ref]["direction"])
        print(f"  {t:10} {own} | {same:+.3f}")
    print("\npairwise cosine of the candidates at that same place (rows and columns in the order above):")
    tags = [t for t in args.tags if t in data]
    vecs = [data[t]["candidate_directions"][rs["pos"], rs["layer"]] for t in tags]
    for t, a in zip(tags, vecs):
        print(f"  {t:10} " + " ".join(f"{cos(a, b):+.2f}" for b in vecs))
    off = [cos(a, b) for i, a in enumerate(vecs) for j, b in enumerate(vecs) if i < j]
    if off:
        print(f"\noff-diagonal: mean {sum(off) / len(off):+.3f}, lowest {min(off):+.3f}, highest {max(off):+.3f}")


def cmd_align(args):
    """Kazemi et al. Appendix G on the thinker: how far each watched neuron's write vector (its column of
    down_proj) points along the refusal direction, and its rank by |cosine| among the 18,944 neurons of its layer.
    Three versions of the direction: the set's selected one, and the difference of means right after the neuron's
    own layer (candidate layer + 1 is the residual stream that layer writes into) at the last and first newline.
    'along' = the cosine times the sign of the neuron's harmful-minus-harmless value: positive means that when the
    neuron takes its harmful value it pushes the residual stream towards refusal."""
    from arditi import cosine_ranking, gaussian_alignment_null, write_cosines
    short = args.model.split("/")[-1]
    data = torch.load(ROOT / "outputs" / "arditi" / f"{short}_omni_{args.form}_{args.tag}.pt", map_location="cpu")
    search = json.loads((ROOT / args.search).read_text())
    gap = {(c["layer"], c["neuron"]): c["mean_act_harmful"] - c["mean_act_harmless"]
           for c in search["candidates"] + search.get("watched", []) if "mean_act_harmful" in c}
    model, _ = omni.load_omni(args.model, args.dtype)
    cand, sel = data["candidate_directions"], data["selected"]
    out = {"args": {k: v for k, v in vars(args).items() if k != "fn"}, "selected": sel, "neurons": []}
    print(f"{args.tag}: selected direction at token [{sel['post_token_index']}] layer {sel['layer']}")
    for item in args.watch:
        layer, neuron = (int(x) for x in item.split(":"))
        weight = model.model.layers[layer].mlp.down_proj.weight
        d = gap.get((layer, neuron))
        sign = 0.0 if d is None else math.copysign(1.0, d)
        null = gaussian_alignment_null(0.0, *weight.shape)
        row = {"layer": layer, "neuron": neuron, "d": d, "expected_max_abs_cosine": null["expected_max"], "versions": {}}
        print(f"L{layer}:F{neuron}  (harmful minus harmless value {d if d is None else round(d, 2)}; largest |cosine| "
              f"expected by chance in this layer {null['expected_max']:.3f})")
        versions = [("selected", data["direction"])]
        if layer + 1 < cand.shape[1]:
            versions += [("after own layer, last newline", cand[-1, layer + 1]), ("after own layer, first newline", cand[1, layer + 1])]
        for name, vec in versions:
            cosines, _ = write_cosines(weight, vec)
            top, watched = cosine_ranking(cosines, neuron)
            c = float(cosines[neuron])
            rank = int((cosines.abs() > abs(c)).sum()) + 1
            row["versions"][name] = {"cosine": c, "along": sign * c, "rank": rank, "top5": top}
            print(f"  {name:32} cosine {c:+.3f}  along {sign * c:+.3f}  rank {rank:5d} of {weight.shape[1]}   "
                  f"layer's top: F{top[0]['neuron']} {top[0]['cosine']:+.3f}, F{top[1]['neuron']} {top[1]['cosine']:+.3f}, F{top[2]['neuron']} {top[2]['cosine']:+.3f}")
        out["neurons"].append(row)
    path = ROOT / "outputs" / "arditi" / f"{short}_omni_{args.form}_{args.tag}_align.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved outputs/arditi/{path.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=omni.MODEL)
    common.add_argument("--form", default="text", choices=omni.FORMS)
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--tag", required=True, help="name of the language set, e.g. poly_de")
    r.add_argument("--lang", default=None)
    r.add_argument("--system", default=None, choices=sorted(omni.SYSTEMS))
    r.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    r.add_argument("--harmful", default="arditi_harmful_train")
    r.add_argument("--harmless", default="arditi_harmless_train")
    r.add_argument("--harmful_val", default="arditi_harmful_val")
    r.add_argument("--harmless_val", default="arditi_harmless_val")
    r.add_argument("--n_val", type=int, default=0, help="first n prompts of each validation split (0 = all)")
    r.add_argument("--n_train", type=int, default=0, help="first n prompts of each training split (0 = all; for a quick test)")
    r.add_argument("--phrases_json", default="outputs/refusal_phrases/Qwen2.5-Omni-7B_omni.json",
                   help="omni.py phrases output; the entry for --harmful gives the refusal tokens")
    r.add_argument("--eval_split", default="jailbreakbench")
    r.add_argument("--n", type=int, default=0, help="first n evaluation prompts (0 = all)")
    r.add_argument("--max_new", type=int, default=512)
    r.add_argument("--bs", type=int, default=8)
    r.add_argument("--no_eval", action="store_true")
    r.add_argument("--also", nargs="*", default=[], metavar="NAME=PATH.pt",
                   help="other sets' direction files to remove on this set's evaluation prompts (transfer)")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare", parents=[common])
    c.add_argument("--tags", nargs="+", required=True)
    c.add_argument("--ref", default="poly_en")
    c.set_defaults(fn=cmd_compare)
    a = sub.add_parser("align", parents=[common])
    a.add_argument("--tag", default="poly_en")
    a.add_argument("--dtype", default="float16")
    a.add_argument("--search", default="outputs/neuron_search/Qwen2.5-Omni-7B_float16_omni_text_poly_en.json")
    a.add_argument("--watch", nargs="+", default=["13:4787", "14:13014", "16:1870", "14:13435", "16:11620", "17:5484"])
    a.set_defaults(fn=cmd_align)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
