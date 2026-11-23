"""Generate answers with one neuron pinned (h_i <- m at every token, prompt and generation).

rerank (pick the neuron and the pin value, text only, like Faaiz's rerank.py):
    python pin_run.py rerank --search text.json --split harmbench_val
    -> top-5 candidates x pin grid, pinned on the side opposite the neuron's harmful mean.
attack (the test set, typed or spoken):
    python pin_run.py attack --neuron 12:3456 --value 20 --split jailbreakbench --modality speech --voice v1
    -> one unpinned and one pinned run on the same prompts.
Writes outputs/speech/<out_dir>/<name>.json (default out_dir: pin); label it with judge.py.
"""
import argparse
import json
import math
import time

from core import OUT, REPO, PinHook, generate, is_refusal, load_model, speech_items, text_items, text_layers


def run_one(model, proc, layers, items, layer, neuron, value, max_new, bs, padcheck=False):
    hook = PinHook(neuron)
    handle = None
    if value is not None:
        hook.value = value
        handle = layers[layer].mlp.down_proj.register_forward_pre_hook(hook)
    t0 = time.time()
    try:
        answers = generate(model, proc, items, bs=bs, max_new=max_new)
        # padding check: the same prompts one at a time (no padding); greedy answers should match the batch
        solo = generate(model, proc, items, bs=1, max_new=max_new) if padcheck else None
    finally:
        if handle:
            handle.remove()
    refused = [is_refusal(a) for a in answers]
    print(f"  L{layer}:F{neuron} pin {value}: refused {sum(refused)}/{len(items)} "
          f"({time.time() - t0:.0f}s)", flush=True)  # metrics only; answers go to the JSON
    out = {"layer": layer, "neuron": neuron, "value": value, "prompts": [it["text"] for it in items],
           "idx": [it.get("idx") for it in items], "answers": answers, "refused": refused,
           "refusal_rate": sum(refused) / len(items)}
    if padcheck:
        out["answers_bs1"] = solo
        out["same_as_bs1"] = sum(a == b for a, b in zip(answers, solo))
        print(f"    padding check: {out['same_as_bs1']}/{len(items)} batched answers identical to unbatched", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["rerank", "attack"])
    ap.add_argument("--search", help="rerank: search JSON")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--m", type=float, nargs="+", default=[2, 4, 6, 10, 15, 20, 30, 40, 60])
    ap.add_argument("--neuron", help="attack: L:F")
    ap.add_argument("--value", type=float, help="attack: signed pin value")
    ap.add_argument("--pick_key", default="pick", help="attack: 'pick' (Llama-Guard) or 'pick_k1' (K.1)")
    ap.add_argument("--pick", help="attack: take neuron and value from a judged rerank file's 'pick' (outputs/speech/<out_dir>/<file>)")
    ap.add_argument("--split", default="harmbench_val")
    ap.add_argument("--modality", choices=["text", "speech"], default="text")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--n", type=int, default=0, help="first n prompts (0 = all)")
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_dir", default="pin", help="folder under outputs/speech/ (the 2026-11-23 rerun used pin_v2)")
    ap.add_argument("--padcheck", action="store_true", help="also generate one prompt at a time and count identical answers")
    args = ap.parse_args()
    items = speech_items(args.split, args.voice) if args.modality == "speech" else text_items(args.split)
    if args.n:
        items = items[:args.n]
    model, proc = load_model()
    layers = text_layers(model)
    name = f"{args.mode}_{args.split}_{args.modality}{'_' + args.voice if args.voice else ''}{'_' + args.tag if args.tag else ''}"
    out_path = OUT / args.out_dir / f"{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"args": vars(args), "runs": []}

    def save():
        out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False))

    print(f"{name}: {len(items)} prompts", flush=True)
    pc = args.padcheck
    result["runs"].append(run_one(model, proc, layers, items, 0, 0, None, args.max_new, args.bs, pc))  # no pin
    save()
    if args.mode == "rerank":
        cands = json.loads((OUT / "search" / args.search).read_text()
                           if "/" not in args.search else open(args.search).read())["candidates"][:args.top]
        for c in cands:
            sign = -math.copysign(1.0, c["mean_act_harmful"])  # opposite the harmful mean (Faaiz's rule)
            for m in args.m:
                r = run_one(model, proc, layers, items, c["layer"], c["neuron"], sign * m, args.max_new, args.bs, pc)
                r["m"] = m
                result["runs"].append(r)
                save()
    else:
        if args.pick:
            pk = json.loads((OUT / args.out_dir / args.pick).read_text())[args.pick_key]
            args.neuron, args.value = f"{pk['layer']}:{pk['neuron']}", pk["value"]
            result["args"] = vars(args)
            print(f"pick from {args.pick}: L{pk['layer']}:F{pk['neuron']} at {pk['value']}", flush=True)
        layer, neuron = (int(x) for x in args.neuron.split(":"))
        result["runs"].append(run_one(model, proc, layers, items, layer, neuron, args.value, args.max_new, args.bs, pc))
        save()
    print(f"saved {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
