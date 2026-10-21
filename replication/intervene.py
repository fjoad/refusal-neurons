"""Shared single-neuron interventions: constant pin and the anchor variant (paper Eq. 6-7).

constant:  h_i <- m* at every token (prefill and generation).
anchor:    per prompt, read the neuron's natural activation over Table 3's tokens in a
           hook-free pass, v = max (d > 0) or min (d < 0); then pin
           h_i <- clamp(k * m* * v / d - d, m*)   (min if m* > 0, max if m* < 0),
           where d = a_harmful - a_harmless at the discovery token t*.
"""
import json

import torch

from neuron_check import ROOT, paper_score_tokens


class Pin:
    """Forward pre-hook on mlp.down_proj. `values`: None (off), a float, or one float per batch row."""

    def __init__(self, idx):
        self.idx, self.values, self.records = idx, None, []

    def __call__(self, module, args):
        h = args[0]
        if self.values is None:
            self.records.append(h[..., self.idx].detach().float().cpu())
            return None
        h = h.clone()
        v = torch.as_tensor(self.values, dtype=h.dtype, device=h.device)
        h[..., self.idx] = v.view(-1, 1) if v.ndim else v
        return (h,)


def neuron_gap(search_json, layer, neuron):
    """d = a_harmful - a_harmless at t*, from a neuron_search.py output (candidates or watched)."""
    d = json.loads((ROOT / search_json).read_text())
    for c in d["candidates"] + d.get("watched", []):
        if (c["layer"], c["neuron"]) == (layer, neuron) and "mean_act_harmful" in c:
            return c["mean_act_harmful"] - c["mean_act_harmless"]
    raise ValueError(f"L{layer}:F{neuron} not in {search_json}; rerun the search with --watch {layer}:{neuron}")


@torch.no_grad()
def anchor_values(model, tok, pin, prompts, post_tokens, m_star, d, k, bs):
    """Per-prompt anchor pin values (Eq. 6-7). `prompts` are chat-formatted strings."""
    n_post = len(post_tokens)
    n_paper = paper_score_tokens(tok.name_or_path, post_tokens)
    pin.values, out = None, []
    for i in range(0, len(prompts), bs):
        batch = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        pin.records.clear()
        model(**batch, logits_to_keep=1)  # only the hook's record is needed; skip full-vocab logits
        acts = pin.records[0][:, -n_post:][:, :n_paper]  # left padding: suffix is the last n_post tokens
        v = acts.max(1).values if d > 0 else acts.min(1).values
        h = k * m_star * v / d - d
        h = torch.minimum(h, torch.tensor(m_star)) if m_star > 0 else torch.maximum(h, torch.tensor(m_star))
        out += h.tolist()
    pin.records.clear()
    return out
