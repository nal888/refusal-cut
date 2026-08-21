#!/usr/bin/env python3
"""
abliterate.py — general, resource-aware abliteration for open-weight LLMs.

Removes the "refusal direction" from a model's weights (Arditi et al. 2024)
directly on plain HuggingFace `transformers`, so it works on most modern
decoder-only families (Llama, Qwen, Mistral, Gemma, Phi, Yi, DeepSeek-dense…)
with no per-model config.

Beyond the basic method it adds, inspired by Heretic (p-e-w/heretic):
  * SMART LAYER SELECTION — instead of "biggest layer", it searches candidate
    layers and picks the one that kills refusals with the LEAST damage
    (co-minimising refusal score + KL divergence from the original model).
  * A before/after REFUSAL NUMBER so you can see it actually worked.

And it is resource-aware:
  * skips the LM head during direction-finding (logits_to_keep) — big compute save
  * `low_cpu_mem_usage` load; in-place weight edit; chunked embedding edit
  * `--dry-run` to analyse without editing or saving

This is a research / red-teaming tool. Use it on models and data you're
authorised to. Requires open weights you can fit in memory; cannot touch closed
models (GPT/Claude/Gemini) or fit giant/MoE models on small hardware.

Usage
-----
  python abliterate.py --model Qwen/Qwen2.5-1.5B-Instruct --output ./out --test
  python abliterate.py --model <id> --output ./out --select auto --preserve-norms
  python abliterate.py --model <id> --dry-run          # analyse only, no save
"""
import argparse
import contextlib
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------- model-agnostic structure
_LAYER_PATHS = [
    ("model", "layers"), ("transformer", "h"),
    ("gpt_neox", "layers"), ("model", "decoder", "layers"),
]
_ATTN_OUT = {"o_proj", "out_proj", "wo", "dense"}
_MLP_OUT = {"down_proj", "fc2", "w2", "c_proj"}
_REFUSAL_OPENERS = ["I", "As", "Sorry", "I'm", "Cannot", "Unfortunately", "It"]


def _dig(obj, path):
    for p in path:
        if not hasattr(obj, p):
            return None
        obj = getattr(obj, p)
    return obj


def find_layers(model):
    for path in _LAYER_PATHS:
        layers = _dig(model, path)
        if layers is not None and hasattr(layers, "__len__") and len(layers):
            return list(layers)
    best = None
    for m in model.modules():
        if isinstance(m, torch.nn.ModuleList) and (best is None or len(m) > len(best)):
            best = m
    if best is None:
        raise RuntimeError("could not locate decoder layers for this model")
    return list(best)


def residual_writers(layer, hidden_size):
    """Attn-output + MLP-output Linear modules whose out-dim == hidden size.

    The hidden_size guard is what keeps this from grabbing the wrong matrix on
    an unusual architecture (e.g. gate/up projections write to intermediate dim,
    not the residual stream)."""
    found = {}
    for name, mod in layer.named_modules():
        short = name.split(".")[-1]
        if not isinstance(mod, torch.nn.Linear) or mod.out_features != hidden_size:
            continue
        if short in _ATTN_OUT and "attn" not in found:
            found["attn"] = mod
        elif short in _MLP_OUT and "mlp" not in found:
            found["mlp"] = mod
    return list(found.values())


def detection_report(model):
    cfg = model.config
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    layers = find_layers(model)
    sample = residual_writers(layers[0], hidden) if layers else []
    names = [n.split(".")[-1] for lyr in [layers[0]] for n, m in lyr.named_modules()
             if m in sample] if sample else []
    print(f"[detect] arch={cfg.model_type}  hidden={hidden}  layers={len(layers)}  "
          f"residual-writers/block={len(sample)} ({', '.join(names) or '?'})")
    if len(sample) < 2:
        print("[detect] WARNING: found <2 residual-writing matrices per block — "
              "auto-detect may not fully support this architecture.")
    return hidden, len(layers)


# ------------------------------------------------------ inference helpers
def build_prompt(tok, text):
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return text


def _encode(tok, texts, device):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok([build_prompt(tok, t) for t in texts],
               return_tensors="pt", padding=True).to(device)


def _last_logits(model, enc):
    """Logits at the final position only — skips the full-sequence LM head."""
    try:
        return model(**enc, logits_to_keep=1).logits[:, -1, :].float()
    except TypeError:
        try:
            return model(**enc, num_logits_to_keep=1).logits[:, -1, :].float()
        except TypeError:
            return model(**enc).logits[:, -1, :].float()


def _hidden_states(model, enc):
    """Hidden states with the LM head kept minimal (logits_to_keep) when supported."""
    for kw in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
        try:
            return model(**enc, output_hidden_states=True, **kw).hidden_states
        except TypeError:
            continue
    return model(**enc, output_hidden_states=True).hidden_states


@torch.no_grad()
def mean_last_hidden(model, tok, prompts, device, batch_size):
    """Mean last-token hidden state per layer (index 0 = embeddings)."""
    sums, n = None, 0
    for i in range(0, len(prompts), batch_size):
        enc = _encode(tok, prompts[i:i + batch_size], device)
        hs = _hidden_states(model, enc)
        if sums is None:
            sums = [torch.zeros(h.shape[-1], dtype=torch.float64) for h in hs]
        for L, h in enumerate(hs):
            sums[L] += h[:, -1, :].to(torch.float64).sum(0).cpu()
        n += enc.input_ids.shape[0]
    return [s / max(n, 1) for s in sums]


def refusal_token_ids(tok):
    ids = set()
    for w in _REFUSAL_OPENERS:
        for variant in (w, " " + w):
            t = tok(variant, add_special_tokens=False)["input_ids"]
            if t:
                ids.add(t[0])
    return sorted(ids)


@torch.no_grad()
def refusal_score(model, tok, prompts, ref_ids, device, batch_size):
    """logit( P(model's next token is a refusal opener) ), averaged. Higher = refuses more."""
    idx = torch.tensor(ref_ids, device=device)
    total, n = 0.0, 0
    for i in range(0, len(prompts), batch_size):
        enc = _encode(tok, prompts[i:i + batch_size], device)
        probs = torch.softmax(_last_logits(model, enc), dim=-1)
        p = probs.index_select(1, idx).sum(1).clamp(1e-6, 1 - 1e-6)
        total += torch.log(p / (1 - p)).sum().item()
        n += enc.input_ids.shape[0]
    return total / max(n, 1)


@contextlib.contextmanager
def ablation_hooks(model, direction):
    """Project `direction` out of every residual activation at inference time —
    reversible, so candidate directions can be scored without touching weights."""
    d = (direction / direction.norm()).float()

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        dd = d.to(h.device, h.dtype)
        h2 = h - (h @ dd).unsqueeze(-1) * dd
        return (h2,) + out[1:] if isinstance(out, tuple) else h2

    handles = [model.get_input_embeddings().register_forward_hook(hook)]
    for layer in find_layers(model):
        handles.append(layer.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def kl_harmless(model, tok, prompts, direction, device, batch_size):
    """Mean KL(base || ablated) on the last-token distribution over harmless prompts.
    This is the 'damage' the ablation does to normal behaviour — minimise it."""
    total, n = 0.0, 0
    for i in range(0, len(prompts), batch_size):
        enc = _encode(tok, prompts[i:i + batch_size], device)
        base = torch.log_softmax(_last_logits(model, enc), dim=-1)
        with ablation_hooks(model, direction):
            abl = torch.log_softmax(_last_logits(model, enc), dim=-1)
        total += (base.exp() * (base - abl)).sum(-1).sum().item()
        n += enc.input_ids.shape[0]
    return total / max(n, 1)


# ------------------------------------------------------------ selection
def select_direction(model, tok, h_mean, s_mean, harmful_val, harmless_val,
                     ref_ids, device, batch_size, mode, fixed_layer, kl_weight,
                     max_depth=0.8):
    """Return (direction, layer, records).

    mode='auto'      : Heretic-style — score each candidate layer by how much it
                       drops refusals AND how little it perturbs harmless output
                       (objective = refusal + kl_weight*KL), pick the minimum.
    mode='magnitude' : cheap — biggest normalised harmful/harmless separation.
    mode='fixed'     : use --layer N.
    """
    n_layers = len(h_mean) - 1
    cutoff = int(max_depth * n_layers)

    if mode == "fixed":
        L = fixed_layer
        return h_mean[L + 1] - s_mean[L + 1], L, []

    if mode == "magnitude":
        best, bs = None, -1.0
        for L in range(1, cutoff):
            d = h_mean[L] - s_mean[L]
            scale = (h_mean[L].norm() + s_mean[L].norm()) / 2 + 1e-9
            sc = (d.norm() / scale).item()
            if sc > bs:
                bs, best = sc, (d, L - 1)
        return best[0], best[1], []

    # auto: co-minimise refusal + KL over candidate layers (small val sets)
    base_ref = refusal_score(model, tok, harmful_val, ref_ids, device, batch_size)
    print(f"[select] base refusal score = {base_ref:+.3f}  (searching layers 1-{cutoff-1})")
    records, best = [], None
    for L in range(1, cutoff):
        d = h_mean[L] - s_mean[L]
        with ablation_hooks(model, d):
            ref = refusal_score(model, tok, harmful_val, ref_ids, device, batch_size)
        kl = kl_harmless(model, tok, harmless_val, d, device, batch_size)
        obj = ref + kl_weight * kl
        records.append({"layer": L - 1, "refusal": round(ref, 3),
                        "kl": round(kl, 4), "objective": round(obj, 4)})
        print(f"  L{L-1:2d}  refusal={ref:+7.3f}  kl={kl:.4f}  obj={obj:+.4f}")
        if best is None or obj < best[2]:
            best = (d, L - 1, obj)
    print(f"[select] chose layer {best[1]}  (refusal drops, KL stays low)")
    return best[0], best[1], records


# ------------------------------------------------------------- the surgery
@torch.no_grad()
def apply_abliteration(model, direction, hidden_size, edit_embedding,
                       preserve_norms, scale):
    r = (direction / direction.norm()).to(torch.float32)
    edited, skipped = 0, 0
    for layer in find_layers(model):
        for mod in residual_writers(layer, hidden_size):
            W = mod.weight.data
            if W.shape[0] != r.shape[0]:            # dim guard
                skipped += 1
                continue
            orig = W.to(torch.float32)
            rr = r.to(orig.device)
            new = orig - scale * torch.outer(rr, rr @ orig)
            if preserve_norms:
                co = orig.norm(dim=0, keepdim=True)
                cn = new.norm(dim=0, keepdim=True).clamp_min(1e-8)
                new = new * (co / cn)
            mod.weight.data.copy_(new.to(W.dtype))
            edited += 1
    if edit_embedding:
        emb = model.get_input_embeddings().weight.data
        if emb.shape[1] == r.shape[0]:
            rr = r.to(emb.device)
            for s in range(0, emb.shape[0], 16384):
                blk = emb[s:s + 16384].to(torch.float32)
                blk = blk - scale * torch.outer(blk @ rr, rr)
                emb[s:s + 16384].copy_(blk.to(emb.dtype))
            edited += 1
    return edited, skipped


# ------------------------------------------------------------------- data
def load_prompts(path, default):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    return default


def _defaults(name):
    return load_prompts(os.path.join(HERE, "data", name), [])


def pick_device(arg):
    return ("cuda" if torch.cuda.is_available() else "cpu") if arg == "auto" else arg


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description="Standalone abliteration (any dense HF LLM)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", help="where to save (omit with --dry-run)")
    ap.add_argument("--n", type=int, default=128, help="contrast prompts per side for the direction")
    ap.add_argument("--select", default="auto", choices=["auto", "magnitude", "fixed"],
                    help="auto = co-minimise refusal+KL (best); magnitude = fast; fixed = --layer")
    ap.add_argument("--layer", type=int, default=None, help="layer index when --select fixed")
    ap.add_argument("--val-n", type=int, default=16, help="prompts used to score layers in auto mode")
    ap.add_argument("--kl-weight", type=float, default=8.0, help="how much to penalise damage (KL) vs refusal")
    ap.add_argument("--scale", type=float, default=1.0, help="ablation strength alpha (1.0 = full)")
    ap.add_argument("--preserve-norms", action="store_true", help="restore weight magnitude (keeps capability)")
    ap.add_argument("--edit-embedding", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--harmful-file", default=None)
    ap.add_argument("--harmless-file", default=None)
    ap.add_argument("--dry-run", action="store_true", help="analyse + pick layer, don't edit or save")
    ap.add_argument("--test", action="store_true", help="generate on a couple prompts afterwards")
    args = ap.parse_args()
    if not args.dry_run and not args.output:
        ap.error("--output is required unless --dry-run")

    device = pick_device(args.device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[load] {args.model}  device={device} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    model.to(device)          # for models bigger than one GPU: pip install accelerate,
    model.eval()              # then load with device_map="auto" instead (CPU offload)
    dev = next(model.parameters()).device
    hidden, n_layers = detection_report(model)

    harmful = load_prompts(args.harmful_file, _defaults("harmful.txt"))[:args.n]
    harmless = load_prompts(args.harmless_file, _defaults("harmless.txt"))[:args.n]
    if not harmful or not harmless:
        raise SystemExit("need harmful + harmless prompts (data/ or --harmful-file/--harmless-file)")
    print(f"[dirs] {len(harmful)} harmful / {len(harmless)} harmless")

    h_mean = mean_last_hidden(model, tok, harmful, dev, args.batch_size)
    s_mean = mean_last_hidden(model, tok, harmless, dev, args.batch_size)
    ref_ids = refusal_token_ids(tok)
    mode = "fixed" if args.select == "fixed" else args.select
    if mode == "fixed" and args.layer is None:
        raise SystemExit("--select fixed needs --layer N")
    direction, layer, records = select_direction(
        model, tok, h_mean, s_mean, harmful[:args.val_n], harmless[:args.val_n],
        ref_ids, dev, args.batch_size, mode, args.layer, args.kl_weight)
    print(f"[dirs] layer {layer}/{n_layers-1}  |r|={direction.norm():.3f}")

    if args.dry_run:
        print("[dry-run] no weights edited, nothing saved.")
        if records:
            best = min(records, key=lambda r: r["objective"])
            print(f"[dry-run] best layer would be {best['layer']} "
                  f"(refusal {best['refusal']}, kl {best['kl']})")
        return

    # measure refusal before, apply, measure after — the proof it worked
    before = refusal_score(model, tok, harmful[:args.val_n], ref_ids, dev, args.batch_size)
    n_edit, n_skip = apply_abliteration(model, direction.to(dev), hidden,
                                        args.edit_embedding, args.preserve_norms, args.scale)
    after = refusal_score(model, tok, harmful[:args.val_n], ref_ids, dev, args.batch_size)
    print(f"[apply] edited {n_edit} matrices (skipped {n_skip})"
          f"{' +norm-preserved' if args.preserve_norms else ''}  scale={args.scale}")
    print(f"[apply] refusal score {before:+.3f}  ->  {after:+.3f}   "
          f"({'✓ dropped' if after < before else '⚠ did not drop'})")

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    with open(os.path.join(args.output, "abliteration_info.json"), "w") as f:
        json.dump({"base_model": args.model, "layer": layer, "n_layers": n_layers,
                   "select": args.select, "scale": args.scale,
                   "edit_embedding": args.edit_embedding, "preserve_norms": args.preserve_norms,
                   "matrices_edited": n_edit, "refusal_before": round(before, 3),
                   "refusal_after": round(after, 3), "layer_search": records}, f, indent=2)
    print(f"[save] {args.output}")

    if args.test:
        print("\n[test] generating (abliterated model):")
        for p in harmful[:3]:
            enc = tok(build_prompt(tok, p), return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=60, do_sample=False)
            print(f"  Q: {p[:55]}\n  A: {tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()[:180]}\n")


if __name__ == "__main__":
    main()
