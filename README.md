# refusal-cut

A small tool that removes the "refusal direction" from an open-weight language
model, so it stops refusing requests it was trained to refuse. It runs on plain
Hugging Face `transformers`, so it works on most dense decoder models (Llama,
Qwen, Mistral, Gemma, Phi, and similar) without any per-model setup.

It's based on Arditi et al. 2024, *Refusal in Language Models Is Mediated by a
Single Direction*. Refusal turns out to live mostly along one direction in the
model's internal activations. Find that direction from the difference between how
the model represents harmful and harmless prompts, then project it out of the
weights.

This is a research tool. Use it on models and data you're allowed to. It only
works on open weights you can load into memory — it can't touch closed models
(GPT, Claude, Gemini) because you don't have their weights.

## How it works

1. Run a set of harmful and harmless prompts through the model and grab the
   last-token hidden state at each layer.
2. The mean difference between the harmful and harmless activations at a layer is
   the refusal direction.
3. Pick the layer that removes refusal with the least damage to the rest of the
   model (it scores candidates by how much refusal drops against how far the
   output drifts, KL divergence), then project that direction out of every weight
   matrix that writes back to the residual stream.
4. Save the edited model.

## Install

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

## Use

```bash
# small model, CPU is fine
python abliterate.py --model Qwen/Qwen2.5-1.5B-Instruct --output ./out --test

# bigger model on a GPU, keeps capability high
python abliterate.py --model meta-llama/Llama-3.2-3B-Instruct --output ./out \
    --select auto --preserve-norms --test

# just analyse, don't edit anything
python abliterate.py --model <id> --dry-run

# your own prompts
python abliterate.py --model <id> --output ./out \
    --harmful-file my_harmful.txt --harmless-file my_harmless.txt
```

Load the result like any other model: `AutoModelForCausalLM.from_pretrained("./out")`.

## Options

| flag | what it does |
|---|---|
| `--model` | HF id or local path |
| `--output` | where to save the edited model |
| `--select` | `auto` (find the cleanest layer), `magnitude` (fast), or `fixed` |
| `--layer` | layer index when using `--select fixed` |
| `--scale` | ablation strength, 1.0 is full |
| `--preserve-norms` | restore weight magnitude after the edit, keeps the model sharper |
| `--edit-embedding` | also edit the embedding matrix |
| `--dry-run` | analyse and pick a layer, don't edit or save |
| `--test` | generate on a couple of prompts afterwards |
| `--device` | `auto`, `cpu`, or `cuda` |

## What works, what doesn't

Works: open dense models (Llama, Qwen, Mistral, Gemma, Phi, Yi, DeepSeek-dense)
that fit in your RAM or VRAM. The architecture is detected automatically.

Doesn't: closed models (no weights), models too big to load without a serious
GPU, and mixture-of-experts models (Mixtral, DeepSeek-MoE, Kimi) — those route
through per-expert layers and would need extra work, treat them as unsupported.

## Notes

The `data/` folder holds small default prompt lists used only to locate the
refusal direction. The core math follows the paper; `--preserve-norms` is the
common capability-preserving tweak (restore magnitude after the projection).
Layer selection borrows the idea from Heretic (p-e-w/heretic) of picking the
layer that removes refusal while keeping the KL divergence low.
