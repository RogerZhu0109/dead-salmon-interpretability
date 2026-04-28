# 🐟 The Dead Salmons of AI Interpretability

[![Open in HF Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/RZ0109/Dead_Salmon_Interpretability)
[![arXiv](https://img.shields.io/badge/arXiv-2512.18792-b31b1b.svg)](https://arxiv.org/abs/2512.18792)
[![marimo](https://marimo.io/shield.svg)](https://marimo.io)

An interactive marimo notebook reproducing and extending Méloux et al.,
*The Dead Salmons of AI Interpretability* (arXiv:2512.18792, Dec 2025).

> Interpretability methods — linear probes, PCA, SAEs, circuit discovery —
> can produce plausible-looking, statistically significant "explanations"
> from **randomly initialized networks that have learned nothing**.
> The authors call these artifacts "dead salmons," after a 2009 fMRI study
> that found significant brain activity in a dead Atlantic salmon.
> Their proposed fix: treat interpretability as hypothesis testing against
> a null distribution from random computation.

## What this notebook does

| Section | What you see |
| --- | --- |
| **Hook** | The 2009 Bennett et al. dead salmon fMRI analogy |
| **Setup** | Randomly re-initialized `bert-tiny`, IMDb embeddings with layer/pooling controls |
| **Artifact A — PCA** | Principal components that "explain" sentiment in a network that learned nothing |
| **Artifact B — Probe** | Logistic regression achieving well-above-chance CV accuracy on random embeddings |
| **The Fix** | Null distribution across random seeds + empirical p-value |
| **Dead Salmon Zoo** | Architecture-independent demo: random MLP, random GPT-2-small |
| **Probe Complexity Sweep** | How more expressive probes find more spurious structure |
| **Takeaways** | Three-bullet summary + link back to paper |

Every quantitative section has interactive `mo.ui` sliders and dropdowns — move them and watch the artifact appear and disappear.

## Run it

**Interactive (recommended):** [Open in HF Spaces](https://huggingface.co/spaces/RZ0109/Dead_Salmon_Interpretability)

**Locally:**
```bash
git clone https://github.com/RogerZhu0109/dead-salmon-interpretability
cd dead-salmon-interpretability
pip install marimo
marimo edit --sandbox notebooks/walkthrough.py
```

## Stack

- [marimo](https://marimo.io) — reactive notebook, deployed as a server-side app
- PyTorch + HuggingFace Transformers — random-init `prajjwal1/bert-tiny`
- scikit-learn — probes and PCA
- Hosted on HuggingFace Spaces (Docker, CPU)

## Reference

Méloux, A., Dirupo, G., Portet, F., & Peyrard, M. (2025).
*The Dead Salmons of AI Interpretability.*
[arXiv:2512.18792](https://arxiv.org/abs/2512.18792)
