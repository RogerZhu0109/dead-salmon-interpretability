---
title: Dead Salmons of AI Interpretability
emoji: 🐟
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Dead Salmons of AI Interpretability

An interactive marimo notebook reproducing and extending the central experiments
from Méloux et al., *The Dead Salmons of AI Interpretability* (arXiv:2512.18792).

Interpretability methods — probes, PCA, SAEs, circuit discovery — can produce
plausible-looking, statistically significant "explanations" from **randomly
initialized networks** that have learned nothing. This notebook shows the
artifact live and implements the proposed fix: hypothesis testing against a null
distribution from random computation.

## Sections

1. **Hook** — the 2009 Bennett et al. dead salmon fMRI study as analogy
2. **Setup** — randomly re-initialized bert-tiny, IMDb embeddings
3. **Artifact A: PCA** — principal components correlating with sentiment
4. **Artifact B: Probe** — logistic regression with nontrivial CV accuracy
5. **The Fix** — null distribution + empirical p-value
6. **Dead Salmon Zoo** — architecture-independent demonstration (MLP, GPT-2-small)
7. **Probe Complexity Sweep** — how expressive probes find more spurious structure
8. **Takeaways**

## Paper

Méloux, Dirupo, Portet, Peyrard (2025). [arXiv:2512.18792](https://arxiv.org/abs/2512.18792)