# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scikit-learn",
#     "matplotlib",
#     "pandas",
#     "torch",
#     "transformers",
#     "datasets",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # The Dead Salmons of AI Interpretability

    _An interactive walkthrough of Méloux, Dirupo, Portet, Peyrard (2025),_
    _[arXiv:2512.18792](https://arxiv.org/abs/2512.18792)._

    ---

    In 2009, a group of neuroscientists put an Atlantic salmon in an fMRI
    scanner. The salmon was dead. They showed it photographs of humans in
    social situations. Run through the standard statistical pipeline of the
    day, the analysis found brain regions in the dead salmon that were
    "significantly" activated by social cognition.

    The explanation wasn't postmortem empathy. It was **multiple comparisons
    without correction**: run enough tests on enough voxels and you'll find
    "significant" effects in pure noise. The 2009 paper became the canonical
    cautionary tale for applied statistics in neuroscience.

    Méloux et al. argue that AI interpretability is in the middle of its own
    dead-salmon crisis. Feature attribution, linear probes, sparse
    autoencoders, and circuit discovery can all produce plausible-looking,
    statistically-significant "explanations" when applied to neural networks
    that have learned **nothing** — networks whose weights have just been
    randomly initialized.

    This notebook reproduces the paper's Figure 1 artifacts live in your
    browser, then implements their proposed fix: reframe interpretability as
    **hypothesis testing against a null distribution derived from random
    computation**.

    Move the sliders. Watch the artifact appear and disappear.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Setup: a randomly-initialized BERT

    We load a small BERT architecture (bert-tiny, ~4M parameters) and then
    **throw away the pre-trained weights**, replacing them with fresh random
    initialization. This is our dead salmon: a network that has seen no
    training data and cannot possibly have learned anything about language.

    We then extract token embeddings from 300 IMDb movie-review sentences
    and mean-pool across sequence length to get one vector per sentence.
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    return AutoConfig, AutoModel, AutoTokenizer, torch


@app.cell
def _(mo):
    # Interactive controls for the setup pipeline.
    seed_ui = mo.ui.slider(
        start=0, stop=99, step=1, value=0, label="Random init seed"
    )
    n_samples_ui = mo.ui.slider(
        start=50, stop=500, step=50, value=300, label="Number of IMDb sentences"
    )
    pool_ui = mo.ui.dropdown(
        options=["mean", "cls", "max"], value="mean", label="Pooling"
    )

    mo.hstack([seed_ui, n_samples_ui, pool_ui])
    return n_samples_ui, pool_ui, seed_ui


@app.cell
def _(AutoConfig, AutoModel, AutoTokenizer, mo, seed_ui, torch):
    # Load a tiny BERT architecture and randomize the weights.
    # We cache by the seed value so re-running downstream cells doesn't rebuild.
    @mo.cache
    def build_random_bert(seed: int):
        model_name = "prajjwal1/bert-tiny"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        # Key move: instantiate from config (random init), NOT from_pretrained.
        torch.manual_seed(seed)
        model = AutoModel.from_config(config)
        model.eval()
        return tokenizer, model

    tokenizer, random_bert = build_random_bert(seed_ui.value)
    mo.md(
        f"Random BERT built with seed **{seed_ui.value}**. "
        f"Parameters: {sum(p.numel() for p in random_bert.parameters()):,}."
    )
    return random_bert, tokenizer


@app.cell
def _(mo, n_samples_ui):
    # Load IMDb. In WASM we use the datasets library's streaming loader.
    # For repeatability and speed we cache a curated, balanced sample.
    import pandas as pd

    @mo.cache
    def load_imdb_subset(n: int):
        # On first run this pulls from Hugging Face datasets. Subsequent runs
        # are cached locally.
        from datasets import load_dataset

        ds = load_dataset("imdb", split="train", streaming=False)
        # Balanced: n/2 positive, n/2 negative, deterministic order.
        pos = [ex for ex in ds if ex["label"] == 1][: n // 2]
        neg = [ex for ex in ds if ex["label"] == 0][: n // 2]
        rows = pos + neg
        return pd.DataFrame(rows)

    imdb = load_imdb_subset(n_samples_ui.value)
    mo.md(
        f"Loaded **{len(imdb)}** IMDb sentences "
        f"({(imdb['label'] == 1).sum()} positive / "
        f"{(imdb['label'] == 0).sum()} negative)."
    )
    return (imdb,)


@app.cell
def _(imdb, mo, pool_ui, random_bert, tokenizer, torch):
    # Forward pass + pooling. Cached by (seed, n_samples, pooling).
    @mo.cache
    def embed(texts_tuple: tuple[str, ...], pooling: str):
        texts = list(texts_tuple)
        with torch.no_grad():
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            out = random_bert(**enc)
            hidden = out.last_hidden_state  # (B, T, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()

            if pooling == "mean":
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            elif pooling == "cls":
                pooled = hidden[:, 0, :]
            elif pooling == "max":
                hidden_masked = hidden.masked_fill(mask == 0, float("-inf"))
                pooled = hidden_masked.max(dim=1).values
            else:
                raise ValueError(pooling)
        return pooled.cpu().numpy()

    texts = tuple(imdb["text"].str.slice(0, 500).tolist())
    X = embed(texts, pool_ui.value)
    y = imdb["label"].to_numpy()
    mo.md(
        f"Embedded into shape **{X.shape}** "
        f"(`{pool_ui.value}` pooling). Labels: **{y.shape}**."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Artifact A — PCA components that "explain" sentiment

    _Reproduces Figure 1A of the paper._

    We run PCA on the random-BERT embeddings and correlate each principal
    component with the binary sentiment label. If the embeddings were pure
    noise, no component should correlate with the label beyond chance. And
    yet...

    **TODO** — implement in Day 2.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Artifact B — a linear probe that "works"

    _Reproduces Figure 1B of the paper._

    We train a logistic regression classifier on the random embeddings with
    5-fold cross-validation and report accuracy. A probe on a truly random
    network should sit at chance (50%). The paper reports the probe instead
    lands around 60–65% — a difference that, with a tight confidence
    interval over hundreds of samples, would conventionally be reported as
    "highly significant."

    **TODO** — implement in Day 2.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The fix — null distribution across random seeds

    The paper's key methodological proposal: instead of comparing probe
    accuracy to chance (50%), compare it to a **null distribution** built by
    re-running the whole pipeline on many independently-randomized networks.
    If your "significant" finding falls inside that null, it isn't a finding.

    **TODO** — implement in Day 3.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The dead salmon zoo — extension

    Is the artifact BERT-specific? We swap in a random MLP, a random
    GPT-2-small, and a random ConvNet on a toy vision task and show the
    same phenomenon appears everywhere. This is our original contribution
    beyond the paper.

    **TODO** — implement in Day 3.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaways

    **TODO** — write in Day 4, after everything else works.
    """)
    return


if __name__ == "__main__":
    app.run()
