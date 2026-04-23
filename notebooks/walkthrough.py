# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "scikit-learn",
#     "matplotlib",
#     "pandas",
#     "torch",
#     "transformers",
#     "datasets",
# ]
# ///

# marimo edit --sandbox --watch notebooks/walkthrough.py

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

    We load a small BERT architecture (bert-mini dimensions: 4 layers,
    256 hidden, ~11M params) and then **throw away any pre-trained
    weights** — the model is instantiated from a config, so every weight
    is freshly sampled from $\mathcal{N}(0, 0.02)$. This is our dead
    salmon: a network that has seen no training data and cannot possibly
    have learned anything about language.

    We extract token embeddings from a balanced sample of IMDb reviews
    and pool across the sequence to get one vector per document.
    Mean-pooling is the default because it captures per-document
    vocabulary variation most directly in a shallow random net (try the
    dropdown to compare).
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from scipy import stats
    from sklearn.decomposition import PCA

    return PCA, np, plt, stats, torch


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
def _(mo, seed_ui):
    # Load a tiny BERT architecture and randomize the weights.
    # We cache by the seed value so re-running downstream cells doesn't rebuild.
    # Imports live inside the cached function because mo.cache doesn't always
    # preserve closure capture over classes imported at cell scope.
    @mo.cache
    def build_random_bert(seed: int):
        import torch as _torch
        from transformers import AutoModel, AutoTokenizer, BertConfig

        # bert-tiny ships no tokenizer files and its config.json predates the
        # `model_type` field, so we build both by hand: bert-base-uncased's
        # WordPiece tokenizer (the vocab bert-tiny was trained for) and a
        # BertConfig with bert-tiny's dimensions.
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # bert-mini dimensions: a middle ground between bert-tiny (too
        # weak to show the artifact cleanly in a 2-layer/128-dim net) and
        # bert-base (440MB, unusable in WASM). ~11M params.
        config = BertConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
        )
        # Key move: instantiate from config (random init), NOT from_pretrained.
        _torch.manual_seed(seed)
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
    # Batched so activations don't balloon for 500 docs × 256 tokens × 256
    # hidden, which would OOM WASM if done in one shot.
    @mo.cache
    def embed(texts_tuple: tuple[str, ...], pooling: str):
        import numpy as _np

        texts = list(texts_tuple)
        batch_size = 32
        pooled_batches = []
        with torch.no_grad():
            for _start in range(0, len(texts), batch_size):
                _batch = texts[_start : _start + batch_size]
                enc = tokenizer(
                    _batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
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
                pooled_batches.append(pooled.cpu().numpy())
        return _np.concatenate(pooled_batches, axis=0)

    texts = tuple(imdb["text"].tolist())
    X = embed(texts, pool_ui.value)
    y = imdb["label"].to_numpy()
    mo.md(
        f"Embedded into shape **{X.shape}** "
        f"(`{pool_ui.value}` pooling). Labels: **{y.shape}**."
    )
    return X, y


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Artifact A — PCA components that "explain" sentiment

    _Reproduces Figure 1A of the paper._

    Here's the first dead-salmon finding. We run PCA on the random-BERT
    embeddings, keep a pool of components, and **rank them by the
    absolute correlation of each PC with the binary sentiment label**.
    Then we look at the top-ranked few. This is the interpretability
    researcher's move: you don't stop at the variance-top PCs, you hunt
    for directions that "look meaningful" — correlations with the
    concept you already care about. Pearson's $r$ plus the textbook
    two-tailed $t$-test gives each selected PC a $p$-value.

    A network that has learned nothing *should* give correlations near
    zero and $p$-values drawn from uniform $[0, 1]$. What you'll see
    instead: the cherry-picked PCs sit well above the 95% null band,
    multiple clear $p < 0.05$, and their $p$-values collapse through
    $10^{-2}$, $10^{-3}$, and below as $n$ grows. A correlation that
    looks unremarkable at $n = 30$ is "highly significant" at $n = 500$.

    Nothing about the random-init network changed between panels. What
    grew is $n$, which shrinks the standard error of the (selection-
    inflated) correlation until the null-hypothesis test is forced to
    reject. Move the slider below and watch it happen.

    _Artifact strength is sensitive to the upstream knobs at the top of
    the notebook. The **random init seed** resamples a different dead
    salmon — some seeds show the effect more dramatically than others.
    **Pooling** matters too: `mean` (the default) captures per-document
    vocabulary variation most directly in a shallow random net; `cls` and
    `max` are interesting comparison points but tend to be weaker._

    _The sweep plot's **gray band** is the 95% null envelope — under a
    true $r = 0$, $|r|$ only exceeds $1.96/\sqrt{n-2}$ about 5% of the
    time. Any curve that stays above the band is, by definition,
    "significant" at $p < 0.05$. The dead salmon is the PC curves that
    poke out of the band and stay there as $n$ grows._
    """)
    return


@app.cell
def _(mo, n_samples_ui):
    # Local controls for Section 2. The subsample slider lets users sweep n
    # without re-embedding — we take a deterministic prefix of a fixed random
    # permutation, so growing n always adds samples rather than resampling.
    # The global "Number of IMDb sentences" slider caps how many are
    # available; we clamp at use time.
    pca_n_ui = mo.ui.slider(
        start=30, stop=500, step=10, value=60,
        label=f"Sample size for PCA (n, max {n_samples_ui.value})",
    )
    pca_k_ui = mo.ui.slider(
        start=2, stop=10, step=1, value=6,
        label="Principal components",
    )
    mo.hstack([pca_n_ui, pca_k_ui])
    return pca_k_ui, pca_n_ui


@app.cell
def _(PCA, X, mo, np, pca_k_ui, pca_n_ui, stats, y):
    # Fit PCA with a generous pool of components, then RANK them by |r|
    # with the sentiment label on the full data and keep the top k. This
    # mimics the interpretability researcher's workflow: you don't stop
    # at the variance-top PCs, you look for directions that correlate
    # with the concept you care about. The paper's argument is that this
    # post-hoc selection inflates |r| by multiple-comparisons on its own
    # — and that's the artifact we want to surface.
    k_pool = min(30, X.shape[1], X.shape[0] - 1)
    _pca = PCA(n_components=k_pool)
    _Z_full = _pca.fit_transform(X)
    var_ratio = _pca.explained_variance_ratio_

    # Rank every PC in the pool by |r| with sentiment on the full data.
    _r_full = np.array(
        [abs(stats.pearsonr(_Z_full[:, _j], y)[0]) for _j in range(k_pool)]
    )
    _order = np.argsort(_r_full)[::-1]

    k_use = min(int(pca_k_ui.value), k_pool)
    top_idx = _order[:k_use]

    # Subsample for the single-n snapshot.
    n_use = min(int(pca_n_ui.value), X.shape[0])
    _rng = np.random.default_rng(42)
    _perm = _rng.permutation(X.shape[0])
    _sl = _perm[:n_use]
    Z = _Z_full[_sl][:, top_idx]
    y_sub = y[_sl]

    pc_r = np.empty(k_use)
    pc_p = np.empty(k_use)
    for _k in range(k_use):
        _r, _p = stats.pearsonr(Z[:, _k], y_sub)
        pc_r[_k] = _r
        pc_p[_k] = _p

    _n_sig = int((pc_p < 0.05).sum())
    _idx_list = ", ".join(f"PC{_i + 1}" for _i in top_idx)
    mo.md(
        f"PCA fit on all **{X.shape[0]}** embeddings, keeping "
        f"**{k_pool}** components total. Displaying the **{k_use}** PCs "
        f"most correlated with sentiment on the full data — "
        f"{_idx_list}. "
        f"**{_n_sig} / {k_use}** clear $p < 0.05$ at $n = {n_use}$. "
        f"Min $p$-value: **{float(pc_p.min()):.2e}**."
    )
    return Z, k_use, n_use, pc_p, pc_r, top_idx, var_ratio, y_sub


@app.cell
def _(Z, k_use, n_use, np, pc_p, pc_r, plt, top_idx, var_ratio, y_sub):
    # Two-panel figure at the current n. Bars are the |r|-ranked top PCs
    # (red when p<0.05), labeled by their original variance rank so the
    # cherry-picking is visible. Scatter shows the two strongest by |r|.
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))

    _xs = np.arange(1, k_use + 1)
    _colors = ["#d62728" if _p < 0.05 else "#888888" for _p in pc_p]
    _ax1.bar(_xs, np.abs(pc_r), color=_colors)
    _ax1.set_xticks(_xs)
    _ax1.set_xticklabels(
        [f"PC{_i + 1}" for _i in top_idx], rotation=45, fontsize=8,
    )
    _ax1.set_xlabel("PC (ranked by full-data |r|)")
    _ax1.set_ylabel(r"$|r|$ with sentiment")
    _ax1.set_title(f"Top-{k_use} PCs by |r|  (n = {n_use}; red: p < 0.05)")
    _ymax = float(np.abs(pc_r).max())
    for _i, (_rv, _pv) in enumerate(zip(pc_r, pc_p)):
        _label = f"p={_pv:.1e}" if _pv < 0.01 else f"p={_pv:.2f}"
        _ax1.text(
            _i + 1, abs(_rv) + 0.03 * (_ymax or 1), _label,
            ha="center", fontsize=7,
        )
    _ax1.set_ylim(0, (_ymax * 1.3) if _ymax > 0 else 1)

    _ax2.scatter(
        Z[:, 0], Z[:, 1], c=y_sub, cmap="coolwarm",
        s=20, alpha=0.75, edgecolor="none",
    )
    _ax2.set_xlabel(
        f"PC{top_idx[0] + 1}  ({100 * var_ratio[top_idx[0]]:.1f}% var)"
    )
    _ax2.set_ylabel(
        f"PC{top_idx[1] + 1}  ({100 * var_ratio[top_idx[1]]:.1f}% var)"
    )
    _ax2.set_title("Top two PCs by |r|, colored by sentiment")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(PCA, X, np, pca_k_ui, plt, stats, y):
    # Sweep n with PC directions LOCKED: fit PCA once on the full embedding
    # matrix, then for each n compute r, p for every fixed PC on the
    # prefix subsample. Because the axes don't rotate under subsampling,
    # each PC's |r| converges to its true (small) asymptote and its p-value
    # collapses exponentially — the genuine dead-salmon signature.
    # Same |r|-ranked cherry-pick as the single-n cell: a pool of PCs,
    # ranked by correlation with the label on the full data, then the
    # top k swept across n. This is the interpretability researcher's
    # workflow; the paper's argument is that it produces artifact
    # "significance" even on a random net.
    _k_pool = min(30, X.shape[1], X.shape[0] - 1)
    _pca = PCA(n_components=_k_pool)
    _Z = _pca.fit_transform(X)

    _r_full = np.array(
        [abs(stats.pearsonr(_Z[:, _j], y)[0]) for _j in range(_k_pool)]
    )
    _order = np.argsort(_r_full)[::-1]
    _k = min(int(pca_k_ui.value), _k_pool)
    _top_idx = _order[:_k]

    _ns = np.unique(np.linspace(30, X.shape[0], 20).astype(int))
    _rng = np.random.default_rng(42)
    _perm = _rng.permutation(X.shape[0])

    _r_curves = np.empty((_k, len(_ns)))
    _p_curves = np.empty((_k, len(_ns)))
    for _i, _n in enumerate(_ns):
        _sl = _perm[:_n]
        _yn = y[_sl]
        for _j in range(_k):
            _rv, _pv = stats.pearsonr(_Z[_sl, _top_idx[_j]], _yn)
            _r_curves[_j, _i] = abs(_rv)
            _p_curves[_j, _i] = max(_pv, 1e-300)

    # 95% null envelope for |r|: under true r=0, |r| exceeds
    # 1.96/sqrt(n-2) only 5% of the time. A curve that stays ABOVE the
    # shaded band is "significant" at p<0.05; the artifact is exactly
    # those curves that cross the band and stay there as n grows.
    _null_threshold = 1.96 / np.sqrt(np.maximum(_ns - 2, 1))

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _cmap = plt.cm.viridis(np.linspace(0, 0.85, _k))

    _ax1.fill_between(
        _ns, 0, _null_threshold, color="gray", alpha=0.2,
        label="null: |r| expected under r=0",
    )
    _ax1.plot(_ns, _null_threshold, color="gray", lw=0.8, linestyle="--")
    for _j in range(_k):
        _ax1.plot(
            _ns, _r_curves[_j], marker="o", color=_cmap[_j],
            lw=1.2, markersize=4, label=f"PC{_top_idx[_j] + 1}",
        )
        _ax2.semilogy(
            _ns, _p_curves[_j], marker="o", color=_cmap[_j],
            lw=1.2, markersize=4, label=f"PC{_top_idx[_j] + 1}",
        )

    _ax1.set_xlabel("n (sentences)")
    _ax1.set_ylabel(r"$|r|$ with sentiment")
    _ax1.set_title("Curves above the gray band are 'significant' artifacts")
    _ax1.set_ylim(0, None)
    _ax1.legend(fontsize=7, ncol=2, loc="upper right")

    _ax2.axhline(0.05, color="k", lw=0.6, linestyle="--")
    _ax2.text(_ns[-1], 0.05, "  p=0.05", va="center", fontsize=8)
    _ax2.set_xlabel("n (sentences)")
    _ax2.set_ylabel("p-value (log scale)")
    _ax2.set_title("Significant PCs' p-values collapse exponentially")
    _ax2.legend(fontsize=7, ncol=2, loc="lower left")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **What just happened.** In the sweep plot, a handful of PCs settle
    into an $|r|$ asymptote around $0.1$–$0.2$ and stay there — that's the
    spurious correlation the random network "learned" between its own
    embedding geometry and the sentiment label. Meanwhile, their
    $p$-values on the right panel march downward on a log scale, crossing
    $p = 0.05$, then $10^{-2}$, $10^{-3}$, and beyond as $n$ grows. A
    correlation that looks unremarkable at $n = 30$ is "highly
    significant" at $n = 500$.

    Nothing about the network changed. Nothing about the data changed. We
    just collected more of it, and the standard null-hypothesis test
    rejected a null that — in the paper's framing — is the wrong null to
    begin with. Our random BERT cannot have learned anything about
    sentiment, but its embeddings are **not** isotropic noise: they are
    the output of a nonlinear random projection, and that projection
    inherits small but systematic covariance with any downstream label
    that shares the data's marginal statistics (vocabulary, length,
    punctuation density).

    The paper's fix (Section 4) is to compare the observed effect to a
    null distribution generated by **other** random initializations of
    the same architecture — a null that scales with $n$ the same way the
    artifact does, so growing $n$ stops giving a free win.
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
