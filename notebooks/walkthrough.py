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

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return (
        LogisticRegression,
        PCA,
        Pipeline,
        StandardScaler,
        StratifiedKFold,
        confusion_matrix,
        np,
        plt,
        stats,
        torch,
        train_test_split,
    )


@app.cell
def _(mo):
    seed_ui = mo.ui.slider(
        start=0, stop=99, step=1, value=0, label="Random init seed"
    )
    n_samples_ui = mo.ui.slider(
        start=50, stop=1000, step=50, value=1000, label="Number of IMDb sentences"
    )
    pool_ui = mo.ui.dropdown(
        options=["mean", "cls", "max"], value="mean", label="Pooling"
    )
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


@app.cell(hide_code=True)
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
def _(mo, n_samples_ui, pool_ui, seed_ui):
    pca_n_ui = mo.ui.slider(
        start=30, stop=1000, step=10, value=60,
        label=f"Sample size for PCA (n, max {n_samples_ui.value})",
    )
    pca_k_ui = mo.ui.slider(
        start=2, stop=10, step=1, value=6,
        label="Principal components",
    )
    mo.vstack([
        mo.md("**Embedding controls** — change these to reload embeddings (slow)"),
        mo.hstack([seed_ui, n_samples_ui, pool_ui]),
        mo.md("**PCA controls** — fast, no re-embedding"),
        mo.hstack([pca_n_ui, pca_k_ui]),
    ])
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Artifact B — a linear probe that "works"

    _Reproduces Figure 1B of the paper._

    The PCA section showed that *individual directions* in the random
    embedding correlate with sentiment. The probe section shows the
    stronger, scarier version: a **logistic regression** trained on those
    embeddings with 5-fold cross-validation lands well above the 50%
    chance line, with a tight confidence interval and a binomial
    $p$-value against chance that is for all practical purposes zero.

    The paper's Figure 1B reports a probe at roughly $60$–$65\%$ accuracy
    on a fully random transformer. By the standard interpretability
    pipeline this is a finding: "the model has learned a sentiment
    feature." But the model has not been trained. There is no feature.
    What the probe is doing is fitting the marginal statistics of the
    text — vocabulary, length, punctuation density — that any nonlinear
    random projection inherits from its inputs. The probe is real; the
    interpretation is not.

    Two knobs below.

    - **Probe regularization (log$_{10}$ C).** $C$ is sklearn's inverse-
      regularization-strength: small $C$ means heavy $L_2$ shrinkage and
      a simpler decision boundary, large $C$ lets the probe overfit.
      The artifact survives across the whole range — *that* is the point.
    - **Held-out test fraction.** A second, independent train/test split
      (separate from the 5-fold CV) is used to draw the confusion matrix
      and report a Wilson 95% CI on a single held-out slice. Smaller test
      fraction means a tighter probe but a noisier held-out estimate.

    Underneath, the embedding controls (seed, sample size, pooling) at
    the top of section 1 still apply — change them to swap in a different
    dead salmon.
    """)
    return


@app.cell
def _(mo, n_samples_ui, pool_ui, seed_ui):
    log10_C_ui = mo.ui.slider(
        start=-3, stop=2, step=0.25, value=0,
        label="Probe regularization (log10 C)",
        show_value=True,
    )
    test_size_ui = mo.ui.slider(
        start=0.1, stop=0.5, step=0.05, value=0.2,
        label="Held-out test fraction",
        show_value=True,
    )
    mo.vstack([
        mo.md("**Embedding controls** — change these to reload embeddings (slow)"),
        mo.hstack([seed_ui, n_samples_ui, pool_ui]),
        mo.md("**Probe controls** — fast, no re-embedding"),
        mo.hstack([log10_C_ui, test_size_ui]),
    ])
    return log10_C_ui, test_size_ui


@app.cell
def _(
    LogisticRegression,
    Pipeline,
    StandardScaler,
    StratifiedKFold,
    X,
    confusion_matrix,
    log10_C_ui,
    mo,
    np,
    stats,
    test_size_ui,
    train_test_split,
    y,
):
    # Logistic-regression probe behind a StandardScaler. Scaling is
    # standard practice in probing work and stabilizes liblinear's
    # convergence across the C sweep; the artifact shows up either way.
    def _make_probe(C):
        return Pipeline([
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(
                C=C, max_iter=2000, solver="liblinear",
            )),
        ])

    @mo.cache
    def probe_cv(X_arr, y_arr, log10_C, n_splits=5):
        # 5-fold stratified CV. Returns the per-fold accuracies so the
        # downstream cell can plot them and compute a t-based 95% CI.
        C = float(10 ** log10_C)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        accs = np.empty(n_splits)
        for _i, (_tr, _te) in enumerate(skf.split(X_arr, y_arr)):
            _clf = _make_probe(C)
            _clf.fit(X_arr[_tr], y_arr[_tr])
            accs[_i] = _clf.score(X_arr[_te], y_arr[_te])
        return accs

    @mo.cache
    def probe_holdout(X_arr, y_arr, log10_C, test_size):
        # Independent train/test split, separate from the CV above.
        # We use it to draw a confusion matrix and report a Wilson CI on
        # a single held-out slice.
        C = float(10 ** log10_C)
        Xtr, Xte, ytr, yte = train_test_split(
            X_arr, y_arr,
            test_size=test_size, stratify=y_arr, random_state=0,
        )
        clf = _make_probe(C)
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        return float((yhat == yte).mean()), confusion_matrix(yte, yhat), len(yte)

    fold_accs = probe_cv(X, y, log10_C_ui.value, n_splits=5)
    cv_mean = float(fold_accs.mean())
    cv_sem = float(fold_accs.std(ddof=1) / np.sqrt(len(fold_accs)))
    cv_ci_lo, cv_ci_hi = stats.t.interval(
        0.95, df=len(fold_accs) - 1, loc=cv_mean,
        scale=max(cv_sem, 1e-12),
    )

    holdout_acc, cm, n_test = probe_holdout(
        X, y, log10_C_ui.value, float(test_size_ui.value),
    )
    n_correct = int(round(holdout_acc * n_test))
    binom = stats.binomtest(n_correct, n_test, p=0.5, alternative="two-sided")
    holdout_ci = binom.proportion_ci(confidence_level=0.95, method="wilson")

    mo.md(
        f"**5-fold CV accuracy:** **{cv_mean:.3f}** "
        f"(95% CI [{cv_ci_lo:.3f}, {cv_ci_hi:.3f}], chance = 0.500). "
        f"**Held-out accuracy** on {n_test} samples: **{holdout_acc:.3f}** "
        f"(Wilson 95% CI [{holdout_ci.low:.3f}, {holdout_ci.high:.3f}]). "
        f"Binomial $p$ vs chance: **{binom.pvalue:.2e}**."
    )
    return (
        binom,
        cm,
        cv_ci_hi,
        cv_ci_lo,
        cv_mean,
        fold_accs,
        holdout_acc,
        n_test,
    )


@app.cell
def _(
    binom,
    cm,
    cv_ci_hi,
    cv_ci_lo,
    cv_mean,
    fold_accs,
    holdout_acc,
    log10_C_ui,
    n_test,
    np,
    plt,
):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: per-fold accuracies as dots, mean line, 95% CI band, chance line.
    _xs = np.arange(1, len(fold_accs) + 1)
    _ax1.fill_between(
        [0.5, len(fold_accs) + 0.5], cv_ci_lo, cv_ci_hi,
        color="#1f77b4", alpha=0.18, label="mean 95% CI",
    )
    _ax1.axhline(
        cv_mean, color="#1f77b4", linewidth=2,
        label=f"mean = {cv_mean:.3f}",
    )
    _ax1.axhline(
        0.5, color="k", linewidth=0.9, linestyle="--", label="chance",
    )
    _ax1.scatter(
        _xs, fold_accs, s=90, color="#1f77b4", zorder=3,
        edgecolor="white", linewidth=1.4,
    )
    for _i, _a in enumerate(fold_accs):
        _ax1.text(
            _i + 1, _a + 0.012, f"{_a:.3f}",
            ha="center", fontsize=8, color="#1f77b4",
        )
    _ax1.set_xticks(_xs)
    _ax1.set_xlim(0.5, len(fold_accs) + 0.5)
    _ymin = min(0.45, float(fold_accs.min()) - 0.05)
    _ymax = max(0.75, float(fold_accs.max()) + 0.06)
    _ax1.set_ylim(_ymin, _ymax)
    _ax1.set_xlabel("Fold")
    _ax1.set_ylabel("Accuracy")
    _ax1.set_title(
        f"5-fold CV  (log$_{{10}}$ C = {log10_C_ui.value:.2f})"
    )
    _ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)
    _ax1.grid(True, axis="y", alpha=0.3)

    # Right: confusion matrix from the independent held-out split.
    _ax2.imshow(cm, cmap="Blues", aspect="equal")
    _ax2.set_xticks([0, 1])
    _ax2.set_yticks([0, 1])
    _ax2.set_xticklabels(["neg", "pos"])
    _ax2.set_yticklabels(["neg", "pos"])
    _ax2.set_xlabel("Predicted")
    _ax2.set_ylabel("True")
    _ax2.set_title(
        f"Held-out  (n = {n_test}, acc = {holdout_acc:.3f}, "
        f"p = {binom.pvalue:.1e})"
    )
    _vmax = float(cm.max())
    for _i in range(2):
        for _j in range(2):
            _val = int(cm[_i, _j])
            _ax2.text(
                _j, _i, f"{_val}",
                ha="center", va="center",
                color="white" if _val > _vmax / 2 else "black",
                fontsize=15, fontweight="bold",
            )

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    LogisticRegression,
    Pipeline,
    StandardScaler,
    StratifiedKFold,
    X,
    log10_C_ui,
    mo,
    np,
    stats,
    y,
):
    # Sweep accuracy as a function of training-set size, holding the probe
    # complexity (C) and embedding fixed. Mirrors §2: the dead-salmon
    # signature is that "significance" against chance grows with n, even
    # though nothing about the network has changed. Subsamples are
    # balanced (n/2 positive, n/2 negative) so chance stays at exactly 0.5.
    @mo.cache
    def probe_n_sweep(X_arr, y_arr, log10_C, n_points=12, n_min=60):
        C = float(10 ** log10_C)
        N = len(y_arr)
        n_min_eff = min(n_min, N)
        ns = np.unique(
            np.linspace(n_min_eff, N, n_points).astype(int)
        )
        ns = ns[ns >= 20]

        rng = np.random.default_rng(0)
        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        means = np.empty(len(ns))
        ci_lo = np.empty(len(ns))
        ci_hi = np.empty(len(ns))
        for _i, _n in enumerate(ns):
            _half = int(_n) // 2
            _idx = np.concatenate([pos_idx[:_half], neg_idx[:_half]])
            _Xs = X_arr[_idx]
            _ys = y_arr[_idx]
            _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            _accs = np.empty(5)
            for _k, (_tr, _te) in enumerate(_skf.split(_Xs, _ys)):
                _clf = Pipeline([
                    ("scale", StandardScaler()),
                    ("lr", LogisticRegression(
                        C=C, max_iter=2000, solver="liblinear",
                    )),
                ])
                _clf.fit(_Xs[_tr], _ys[_tr])
                _accs[_k] = _clf.score(_Xs[_te], _ys[_te])
            _m = float(_accs.mean())
            _sem = float(_accs.std(ddof=1) / np.sqrt(5))
            _lo, _hi = stats.t.interval(
                0.95, df=4, loc=_m, scale=max(_sem, 1e-12),
            )
            means[_i] = _m
            ci_lo[_i] = _lo
            ci_hi[_i] = _hi
        return ns, means, ci_lo, ci_hi

    sweep_ns, sweep_means, sweep_lo, sweep_hi = probe_n_sweep(
        X, y, log10_C_ui.value,
    )
    return sweep_hi, sweep_lo, sweep_means, sweep_ns


@app.cell
def _(log10_C_ui, np, plt, stats, sweep_hi, sweep_lo, sweep_means, sweep_ns):
    # 95% null band: under chance = 0.5 the count of correct predictions
    # on n samples is Binomial(n, 0.5). The 2.5/97.5 percentiles of that
    # over n give a band that any "real" probe must escape to be called
    # significant at p < 0.05. Curves above the band are dead salmons.
    _null_lo = np.array(
        [stats.binom.ppf(0.025, int(_n), 0.5) / _n for _n in sweep_ns]
    )
    _null_hi = np.array(
        [stats.binom.ppf(0.975, int(_n), 0.5) / _n for _n in sweep_ns]
    )

    _fig, _ax = plt.subplots(figsize=(8.5, 4.5))
    _ax.fill_between(
        sweep_ns, _null_lo, _null_hi,
        color="gray", alpha=0.22,
        label="null: 95% binomial band at chance",
    )
    _ax.axhline(0.5, color="gray", lw=0.8, linestyle="--")
    _ax.fill_between(
        sweep_ns, sweep_lo, sweep_hi,
        color="#d62728", alpha=0.22, label="probe 95% CI",
    )
    _ax.plot(
        sweep_ns, sweep_means,
        color="#d62728", marker="o", lw=1.6, markersize=5,
        label="probe mean (5-fold CV)",
    )
    _ax.set_xlabel("Number of training sentences (n, balanced)")
    _ax.set_ylabel("5-fold CV accuracy")
    _ax.set_ylim(
        min(0.4, float(sweep_lo.min()) - 0.03),
        max(0.8, float(sweep_hi.max()) + 0.03),
    )
    _ax.set_title(
        f"Probe accuracy escapes the null band as $n$ grows  "
        f"(log$_{{10}}$ C = {log10_C_ui.value:.2f})"
    )
    _ax.legend(loc="lower right", fontsize=9)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What just happened.** A logistic-regression probe on a random
    BERT's embeddings achieves clearly-above-chance accuracy. The 95%
    confidence interval over 5 folds excludes 50%. The binomial
    $p$-value against chance on the held-out split is for all practical
    purposes zero. By every conventional standard this is a "successful"
    probe — and yet the network has been trained on nothing.

    The sweep plot makes the dead-salmon dynamic vivid: at small $n$ the
    probe sits inside the null band (you cannot reject chance); as $n$
    grows the mean and its CI lift cleanly above. What's growing is
    statistical resolution, not real signal. The probe is fitting
    spurious structure that the random projection inherits from the
    data's marginal statistics — token frequencies, sequence length,
    punctuation density, and the like.

    Try cranking $\log_{10} C$ to either extreme: heavy regularization
    ($C \!\to\! 0$) shrinks the probe toward a near-mean classifier and
    only marginally hurts accuracy, while $C \!\to\! \infty$ lets the
    probe overfit hard without paying much of a generalization cost on
    a dataset this small. The artifact lives in the embedding geometry,
    not in the probe's degrees of freedom.

    Section 4 implements the paper's fix: don't compare to chance —
    compare to a **null distribution** built from many independent
    random initializations of the same architecture. If the observed
    accuracy falls inside that null, the probe has not learned anything
    that random computation could not have produced on its own.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The fix — a null distribution from random computation

    _Reproduces the spirit of the paper's Figure 1C and Section 4._

    Sections 2 and 3 produced two "findings" on a network that learned
    nothing: principal components that "explain" sentiment and a
    logistic-regression probe well above chance. Both pass the
    conventional null-hypothesis tests — correlation against $r = 0$,
    accuracy against chance = $0.5$. The paper argues those are the
    **wrong nulls**. The right question is:

    > If I reran this pipeline end-to-end on a **different random
    > initialization** of the same architecture, would I get the same
    > finding?

    If the answer is yes, the finding isn't about what the network
    learned; it's about what the architecture plus the data's marginal
    statistics produce on their own. The paper calls this **null
    hypothesis significance testing against random computation**.

    We reseed the BERT $N$ times, rerun random-init → embed → 5-fold CV
    probe on each seed, and collect the resulting accuracies into an
    empirical null. We overlay the **observed** accuracy at the current
    seed and report the empirical right-tail $p$-value

    $$
    p = \frac{1 + \#\{s : \mathrm{acc}_s \geq \mathrm{acc}_\mathrm{obs}\}}{1 + N}.
    $$

    Because the "observed" network is itself just another random init,
    this $p$-value should be approximately uniform on $[0, 1]$ — we
    should **fail to reject** the correct null, even though the
    conventional chance-level null was rejected with overwhelming
    confidence. That gap is the fix working.
    """)
    return


@app.cell
def _(log10_C_ui, mo, pool_ui, seed_ui):
    n_null_seeds_ui = mo.ui.slider(
        start=10, stop=100, step=5, value=30,
        label="Null seeds (N)", show_value=True,
    )
    null_n_ui = mo.ui.slider(
        start=100, stop=500, step=50, value=200,
        label="Sentences per seed", show_value=True,
    )
    mo.vstack([
        mo.md(
            "**Null-distribution controls.** Each seed triggers a full "
            "random-init + embed + probe run. The first pass is slow "
            "(~1 s/seed native, ~3 s/seed on WASM); reruns are instant "
            "thanks to `mo.cache`. The **observed** run uses the current "
            f"settings from sections 1 and 3: seed **{seed_ui.value}**, "
            f"pool **{pool_ui.value}**, "
            f"log$_{{10}}\\,C$ = **{log10_C_ui.value:.2f}**."
        ),
        mo.hstack([n_null_seeds_ui, null_n_ui]),
    ])
    return n_null_seeds_ui, null_n_ui


@app.cell
def _(
    LogisticRegression,
    Pipeline,
    StandardScaler,
    StratifiedKFold,
    log10_C_ui,
    mo,
    n_null_seeds_ui,
    np,
    null_n_ui,
    pool_ui,
    seed_ui,
):
    # Self-contained per-seed pipeline: build a random BERT at `seed`,
    # embed a balanced IMDb subset of size `n`, then run 5-fold CV with
    # the same logistic-regression probe as section 3. mo.cache keys on
    # the args, so increasing N just computes the new seeds and leaves
    # already-cached results alone. Imports live inside the function to
    # match the pattern used by `build_random_bert` — mo.cache does not
    # reliably close over classes captured at cell scope.

    @mo.cache
    def _load_balanced_imdb(n: int):
        from datasets import load_dataset

        ds = load_dataset("imdb", split="train", streaming=False)
        pos_texts, neg_texts = [], []
        half = n // 2
        for ex in ds:
            if ex["label"] == 1 and len(pos_texts) < half:
                pos_texts.append(ex["text"])
            elif ex["label"] == 0 and len(neg_texts) < half:
                neg_texts.append(ex["text"])
            if len(pos_texts) >= half and len(neg_texts) >= half:
                break
        texts = tuple(pos_texts + neg_texts)
        labels = tuple([1] * len(pos_texts) + [0] * len(neg_texts))
        return texts, labels

    @mo.cache
    def _seed_probe_accuracy(seed: int, n: int, pooling: str, log10_C: float):
        import numpy as _np
        import torch as _torch
        from transformers import AutoModel, AutoTokenizer, BertConfig

        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        cfg = BertConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
        )
        _torch.manual_seed(seed)
        model = AutoModel.from_config(cfg)
        model.eval()

        texts, labels = _load_balanced_imdb(n)
        labels_arr = _np.array(labels)

        batches = []
        with _torch.no_grad():
            for _s in range(0, len(texts), 32):
                _batch = list(texts[_s : _s + 32])
                enc = tok(
                    _batch, padding=True, truncation=True,
                    max_length=256, return_tensors="pt",
                )
                out = model(**enc)
                hidden = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                if pooling == "mean":
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                elif pooling == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    hm = hidden.masked_fill(mask == 0, float("-inf"))
                    pooled = hm.max(dim=1).values
                batches.append(pooled.cpu().numpy())
        X_s = _np.concatenate(batches, axis=0)

        C = float(10 ** log10_C)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accs = _np.empty(5)
        for _i, (_tr, _te) in enumerate(skf.split(X_s, labels_arr)):
            _clf = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(
                    C=C, max_iter=2000, solver="liblinear",
                )),
            ])
            _clf.fit(X_s[_tr], labels_arr[_tr])
            accs[_i] = _clf.score(X_s[_te], labels_arr[_te])
        return float(accs.mean())

    _N = int(n_null_seeds_ui.value)
    _null_n = int(null_n_ui.value)
    _pool = pool_ui.value
    _logC = float(log10_C_ui.value)
    _user_seed = int(seed_ui.value)

    # Null seeds: the first N nonnegative integers that aren't the
    # user's seed. This keeps the null deterministic and lets us talk
    # about "the first N random inits" as a stable reference set.
    _null_seeds = []
    _k = 0
    while len(_null_seeds) < _N:
        if _k != _user_seed:
            _null_seeds.append(_k)
        _k += 1

    _null_list = []
    for _s in mo.status.progress_bar(
        _null_seeds,
        title="Random-computation null",
        subtitle=f"building null across {_N} seeds",
        remove_on_exit=True,
    ):
        _null_list.append(_seed_probe_accuracy(_s, _null_n, _pool, _logC))
    null_accs = np.array(_null_list)
    observed_acc = _seed_probe_accuracy(_user_seed, _null_n, _pool, _logC)

    # Empirical right-tail p-value with add-one (Phipson–Smyth) correction.
    n_exceed = int((null_accs >= observed_acc).sum())
    p_emp = (1 + n_exceed) / (1 + len(null_accs))
    null_mean = float(null_accs.mean())
    null_sd = float(null_accs.std(ddof=1))

    _verdict = (
        "**reject** the random-computation null — the observed "
        "accuracy is in the upper tail of what the architecture alone "
        "can produce"
        if p_emp < 0.05
        else "**fail to reject** the random-computation null — "
        "exactly what we expect, because the 'observed' network is "
        "itself just another dead salmon"
    )

    mo.md(
        f"**Observed** (seed {_user_seed}): "
        f"acc = **{observed_acc:.3f}**. "
        f"**Null** ({len(null_accs)} other random inits): "
        f"mean = **{null_mean:.3f}**, sd = **{null_sd:.3f}**, "
        f"range = [{null_accs.min():.3f}, {null_accs.max():.3f}]. "
        f"**Empirical $p$-value** = "
        f"$(1 + {n_exceed}) / (1 + {len(null_accs)})$ = "
        f"**{p_emp:.3f}** — {_verdict}."
    )
    return n_exceed, null_accs, null_mean, observed_acc, p_emp


@app.cell
def _(
    log10_C_ui,
    n_exceed,
    np,
    null_accs,
    null_mean,
    null_n_ui,
    observed_acc,
    p_emp,
    plt,
    pool_ui,
):
    # Histogram of null accuracies, observed overlaid as a vertical line,
    # right-tail shaded red. Annotations in the plot carry the p-value and
    # the two competing nulls (chance, and random computation) so the
    # figure stands on its own if lifted out of the notebook.
    _all = np.concatenate([null_accs, [observed_acc]])
    _lo = float(min(0.48, _all.min() - 0.02))
    _hi = float(max(0.72, _all.max() + 0.02))
    _bins = np.linspace(_lo, _hi, 24)

    _fig, _ax = plt.subplots(figsize=(9, 4.6))

    _counts, _edges, _ = _ax.hist(
        null_accs, bins=_bins, color="#aaaaaa",
        edgecolor="white", alpha=0.88,
        label=f"null  (N = {len(null_accs)} random inits)",
    )
    _tail = null_accs[null_accs >= observed_acc]
    if len(_tail) > 0:
        _ax.hist(
            _tail, bins=_bins, color="#d62728",
            edgecolor="white", alpha=0.85,
            label=f"null $\\geq$ observed  ({n_exceed})",
        )

    _ymax = (float(_counts.max()) if _counts.max() > 0 else 1.0) * 1.35

    _ax.axvline(
        0.5, color="k", lw=0.9, linestyle="--", alpha=0.7,
        label="chance (the conventional, wrong null)",
    )
    _ax.axvline(
        null_mean, color="#444444", lw=1.2, linestyle=":",
        label=f"null mean = {null_mean:.3f}",
    )
    _ax.axvline(
        observed_acc, color="#1f77b4", lw=2.6,
        label=f"observed = {observed_acc:.3f}",
    )
    _ax.annotate(
        f"$p_{{\\mathrm{{emp}}}} = {p_emp:.3f}$",
        xy=(observed_acc, _ymax * 0.92),
        xytext=(
            observed_acc + (_hi - _lo) * 0.035,
            _ymax * 0.92,
        ),
        fontsize=13, fontweight="bold", color="#1f77b4",
        va="center",
    )

    _ax.set_xlabel("5-fold CV accuracy")
    _ax.set_ylabel("# random initializations")
    _ax.set_xlim(_lo, _hi)
    _ax.set_ylim(0, _ymax)
    _ax.set_title(
        f"Observed vs. random-computation null  "
        f"(n = {int(null_n_ui.value)} sentences/seed, "
        f"pool = {pool_ui.value}, "
        f"log$_{{10}}\\,C$ = {log10_C_ui.value:.2f})"
    )
    _ax.legend(loc="upper left", fontsize=9, framealpha=0.93)
    _ax.grid(True, axis="y", alpha=0.28)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What just happened.** The blue line is our observed probe
    accuracy — the same "highly significant" finding from section 3,
    where a binomial test against chance gave $p$ on the order of
    $10^{-6}$ or smaller. Against the **right** null — the distribution
    of probe accuracies across other random inits of the same
    architecture — the same number is completely unremarkable. It
    lands near the middle of the histogram, with an empirical
    $p$-value that is nowhere near $0.05$ and behaves like a uniform
    draw as you scan the seed slider.

    That is the fix. The conventional null ($r = 0$ for PCs, 50% for
    probes) assumes the baseline is **pure noise**. But a random-init
    network is not pure noise: it is a fixed nonlinear projection of
    its input, and that projection inherits systematic covariance with
    any label that shares the data's marginal statistics (token
    frequencies, sequence length, punctuation density). The
    random-computation null bakes those marginals into the baseline,
    so a real finding must exceed what the architecture alone can
    produce — not just what chance would.

    Try nudging the **seed** slider at the top of the notebook. As you
    scan across seeds the blue line slides left and right along the
    gray histogram, because every one of those seeds is itself another
    dead salmon. The histogram **is** the space of dead salmons; our
    observation is one more sample from it. The conventional test
    asked whether our probe beat chance and got a resounding yes. The
    paper's test asks the better question — whether our probe beat
    random computation — and the answer, as it should be, is no.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The dead salmon zoo: generalizing across architectures and datasets

    Sections 1-4 produced two "findings" on a single random architecture
    (BERT-mini) and a single dataset (IMDb sentiment). A reasonable
    skeptical move at this point: maybe what we're seeing is BERT-specific
    — something about self-attention, or about IMDb's particular
    distribution of words and lengths, that lets a random encoder pick up
    the label by accident.

    The paper's claim is much stronger than that. The artifact is supposed
    to be a property of **random computation in general**, not any one
    inductive bias. To test that, we run the same pipeline (random init →
    pooled embedding → 5-fold CV logistic-regression probe →
    random-computation null) across a small zoo of architectures crossed
    with two datasets. Same tokenization, same probe hyperparameters, same
    null-distribution machinery — only the random network in the middle
    changes.

    The zoo spans the standard inductive-bias menu:

    - **BERT-mini** — encoder transformer with bidirectional attention.
    - **GPT-2-mini** — decoder transformer with causal attention.
    - **bi-LSTM** — recurrent, no attention at all.
    - **1D ConvNet** — local n-gram filters, no recurrence or attention.
    - **mean-pool MLP** — no sequence inductive bias whatsoever (averages
      the embedding layer first, then runs an MLP on the bag-of-tokens).

    The two datasets:

    - **IMDb** — long movie reviews labelled positive vs. negative.
      Sentiment task.
    - **AG News** (binarized to *World* vs. *Sci/Tech*) — short news
      headlines + leads. Topic classification, not sentiment, with
      shorter and more topical inputs.

    For each (architecture, dataset) cell, the table reports observed
    probe accuracy, the random-computation null's mean and standard
    deviation across $N$ other random initializations, and **two**
    p-values: against chance (the conventional, wrong null) and against
    the random-computation null (the paper's fix).

    The pattern we expect, if the paper is right: $p$ vs. chance is
    $\ll 0.05$ in **every** cell — significant artifact everywhere — and
    $p$ vs. random computation is approximately uniform on $[0, 1]$,
    failing to reject in basically every cell, because the "observed"
    run is itself just another sample from that null.
    """)
    return


@app.cell
def _(mo):
    zoo_archs_ui = mo.ui.multiselect(
        options=["BERT", "GPT-2", "LSTM", "Conv1D", "MLP"],
        value=["BERT", "GPT-2", "LSTM", "MLP"],
        label="Architectures",
    )
    zoo_datasets_ui = mo.ui.multiselect(
        options=["IMDb", "AG News"],
        value=["IMDb", "AG News"],
        label="Datasets",
    )
    zoo_n_null_ui = mo.ui.slider(
        start=3, stop=20, step=1, value=8,
        label="Null seeds per cell", show_value=True,
    )
    zoo_n_ui = mo.ui.slider(
        start=80, stop=300, step=20, value=120,
        label="Sentences per run", show_value=True,
    )
    mo.vstack([
        mo.md(
            "**Zoo controls.** Each (architecture × dataset × seed) is "
            "one full random-init + embed + probe run, cached. The first "
            "pass is slow (one forward pass + one probe per cell × seed); "
            "reruns and slider drags are instant once cached. Defaults are "
            "tuned to be tolerable in WASM."
        ),
        mo.hstack([zoo_archs_ui, zoo_datasets_ui]),
        mo.hstack([zoo_n_null_ui, zoo_n_ui]),
    ])
    return zoo_archs_ui, zoo_datasets_ui, zoo_n_null_ui, zoo_n_ui


@app.cell
def _(torch):
    # Custom random architectures for the zoo. All take BERT-tokenized
    # input (input_ids, attention_mask) and return a pooled (B, D)
    # embedding. The shared interface lets us swap any of them in for the
    # random BERT in section 1's pipeline without changing the downstream
    # probe.

    class RandomLSTM(torch.nn.Module):
        def __init__(self, vocab_size, hidden_size, num_layers=2):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden_size)
            self.lstm = torch.nn.LSTM(
                hidden_size, hidden_size,
                num_layers=num_layers, bidirectional=True, batch_first=True,
            )

        def forward(self, input_ids, attention_mask):
            x = self.embed(input_ids)
            out, _ = self.lstm(x)
            mask = attention_mask.unsqueeze(-1).float()
            return (out * mask).sum(1) / mask.sum(1).clamp_min(1)

    class RandomConv1D(torch.nn.Module):
        def __init__(self, vocab_size, hidden_size, num_layers=3):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden_size)
            blocks = []
            for _ in range(num_layers):
                blocks.append(torch.nn.Conv1d(
                    hidden_size, hidden_size, kernel_size=3, padding=1,
                ))
                blocks.append(torch.nn.GELU())
            self.conv = torch.nn.Sequential(*blocks)

        def forward(self, input_ids, attention_mask):
            x = self.embed(input_ids).transpose(1, 2)
            x = self.conv(x).transpose(1, 2)
            mask = attention_mask.unsqueeze(-1).float()
            return (x * mask).sum(1) / mask.sum(1).clamp_min(1)

    class RandomMLP(torch.nn.Module):
        # Mean-pools the token embeddings FIRST, then runs an MLP. So
        # the MLP has no per-token sequence information at all — it sees
        # only the bag-of-tokens average. The "no inductive bias" control:
        # any artifact this network produces comes from random projections
        # of bag-of-tokens, nothing more.
        def __init__(self, vocab_size, hidden_size, num_layers=3):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden_size)
            blocks = []
            for _ in range(num_layers):
                blocks.append(torch.nn.Linear(hidden_size, hidden_size))
                blocks.append(torch.nn.GELU())
            self.mlp = torch.nn.Sequential(*blocks)

        def forward(self, input_ids, attention_mask):
            x = self.embed(input_ids)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(1) / mask.sum(1).clamp_min(1)
            return self.mlp(pooled)

    return RandomConv1D, RandomLSTM, RandomMLP


@app.cell
def _(
    LogisticRegression,
    Pipeline,
    RandomConv1D,
    RandomLSTM,
    RandomMLP,
    StandardScaler,
    StratifiedKFold,
    mo,
):
    # One function: build a random model of the given architecture, embed
    # `n` balanced sentences from the given dataset, return the 5-fold CV
    # accuracy of a logistic-regression probe. Cached on every argument,
    # so dragging a slider only ever recomputes the cells that changed.
    # Imports live inside the cached functions because mo.cache does not
    # reliably close over classes captured at cell scope.

    @mo.cache
    def _zoo_load_balanced(dataset_name: str, n: int):
        from datasets import load_dataset

        if dataset_name == "IMDb":
            ds = load_dataset("imdb", split="train", streaming=False)
            pos_label, neg_label = 1, 0
        elif dataset_name == "AG News":
            # AG News labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech.
            # Binarize to World vs. Sci/Tech for a topic task that's
            # clearly distinct from IMDb's sentiment task.
            ds = load_dataset("ag_news", split="train", streaming=False)
            pos_label, neg_label = 3, 0
        else:
            raise ValueError(dataset_name)

        half = n // 2
        pos_texts, neg_texts = [], []
        for ex in ds:
            if ex["label"] == pos_label and len(pos_texts) < half:
                pos_texts.append(ex["text"])
            elif ex["label"] == neg_label and len(neg_texts) < half:
                neg_texts.append(ex["text"])
            if len(pos_texts) >= half and len(neg_texts) >= half:
                break
        texts = tuple(pos_texts + neg_texts)
        labels = tuple([1] * len(pos_texts) + [0] * len(neg_texts))
        return texts, labels

    @mo.cache
    def zoo_seed_accuracy(
        arch: str, dataset_name: str, seed: int, n: int, log10_C: float,
    ):
        import numpy as _np
        import torch as _torch
        from transformers import (
            AutoModel, AutoTokenizer, BertConfig, GPT2Config,
        )

        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        vocab_size = tok.vocab_size
        H = 256

        _torch.manual_seed(seed)
        if arch == "BERT":
            cfg = BertConfig(
                vocab_size=vocab_size, hidden_size=H,
                num_hidden_layers=4, num_attention_heads=4,
                intermediate_size=H * 4,
            )
            model = AutoModel.from_config(cfg).eval()
            arch_kind = "hf"
        elif arch == "GPT-2":
            cfg = GPT2Config(
                vocab_size=vocab_size, n_embd=H, n_layer=4, n_head=4,
                n_positions=512,
            )
            model = AutoModel.from_config(cfg).eval()
            arch_kind = "hf"
        elif arch == "LSTM":
            model = RandomLSTM(vocab_size, H).eval()
            arch_kind = "custom"
        elif arch == "Conv1D":
            model = RandomConv1D(vocab_size, H).eval()
            arch_kind = "custom"
        elif arch == "MLP":
            model = RandomMLP(vocab_size, H).eval()
            arch_kind = "custom"
        else:
            raise ValueError(arch)

        texts, labels = _zoo_load_balanced(dataset_name, n)
        labels_arr = _np.array(labels)

        batches = []
        with _torch.no_grad():
            for _s in range(0, len(texts), 32):
                _batch = list(texts[_s : _s + 32])
                enc = tok(
                    _batch, padding=True, truncation=True,
                    max_length=256, return_tensors="pt",
                )
                if arch_kind == "hf":
                    out = model(**enc)
                    hidden = out.last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    pooled = (
                        (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                    )
                else:
                    pooled = model(
                        enc["input_ids"], enc["attention_mask"],
                    )
                batches.append(pooled.cpu().numpy())
        X_s = _np.concatenate(batches, axis=0)

        C = float(10 ** log10_C)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accs = _np.empty(5)
        for _i, (_tr, _te) in enumerate(skf.split(X_s, labels_arr)):
            _clf = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(
                    C=C, max_iter=2000, solver="liblinear",
                )),
            ])
            _clf.fit(X_s[_tr], labels_arr[_tr])
            accs[_i] = _clf.score(X_s[_te], labels_arr[_te])
        return float(accs.mean())

    return (zoo_seed_accuracy,)


@app.cell
def _(
    log10_C_ui,
    mo,
    np,
    stats,
    zoo_archs_ui,
    zoo_datasets_ui,
    zoo_n_null_ui,
    zoo_n_ui,
    zoo_seed_accuracy,
):
    import pandas as _pd

    _archs = list(zoo_archs_ui.value)
    _datasets = list(zoo_datasets_ui.value)
    _N_null = int(zoo_n_null_ui.value)
    _n_per = int(zoo_n_ui.value)
    _logC = float(log10_C_ui.value)

    # Seed convention: 0 is the "observed" run, 1..N are the null seeds.
    # Held constant across (arch, dataset) so we never compare cells on
    # different seed sets.
    _all_seeds = list(range(_N_null + 1))

    # One progress bar over the whole zoo. With mo.cache, only the cells
    # whose args changed actually do work.
    _items = [
        (_a, _d, _s)
        for _a in _archs for _d in _datasets for _s in _all_seeds
    ]
    zoo_results = {}
    for _a, _d, _s in mo.status.progress_bar(
        _items,
        title="Zoo: random-init × dataset × seed",
        subtitle=(
            f"{len(_archs)} archs × {len(_datasets)} datasets × "
            f"{len(_all_seeds)} seeds"
        ),
        remove_on_exit=True,
    ):
        zoo_results[(_a, _d, _s)] = zoo_seed_accuracy(
            _a, _d, _s, _n_per, _logC,
        )

    _rows = []
    zoo_nulls = {}
    zoo_obs = {}
    for _a in _archs:
        for _d in _datasets:
            _obs = zoo_results[(_a, _d, 0)]
            _null = np.array([
                zoo_results[(_a, _d, _s)]
                for _s in range(1, _N_null + 1)
            ])
            _n_exc = int((_null >= _obs).sum())
            _p_rc = (1 + _n_exc) / (1 + len(_null))

            _n_correct = int(round(_obs * _n_per))
            _p_ch = stats.binomtest(
                _n_correct, _n_per, p=0.5, alternative="two-sided",
            ).pvalue

            zoo_nulls[(_a, _d)] = _null
            zoo_obs[(_a, _d)] = _obs
            _rows.append({
                "Architecture": _a,
                "Dataset": _d,
                "Observed acc": _obs,
                "Null mean": float(_null.mean()),
                "Null sd": (
                    float(_null.std(ddof=1)) if len(_null) > 1 else 0.0
                ),
                "p (vs chance)": _p_ch,
                "p (vs random comp)": _p_rc,
            })

    zoo_df = _pd.DataFrame(_rows)
    return zoo_df, zoo_nulls, zoo_obs


@app.cell
def _(plt, zoo_df):
    # Matplotlib table with color-coded p-value cells. The contrast we
    # want to surface: every "p vs chance" cell red (significant against
    # the wrong null), every "p vs random comp" cell green
    # (insignificant against the right null) — the paper's fix at a
    # glance. Two-line headers + explicit colWidths so labels fit cleanly.
    _src_cols = list(zoo_df.columns)
    _header_labels = {
        "Architecture": "Architecture",
        "Dataset": "Dataset",
        "Observed acc": "Observed\naccuracy",
        "Null mean": "Null\nmean",
        "Null sd": "Null\nstd. dev.",
        "p (vs chance)": "$p$ vs.\nchance",
        "p (vs random comp)": "$p$ vs. random\ncomputation",
    }
    _col_labels = [_header_labels[_c] for _c in _src_cols]
    _col_widths = [0.13, 0.10, 0.13, 0.11, 0.11, 0.13, 0.16]

    _n_rows = len(zoo_df)
    _fig, _ax = plt.subplots(
        figsize=(11.5, 0.7 + 0.55 * (_n_rows + 1.4)),
    )
    _ax.axis("off")

    def _fmt_p(v):
        return f"{v:.1e}" if v < 1e-4 else f"{v:.3f}"

    _cell_text = []
    _cell_colors = []
    for _row in zoo_df.itertuples(index=False):
        _r = []
        _c = []
        for _col, _val in zip(_src_cols, _row):
            if _col in ("Architecture", "Dataset"):
                _r.append(str(_val))
                _c.append("#f7f7f7")
            elif _col == "Observed acc":
                _r.append(f"{_val:.3f}")
                # Color by margin above 0.5: stronger artifact = darker.
                _intensity = min(max((_val - 0.5) / 0.2, 0.0), 1.0)
                _c.append(plt.cm.Reds(0.18 + 0.5 * _intensity))
            elif _col in ("Null mean", "Null sd"):
                _r.append(f"{_val:.3f}")
                _c.append("#fbfbfb")
            elif _col == "p (vs chance)":
                _r.append(_fmt_p(_val))
                _c.append("#fcd5c4" if _val < 0.05 else "#f5f5f5")
            elif _col == "p (vs random comp)":
                _r.append(_fmt_p(_val))
                _c.append("#d4ecc8" if _val >= 0.05 else "#fcd5c4")
            else:
                _r.append(str(_val))
                _c.append("#f7f7f7")
        _cell_text.append(_r)
        _cell_colors.append(_c)

    _table = _ax.table(
        cellText=_cell_text,
        colLabels=_col_labels,
        cellColours=_cell_colors,
        colWidths=_col_widths,
        loc="center",
        cellLoc="center",
    )
    _table.auto_set_font_size(False)
    _table.set_fontsize(10)
    # Vertical scale: enough room for two-line headers and breathable rows.
    _table.scale(1.0, 1.9)

    # Style the header row: bold, dark fill, taller to fit two lines.
    for _j in range(len(_col_labels)):
        _hcell = _table[(0, _j)]
        _hcell.set_text_props(
            fontweight="bold", color="#222222", linespacing=1.15,
        )
        _hcell.set_facecolor("#e2e2e2")
        _hcell.set_height(_hcell.get_height() * 1.35)

    # Subtle cell borders so the heatmap colors don't run together.
    for _key, _cell in _table.get_celld().items():
        _cell.set_edgecolor("#bbbbbb")
        _cell.set_linewidth(0.6)

    _ax.set_title(
        "The dead salmon zoo  —  conventional vs. random-computation null",
        fontsize=13, pad=12, loc="center", fontweight="bold",
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(np, plt, zoo_archs_ui, zoo_datasets_ui, zoo_nulls, zoo_obs):
    # One small histogram per (architecture, dataset). Shared x-axis so
    # eyes can compare distributions across the grid. Blue line is the
    # observed seed; gray bars are the random-computation null. If the
    # blue line lands in the body of the gray distribution, the
    # "observed" run is just another dead salmon.
    _archs = list(zoo_archs_ui.value)
    _datasets = list(zoo_datasets_ui.value)
    _nrow = len(_archs)
    _ncol = len(_datasets)

    _fig, _axes = plt.subplots(
        _nrow, _ncol,
        figsize=(3.6 * _ncol, 2.0 * _nrow),
        sharex=True, sharey=True, squeeze=False,
    )

    _all_vals = np.concatenate(
        [v for v in zoo_nulls.values()]
        + [np.array(list(zoo_obs.values()))]
    )
    _xlo = float(min(0.45, _all_vals.min() - 0.02))
    _xhi = float(max(0.75, _all_vals.max() + 0.02))
    _bins = np.linspace(_xlo, _xhi, 16)

    for _i, _a in enumerate(_archs):
        for _j, _d in enumerate(_datasets):
            _ax = _axes[_i, _j]
            _null = zoo_nulls[(_a, _d)]
            _obs = zoo_obs[(_a, _d)]
            _ax.hist(
                _null, bins=_bins, color="#aaaaaa",
                edgecolor="white", alpha=0.85,
            )
            _ax.axvline(
                0.5, color="k", lw=0.8, linestyle="--", alpha=0.6,
            )
            _ax.axvline(_obs, color="#1f77b4", lw=2.0)
            _ax.set_xlim(_xlo, _xhi)
            if _i == 0:
                _ax.set_title(_d, fontsize=11)
            if _j == 0:
                _ax.set_ylabel(_a, fontsize=11, rotation=0,
                               ha="right", va="center", labelpad=24)
            if _i == _nrow - 1:
                _ax.set_xlabel("CV accuracy", fontsize=9)
            _ax.grid(True, axis="y", alpha=0.25)

    _fig.suptitle(
        "Random-computation null per (architecture × dataset).  "
        "Blue line = observed seed; dashed = chance.",
        fontsize=11,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What just happened.** The same dead-salmon dynamic appears in
    **every** cell of the table. Every random architecture clears chance
    by several standard errors, every conventional binomial test against
    $0.5$ rejects with $p \!\ll\! 0.05$, and the random-computation
    p-value is approximately uniform on $[0, 1]$ across cells —
    indistinguishable from what you'd get if the null were exactly
    correct (which it is, by construction).

    The artifact is not BERT-specific. It is not transformer-specific.
    It is not even attention-specific — the bi-LSTM and the 1D ConvNet
    show it just as cleanly. Even the **mean-pool MLP**, which has no
    sequence inductive bias at all and only sees the bag-of-tokens
    average, produces a probe that "works." Whatever is happening, it
    is a property of *random nonlinear projections of natural-language
    inputs*, not of any particular architectural choice.

    The dataset axis tells the same story. AG News (a topic task on
    short news leads) shows the artifact just as clearly as IMDb (a
    sentiment task on long movie reviews), even though the two tasks
    have nothing semantically in common. Whatever the random nets are
    picking up on — vocabulary, length, punctuation density — is
    something the labels of *both* datasets happen to correlate with,
    because the labels of any natural-language dataset correlate with
    those features.

    The histograms drive the same point home graphically. The blue
    "observed" line lands somewhere typical inside the gray
    distribution in every panel. There is no panel where the observed
    seed is anomalously good. The conventional probing pipeline screams
    "significant!" $|\text{archs}| \times |\text{datasets}|$ times in a
    row; the paper's fix correctly screams nothing in a row, exactly
    that many times.
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
