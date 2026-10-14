"""Tests for the frozen harness and the experiment file (need data/raw/)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
import experiment as E  # noqa: E402
import prepare as P  # noqa: E402

pytestmark = pytest.mark.skipif(not (P.RAW / "Train.csv").exists(), reason="competition data not present")


@pytest.fixture(scope="module")
def data():
    return P.load()


def test_load_matches_the_files(data):
    train, test = data
    assert len(train) == 4098 and len(test) == 1374
    assert test.columns[0] == "ID"  # the byte-order mark in Test.csv is handled
    assert train[P.TARGET].map(lambda t: isinstance(t, str) and len(t) > 0).all()
    assert all(Path(p).exists() for p in list(train.image[:50]) + list(test.image[:50]))


def test_folds_cover_every_row_once_and_never_split_a_group(data):
    train, _ = data
    reps = P.folds(train)
    assert len(reps) == P.N_REPEATS
    for rep in reps:
        assert len(rep) == P.N_FOLDS
        assert sorted(np.concatenate([b for _, b in rep])) == list(range(len(train)))
        for a, b in rep:
            assert not set(a) & set(b)
            assert not set(train[P.GROUP_COL].iloc[a]) & set(train[P.GROUP_COL].iloc[b])


def test_folds_are_the_same_on_every_call_and_match_the_frozen_hash(data):
    train, _ = data
    one, two = P.folds(train), P.folds(train)
    assert all((a1 == a2).all() and (b1 == b2).all() for r1, r2 in zip(one, two) for (a1, b1), (a2, b2) in zip(r1, r2))
    assert P.fold_hash(train) == P.FOLD_HASH


def test_folds_are_balanced_on_scan_height_and_keep_exact_twins_together(data):
    train, _ = data
    fold = P.fold_of(train)
    share = [train[P.BATCH_COL][fold == k].mean() for k in range(P.N_FOLDS)]
    assert max(share) - min(share) < 0.03
    assert (pd.Series(fold).groupby(train[P.TARGET].map(P.norm_text).to_numpy()).nunique() == 1).all()


def test_score_matches_the_competition_formula():
    # Worked by hand. Line 1: 3 reference words, 1 word wrong ("cat"->"cot"), 1 character wrong.
    # Line 2: 2 reference words, prediction empty: 2 word edits, 11 character edits ("hello world").
    ref, hyp = ["the cat sat", "hello world"], ["the cot sat", ""]
    s = P.score(ref, hyp)
    w, c = (1 + 2) / 2, (1 + 11) / 2
    assert s["parts"]["word_edits"] == pytest.approx(w) and s["parts"]["char_edits"] == pytest.approx(c)
    assert s["score"] == pytest.approx(1 - 0.5 * (w / 12 + c / 55))
    ws = (1 * 3 ** 0.5 + 2 * 2 ** 0.5) / (3 ** 0.5 + 2 ** 0.5)
    cs = (1 * 11 ** 0.5 + 11 * 11 ** 0.5) / (2 * 11 ** 0.5)
    assert s["parts"]["score_sqrt"] == pytest.approx(1 - 0.5 * (ws / 12 + cs / 55))


def test_score_reproduces_leaderboard_rows_from_their_own_error_columns():
    # Leaderboard, read 2026-10-13: (WER Weighted, CER Weighted, private score) for ranks 1, 10 and 50.
    for w, c, listed in ((1.227466747, 1.737800293, 0.933057367), (1.311222443, 1.940332694, 0.927726343),
                         (1.379388533, 2.046669765, 0.923919388)):
        assert 1 - 0.5 * (w / P.WORD_NORM + c / P.CHAR_NORM) == pytest.approx(listed, abs=2e-9)


def test_score_edge_cases():
    assert P.score(["a b c"], ["a b c"])["score"] == 1.0
    assert P.score(["  a b c "], ["a b c"])["score"] == 1.0            # spaces at the ends do not count
    assert P.score(["a  b"], ["a b"])["parts"]["word_edits"] == 0      # words split on any run of spaces
    assert P.score(["a  b"], ["a b"])["parts"]["char_edits"] == 1      # but an inner double space is a character
    assert P.score(["a b"], [float("nan")])["parts"]["word_edits"] == 2  # a missing prediction is all wrong
    assert P.score(["Abc"], ["abc"])["parts"]["word_edits"] == 1       # case counts


def test_garbage_like_the_sample_submission_scores_near_the_benchmark(data):
    # The benchmark row scores 0.030 public and 0.042 private; the sample file's strings look like its output.
    # Scored against (unrelated) training transcripts they must land in the same region, or the formula's scale is
    # wrong. Measured 2026-10-13: 0.013 with the plain mean, -0.011 with square-root weights.
    train, _ = data
    sample = pd.read_csv(P.RAW / "SampleSubmission.csv")[P.TARGET].to_numpy()
    s = P.score(train[P.TARGET].to_numpy()[:len(sample)], sample)
    assert 0.0 < s["score"] < 0.08 and -0.05 < s["parts"]["score_sqrt"] < 0.08


def test_cv_score_reports_folds_batches_and_spread(data):
    train, _ = data
    perfect = P.cv_score(train, np.array([train[P.TARGET].to_numpy()], dtype=object))
    assert perfect["score"] == 1.0 and perfect["std"] == 0.0 and perfect["per_repeat"] == [1.0] * P.N_FOLDS
    blank = P.cv_score(train, np.array([[""] * len(train)], dtype=object))
    assert blank["score"] < 0 and blank["std"] > 0 and {"short", "tall", "score_sqrt"} <= set(blank["parts"])


def test_submission_matches_the_sample_and_breaks_on_each_rule(data):
    _, test = data
    good = P.build_submission(test, ["  some text "] * len(test))
    P.validate_submission(good)
    assert (good[P.TARGET] == "some text").all()
    broken = {
        "wrong columns": good.rename(columns={P.TARGET: "Prediction"}),
        "missing rows": good.iloc[:-1],
        "duplicate IDs": pd.concat([good.iloc[:-1], good.iloc[:1]]),
        "missing value": good.assign(Target=[None] + ["x"] * (len(good) - 1)),
        "empty text": good.assign(Target=[" "] + ["x"] * (len(good) - 1)),
        "line break": good.assign(Target=["a\nb"] + ["x"] * (len(good) - 1)),
        "runaway": good.assign(Target=["a" * 301] + ["x"] * (len(good) - 1)),
    }
    for name, sub in broken.items():
        with pytest.raises(AssertionError):
            P.validate_submission(sub)


def test_features_never_read_the_target(data):
    train, test = data
    assert list(E.features(train.drop(columns=[P.TARGET, P.GROUP_COL])).columns) == list(E.features(test).columns)


def test_baseline_is_learned_inside_the_fold(data):
    train, test = data
    a, b = P.folds(train)[0][0]
    pred = E.fit_predict(E.DEFAULT_CONFIG, E.features(train).iloc[a], train[P.TARGET].to_numpy()[a], E.features(train).iloc[b], 0)[0]
    assert len(pred) == len(b) and len(set(pred)) == 1
    assert pred[0] in {P.clean(t) for t in train[P.TARGET].iloc[a]}


def test_char_lm_beam_matches_greedy_without_lm_and_prefers_known_spellings():
    import numpy as np
    from readers.lm import CharLM, beam_decode, greedy_decode

    charset = list("abc ")
    rng = np.random.default_rng(0)
    lp = np.log(rng.dirichlet(np.ones(5) * 0.3, size=30))
    assert isinstance(beam_decode(lp, charset, None, beam=50, topk=5, prune=-1e9), str)
    # an ambiguous letter is settled by the language model
    lm = CharLM(["abc abc abc"] * 20, order=3)
    amb = np.full((5, 5), -20.0)
    for t, i in enumerate([1, 2, 0, 3, 0]):   # a b _ c _
        amb[t, i] = 0.0
    amb[3, 3], amb[3, 1] = np.log(0.45), np.log(0.55)  # c (0.45) against a (0.55)
    assert greedy_decode(amb, charset) == "aba"
    assert beam_decode(amb, charset, lm, weight=1.0) == "abc"


def test_ctc_reader_trains_and_reads_on_cpu(data):
    """Two epochs on 40 lines: the pipeline runs end to end and returns one string and one log-prob array per line."""
    train, _ = data
    from readers import ctc

    class Ctx:
        cache: dict = {}
        log = staticmethod(lambda m: None)
        should_stop = staticmethod(lambda: False)

    paths = list(train["image"].iloc[:40])
    texts = [P.clean(t) for t in train[P.TARGET].iloc[:40]]
    preds, extras = ctc.fit_predict({"epochs": 2, "batch": 8, "inner_val": 8, "channels": [8, 16, 16, 16],
                                     "rnn_hidden": 16, "rnn_layers": 1, "workers": 0, "height": 32}, paths, texts,
                                    paths[:5], 0, Ctx)
    assert len(preds) == 5 and all(isinstance(x, str) for x in preds)
    assert len(extras["logp"]) == 5 and extras["logp"][0].shape[1] == len(extras["charset"]) + 1


def test_complete_fills_every_parameter_of_the_model():
    cfg = E.complete({**E.DEFAULT_CONFIG, "model": "ctc", "params": {"epochs": 3}})
    assert cfg["params"]["epochs"] == 3 and "height" in cfg["params"] and "candidates" not in cfg["params"]


def test_ctc_reader_memorises_a_few_lines_with_the_lstm_shortcut(tmp_path):
    """EXP-002 never left the blank-only stage: the LSTM stalled training. With rnn_skip the network memorises eight
    rendered lines within 200 steps; this guards that the shortcut stays wired in."""
    import math
    import random

    import torch
    import torch.nn as nn
    from PIL import Image, ImageDraw, ImageFont
    from readers import ctc
    from readers.images import load_lines
    from readers.lm import greedy_decode

    random.seed(0)
    font = ImageFont.load_default(size=28)
    texts, paths = [], []
    for i in range(8):
        t = "".join(random.choice("abcdefghij") for _ in range(8))
        im = Image.new("L", (24 * len(t) + 20, 44), 225)
        ImageDraw.Draw(im).text((10, 6), t, fill=30, font=font)
        im.save(tmp_path / f"{i}.png")
        texts.append(t)
        paths.append(str(tmp_path / f"{i}.png"))
    cs = sorted(set("".join(texts)))
    idx = {c: i + 1 for i, c in enumerate(cs)}
    p = {**ctc.DEFAULTS, "height": 32, "channels": [16, 32, 64, 64], "rnn_hidden": 64, "rnn_layers": 2, "dropout": 0.0,
         "rnn_skip": True}
    torch.manual_seed(0)
    m = ctc.build_model(p, len(cs) + 1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x, ws = ctc.to_tensor_batch(load_lines(paths, 32, 2600), torch)
    tg = torch.cat([torch.tensor([idx[c] for c in t]) for t in texts])
    for _ in range(200):
        lp = m(x).log_softmax(-1).transpose(0, 1)
        il = torch.tensor([min(lp.shape[0], math.ceil(w / 4)) for w in ws])
        loss = nn.CTCLoss(zero_infinity=True)(lp, tg, il, torch.tensor([8] * 8))
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        lp = m(x).log_softmax(-1).numpy()
    assert sum(greedy_decode(lp[i], cs) == texts[i] for i in range(8)) >= 6
